"""
    Utilities to sync a GUI with the generation, or to interrupt the generation via a GUI.
"""

from multiprocessing.shared_memory import SharedMemory
import numpy as np
import functools
import threading
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DTYPE = np.dtype('uint32')
"""dtype used for the shared memory array"""

ON_THREAD_JOIN_TIMEOUT = 2.0
"""timeout used when joining a registered thread on JobMemoryManager exit"""


class UiCoordData:
    """
    Manages job progress and interruption signals via SharedMemory.

    The shared memory buffer is expected to be a ``uint32`` array where:
    * Index 0: Interrupt signal (value > 0 triggers interruption).
    * Index 1+: Job data slots for progress tracking.

    :param jobs_shm_name: The unique name of the shared memory block.
    :param job_id: The index offset for this specific job's data slot.
    """

    def __init__(self, jobs_shm_name: str, job_id: int):
        self.jobs_shm_name: str = jobs_shm_name
        self.job_id: int = job_id
        self._shm: SharedMemory | None = None
        self._view: np.ndarray | None = None

    def _connect(self) -> None:
        """Connects to shared memory and creates a persistent numpy view."""
        if self._shm is None:
            self._shm = SharedMemory(name=self.jobs_shm_name)
            # Create a view of the entire buffer for efficiency
            self._view = np.ndarray(
                (2 + self.job_id,),
                dtype=DTYPE,
                buffer=self._shm.buf
            )

    def raise_or_add(self, value: int = 0) -> None:
        """
        Raises JobInterrupted if an interrupt signal is detected.
        Otherwise, adds count to the progress tracker shared memory array.

        :raises JobInterrupted: If the UI has signaled to stop.
        """
        self._connect()

        if self._view[0] > 0:
            raise JobInterrupted()

        self._view[1 + self.job_id] += np.uint32(value)

    def request_interrupt(self) -> None:
        """
        Sets the interrupt signal for this shared memory block.
        Call this from the UI thread/process to stop the worker.
        """
        self._connect()
        self._view[0] = np.uint32(1)

    def close(self) -> None:
        """Closes the shared memory connection."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._view = None

    @staticmethod
    def get_number_of_jobs_for(parallelization_lvl: int, batch_size: int = 1) -> int:
        """
        :param parallelization_lvl: the parallelization_lvl used for the generation
        :param batch_size: how many generations will run simultaneously
        :return: the total number of jobs
        """
        if batch_size > 1:
            n_jobs = batch_size
            n_jobs *= 1 if parallelization_lvl == 0 else 4
        else:
            n_jobs = 1 if parallelization_lvl == 0 else 4
            n_jobs *= parallelization_lvl if parallelization_lvl > 0 else 1
        return n_jobs

    @staticmethod
    def get_required_shm_size_and_number_of_jobs(parallelization_lvl: int, batch_size: int = 1) -> tuple[int, int]:
        n_jobs = UiCoordData.get_number_of_jobs_for(parallelization_lvl, batch_size)
        return (1 + n_jobs) * np.dtype(DTYPE).itemsize, n_jobs


class JobMemoryManager:
    def __init__(self, num_jobs: int):
        self.num_jobs = num_jobs
        self._shm: SharedMemory | None = None
        self._thread: threading.Thread | None = None

    @property
    def name(self):
        return self._shm.name if self._shm else None

    def get_progress(self) -> int:
        """Reads current progress of all jobs from memory."""
        view = np.ndarray((1 + self.num_jobs,), dtype=DTYPE, buffer=self._shm.buf)
        return np.sum(view[1:])

    def stop_all(self):
        """Sets the interrupt flag at index 0."""
        view = np.ndarray((1,), dtype=DTYPE, buffer=self._shm.buf)
        view[0] = 1

    def register_thread(self, thread: threading.Thread):
        """Register the UI monitoring thread so we can join it on exit."""
        self._thread = thread

    def is_interrupted(self) -> bool:
        """Check the shared signal. Use this in the thread loop."""
        if self._shm is None:
            return True
        view = np.ndarray((1,), dtype=DTYPE, buffer=self._shm.buf)
        return view[0] > np.uint32(0)

    def create(self) -> str:
        size = (1 + self.num_jobs) * DTYPE.itemsize
        self._shm = SharedMemory(create=True, size=size)

        # Initialize Memory: [Interrupt(0), Job1(0), Job2(0)...]
        view = np.ndarray((1 + self.num_jobs,), dtype=DTYPE, buffer=self._shm.buf)
        view[:] = 0

        return self._shm.name

    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._shm:
            # Set the global kill-switch
            view = np.ndarray((1,), dtype=DTYPE, buffer=self._shm.buf)
            view[0] = 1

            # Ensure the UI thread sees the signal and stops
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=ON_THREAD_JOIN_TIMEOUT)

                if self._thread.is_alive():
                    logger.warning("Monitoring thread failed to exit after timeout. "
                                   "Potential resource leak!")

            # Close & Unlink shared memory
            self._shm.close()
            try:
                self._shm.unlink()
                logger.info("UICD-JMM: shared memory was unlinked.")
            except FileNotFoundError:
                pass


class JobInterrupted(Exception):
    """Custom exception to signal a UI interrupt."""
    pass


def handle_ui_interrupts(return_on_cancel: any = "INTERRUPTED", auto_close: bool = True, re_raise: bool = False):
    """
    Decorator for job functions that handle UI interruptions via UiCoordData.

    :param return_on_cancel: The value to return if a JobInterrupted exception is caught.
    :param auto_close: If True, automatically calls .close() on the 'uicd' object found in arguments.
    :param re_raise: Raise JobInterrupted again after closing uicd.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Identify the uicd object if it exists in args or kwargs
            uicd: UiCoordData | None = kwargs.get('uicd')
            if uicd is None:
                for arg in args:
                    if isinstance(arg, UiCoordData):
                        uicd = arg
                        break

            # Run the function, catch interrupts, and close uicd shm connection
            try:
                return func(*args, **kwargs)
            except JobInterrupted:
                logger.info("HUI: JobInterrupted catched")
                if re_raise:
                    logger.info("HUI: Re-Raise JobInterrupted")
                    raise JobInterrupted()
                return return_on_cancel
            finally:
                if auto_close and uicd is not None:
                    uicd.close()


        return wrapper

    return decorator


def check_ui(uicd: UiCoordData | None, to_add: int = 0):
    """
    Helper to be called inside functions.
    Keeps the code readable and raises the JobInterrupt when an interrupt is triggered.
    """
    if uicd is not None:
        uicd.raise_or_add(to_add)
