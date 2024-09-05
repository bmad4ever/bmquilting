from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass
from typing import TypeAlias
from enum import Enum
import numpy as np

num_pixels: TypeAlias = int
percentage: TypeAlias = float


class Orientation(Enum):
    H = "H"
    V = "V"
    H_AND_V = "H & V"


class UiCoordData:
    """
    Used to interrupt the generation and keep track of the number of placed patches.
    Index zero in shared memory should be set to a number above zero to interrupt the generation.
    The remaining indices are used to keep track of the task (e.g. number of patches in a generation).
    """

    def __init__(self, jobs_shm_name: str, job_id: int):
        self.jobs_shm_name = jobs_shm_name
        self.job_id = job_id
        self.__shm: SharedMemory | None = None

    @property
    def _shm(self):
        if self.__shm is None:
            self.__shm = SharedMemory(name=self.jobs_shm_name)
        return self.__shm

    def add_to_job_data_slot_and_check_interrupt(self, to_increment: int | np.uint32):
        shm_data_array = np.ndarray((2 + self.job_id,), dtype=np.dtype('uint32'), buffer=self._shm.buf)
        shm_data_array[1 + self.job_id] += to_increment
        return shm_data_array[0] > 0

    @staticmethod
    def get_number_of_jobs_for(parallelization_lvl: int, batch_size: int = 1) -> int:
        """
        Args:
            parallelization_lvl: the parallelization_lvl used for the generation
            batch_size: how many generations will run simultaneously
        Returns: the total number of jobs
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
        return (1 + n_jobs) * np.dtype('uint32').itemsize, n_jobs


@dataclass
class GenParams:
    """
    Data used across multiple quilting subroutines.
    Used in quilting.py and make_seamless.py
    """
    block_size: num_pixels
    overlap: num_pixels
    tolerance: percentage
    blend_into_patch: bool
    version: int

    @property
    def bo(self):
        return self.block_size, self.overlap

    @property
    def bot(self):
        return self.block_size, self.overlap, self.tolerance
