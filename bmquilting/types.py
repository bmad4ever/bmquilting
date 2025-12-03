from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass, field
from typing import TypeAlias
from enum import Enum
import numpy as np

from .misc.functions import FuncWrapper, LogScalingFunc, TwoNTS

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
class BlendConfig:
    sobel_kernel_size: num_pixels = 3
    min_blur_diameter: num_pixels = 3
    max_blur_diameter: num_pixels = 11

    use_vignette: bool = True  # TODO? could add vignette params here, but might be overkill

    blend_scale: float = 1  # TODO... likely to be removed

    blur_size_func: FuncWrapper = field(default_factory=LogScalingFunc)
    """
    Function used to remap the Normalized Gradient Differences (NGDs) computed around the seam.

    The remapped values remain within the interval [0, 1], as they are used to 
    interpolate between the minimum and maximum blur diameters. This remapping 
    introduces a non-linear relationship between the blur size and the NGDs.

    Note:
        NGDs are normalized with respect to the theoretical maximum possible value 
        for the given kernel size used to compute the gradients.

    Default Behavior:
        The `LogScalingFunc` is used with `gain=100` and `top=0.5`.

        - Setting `top=0.5` ensures the function reaches 1 (the maximum blur radius) 
        when the NGD value is half of its theoretical maximum.
        - Setting `gain=100` makes the function concave, steep at the start, meaning 
        the blur radius increases quickly for small gradients and more slowly as 
        it approaches the top.
    """

    blur_shape_func: FuncWrapper = field(default_factory=TwoNTS)
    """
    Function used to shape the transition curve when blending two patches (e.g., in a multi-patch surface).

    This function maps the **signed distance to the seam** to an **interpolation weight** 
    (ranging from 0 to 1) used for blending. It determines how the two patches transition 
    within the specified blur area, influencing the resulting smoothness or sharp change.

    Default Behavior: 
        The `TwoNTS` (Two-NormalizedTunableSigmoid) is used with `k=-0.5`.

        The resulting curve shape is composed of **two scaled sigmoid-like curves** that meet 
        smoothly near the seam (where the distance is zero).
        Visual Analogy: The transition resembles two gentle "hills" meeting at their bases, 
        providing a very smooth, Gaussian-kernel-like transition rather than a simple linear ramp. 
        This makes the blending effect strongest *at* the seam and rapidly decreases away from it.
    """

    adaptive_maximum_filter_number_of_levels: int = 3
    """
    This parameter sets the **number of iterations (or levels)** with different kernel sizes 
        that the adaptive maximum filter executes when computing the blur diameters.

    Context:
    1.  Blur Diameter Calculation: When blending a seam, the initial blur diameter is determined 
        based on the **gradient differences** computed near the seam.
    2.  Diameter Propagation: These initial diameters need to be **propagated** across the texture to 
        determine the correct blend amount for every pixel (i.e., if a pixel's distance to the seam is 
        less than the propagated radius, it needs blending).
    3.  Adaptive Filter: An **adaptive maximum filter** is used for this propagation/expansion. 
        This filter runs with different kernel sizes across multiple iterations to ensure that locally 
        lower blur diameters aren't overshadowed by higher values during expansion.

    Note on Quantization:
        The final quantization interval for the diameters is determined by the *min_blur_diameter* 
        and the *maximum diameter found* in the propagation process (it is **not** based on the 
        theoretical possible maximum defined by `max_blur_diameter`).
    """

    def __post_init__(self):
        if self.sobel_kernel_size % 2 == 0:
            raise ValueError(f"{self.sobel_kernel_size=} is invalid, kernel size should be an odd number.")

        if self.max_blur_diameter < self.min_blur_diameter:
            raise ValueError(f"Invalid range: {self.min_blur_diameter=}"
                             f" must be less or equal to {self.max_blur_diameter=}")


@dataclass
class GenParams:
    """
    Data used across multiple quilting subroutines.
    Used in quilting.py and make_seamless.py
    """
    block_size: num_pixels
    overlap: num_pixels
    tolerance: percentage
    blend_config: BlendConfig | None
    vignette_on_match_template: bool  # whether to use the blending vignette as a mask when searching for a matching patch
    version: int

    @property
    def blend_into_patch(self) -> bool:
        return self.blend_config is not None

    @property
    def bo(self):
        return self.block_size, self.overlap

    @property
    def bot(self):
        return self.block_size, self.overlap, self.tolerance

