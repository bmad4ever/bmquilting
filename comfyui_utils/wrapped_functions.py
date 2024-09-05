from joblib import Parallel, delayed
from dataclasses import dataclass
import cv2

from ..make_seamless import make_seamless_horizontally, make_seamless_vertically, make_seamless_both
from ..make_seamless2 import seamless_horizontal, seamless_vertical, seamless_both
from ..misc.validation_utils import validate_gen_args, validate_seamless_args
from ..generate import generate_texture, generate_texture_parallel
from ..types import GenParams, UiCoordData, Orientation
from .utilities import *


@dataclass
class QuiltingFuncWrapper:
    """Wraps node functionality for easy re-use when using jobs."""
    is_latent: bool
    block_sizes: list[int]
    overlap: float
    out_h: int
    out_w: int
    tolerance: float
    parallelization_lvl: int
    rng: np.random.Generator
    blend_into_patch: bool
    version: int
    jobs_shm_name: str

    def __call__(self, image, job_id):
        if not self.is_latent and self.version == 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

        block_size = self.block_sizes[job_id]
        overlap = overlap_percentage_to_pixels(block_size, self.overlap)
        gen_args = GenParams(block_size, overlap, self.tolerance, self.blend_into_patch, self.version)
        validate_gen_args(image, gen_args)

        if self.parallelization_lvl == 0:
            result = generate_texture(image, gen_args, self.out_h, self.out_w,
                                      self.rng, UiCoordData(self.jobs_shm_name, job_id))
        else:
            result = generate_texture_parallel(
                image, gen_args, self.out_h, self.out_w,
                self.parallelization_lvl, self.rng, UiCoordData(self.jobs_shm_name, job_id))

        if not self.is_latent and self.version == 2:
            result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

        return result


@dataclass
class SeamlessFuncWrapper:
    """Wraps node functionality for easy re-use when using jobs."""
    is_latent: bool
    ori: Orientation  # SEAMLESS_DIRS
    overlap: int
    rng: np.random.Generator
    blend_into_patch: bool
    version: int
    jobs_shm_name: str


def batch_using_jobs(wrapped_func: QuiltingFuncWrapper, src: Tensor, is_latent: bool) -> Tensor:
    """
    Args:
        wrapped_func: wrapper that contains the data and function to run
        src: tensor with the images or latent images
        is_latent: send true if src contains latent images
    Returns: a Tensor containing all the processed images (or latent images)
    """
    results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
        delayed(unwrap_latent_and_quilt)(wrapped_func, src[i], i, is_latent) for i in range(src.shape[0]))
    return torch.stack(results)


# endregion     quilting wrappers


@dataclass
class SeamlessMPFuncWrapper(SeamlessFuncWrapper):
    block_sizes: list[int]
    tolerance: float

    def __call__(self, image, lookup, job_id):
        block_size = self.block_sizes[min(job_id, len(self.block_sizes) - 1)]
        overlap = overlap_percentage_to_pixels(block_size, self.overlap)
        gen_args = GenParams(block_size, overlap, self.tolerance, self.blend_into_patch, self.version)

        if not self.is_latent and self.version == 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            if lookup is not None:
                lookup = cv2.cvtColor(lookup, cv2.COLOR_RGB2Lab) if lookup is not None else None

        match self.ori:
            case Orientation.H:
                func = make_seamless_horizontally
            case Orientation.V:
                func = make_seamless_vertically
            case ___:
                func = make_seamless_both

        validate_seamless_args(self.ori, image, lookup, block_size, overlap)
        result = func(image, gen_args, self.rng, lookup, UiCoordData(self.jobs_shm_name, job_id))

        if not self.is_latent and self.version == 2:
            result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

        return result


@dataclass
class SeamlessSPFuncWrapper(SeamlessFuncWrapper):
    block_sizes: list[int]

    def __call__(self, image, lookup, job_id):

        block_size = self.block_sizes[min(job_id, len(self.block_sizes) - 1)]
        overlap = overlap_percentage_to_pixels(block_size, self.overlap)
        gen_args = GenParams(block_size, overlap, 0, self.blend_into_patch, self.version)

        if not self.is_latent and self.version == 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            if lookup is not None:
                lookup = cv2.cvtColor(lookup, cv2.COLOR_RGB2Lab) if lookup is not None else None

        match self.ori:
            case Orientation.H:
                func = seamless_horizontal
            case Orientation.V:
                func = seamless_vertical
            case ___:
                func = seamless_both

        validate_seamless_args(self.ori, image, lookup, block_size, overlap)
        result = func(image, lookup, gen_args, self.rng, UiCoordData(self.jobs_shm_name, job_id))

        if not self.is_latent and self.version == 2:
            result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

        return result


def batch_seamless_using_jobs(wrapped_func, src, lookup, is_latent: bool = False):
    # process in the same fashion as lists
    number_of_processes = src.shape[0] if lookup is None else max(src.shape[0], lookup.shape[0])
    results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
        delayed(unwrap_latent_and_quilt_seamless)(
            wrapped_func,
            src[min(i, src.shape[0] - 1)],
            lookup[min(i, lookup.shape[0] - 1)]
            if lookup is not None else None,
            i, is_latent) for i in range(number_of_processes))
    return torch.stack(results)
