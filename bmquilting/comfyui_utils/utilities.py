from torch import Tensor
import numpy as np
import torch


def unwrap_latent(image: Tensor, is_latent: bool) -> tuple[np.ndarray, bool]:
    """
    Args:
        image: a single image or latent image
        is_latent: if true, indicates the image is a latent image
    Returns: unwrapped image as a numpy array and whether it was squeezed or not
    """
    squeeze = len(image.shape) > 3
    image = image.cpu().numpy()
    image = image.squeeze() if squeeze else image
    image = np.moveaxis(image, 0, -1) if is_latent else image
    return image, squeeze


def wrap_as_latent(image: np.ndarray, is_latent: bool, unsqueeze: bool) -> Tensor:
    """
    Args:
        image: image or latent image as a numpy array
        is_latent: if true, image represents a latent image, and this must be taken into account
        unsqueeze: whether to unqueeze the image (if it was squeezed, it must unsqueeze)
    Returns: image wrapped as a pytorch latent

    """
    image = np.moveaxis(image, -1, 0) if is_latent else image
    image = torch.from_numpy(image)
    image = image.unsqueeze(0) if unsqueeze else image
    return image


def unwrap_latent_and_quilt(wrapped_func, image: Tensor, job_id: int = 0, is_latent: bool = False) -> Tensor:
    """
    quilting job when using batches
    Args:
        wrapped_func: class that wraps the job to execute. Must receive the image and job_id as inputs.
        image: single image in tensor format
        job_id: job_id used by uicd to keep track of the job completion
        is_latent: if true, indicates if the provided image is a latent image
    Returns:
    """
    image, squeeze = unwrap_latent(image, is_latent)
    result = wrapped_func(image, job_id)
    return wrap_as_latent(result, is_latent, squeeze)


def unwrap_latent_and_quilt_seamless(wrapped_func, image: Tensor, lookup: Tensor,
                                     job_id: int, is_latent: bool = False) -> Tensor:
    """
    seamless quilting job when using batches, similar to unwrap_latent_and_quilt, but requires the lookup
    """
    image, squeeze = unwrap_latent(image, is_latent)
    if lookup is not None:
        lookup, _ = unwrap_latent(lookup, is_latent)
    result = wrapped_func(image, lookup, job_id)
    return wrap_as_latent(result, is_latent, squeeze)


def overlap_percentage_to_pixels(block_size: int, overlap: float):
    """ Detail: in case of extreme values, clips to 1 or block_size-1."""
    return np.clip(round(block_size * overlap), 1, block_size - 1)
