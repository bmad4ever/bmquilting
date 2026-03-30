from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from functools import lru_cache
import numpy as np
import cv2

from ..utils.functions import FuncWrapper, LogScalingFunc, TwoNTS
from .common import NumPixels


USE_SCHARR_WHEN_KSIZE_EQUALS_3 = True
"""When cv2.Sobel is used, use ksize=cv2.FILTER_SCHARR if the provided kernel size is equal to 3."""


@dataclass(frozen=True, slots=True)
class BlendConfig:
    sobel_kernel_size: NumPixels = 3
    min_blur_diameter: NumPixels = 3
    max_blur_diameter: NumPixels = 11

    use_vignette: bool = True
    """
    Applies a fade to the patch as its content approaches its shape edges (not the seams).

    This effect can be used alongside seams to soften transitions, or with seams disabled to achieve pure feathering 
    between patches. While effective at masking color or value discrepancies, 
    it introduces a blurring effect that make it unsuitable for textures containing 
    fine, sharp details, such as text, where clarity is required.
    """

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

    grad_diff_func: Callable = np.amax
    """
    Reduces multi-channel gradient differences into a single-channel intensity map.

    When processing sources with multiple channels (e.g., RGB), this function 
    aggregates the values across the channel dimension. The resulting 2D matrix 
    represents the local gradient intensity; higher values indicate a more 
    pronounced blur effect at that specific coordinate.

    Default behavior:
        Uses `numpy.max` to select the maximum gradient difference across channels for each pixel/element.

    Notes:
        `numpy.max` can be substituted with `numpy.mean` or other reduction functions depending on the desired sensitivity.
        The output map is normalized against the theoretical maximum possible gradient difference. 
        The selected function must have the `out` parameter.
    """

    use_blur_radii_limiter: bool = True
    """
    Attempts to mitigate seam blurring artifacts AFTER seam computation.
    This is done by limiting the seam blur radius with respect to its proximity to the overlapping area edges.
    This helps prevent seam artifacts near the edges that could visually give away the generation grid layout.
    """

    def __post_init__(self):
        if self.sobel_kernel_size % 2 == 0:
            raise ValueError(f"{self.sobel_kernel_size = } is invalid, kernel size should be an odd number.")

        if self.max_blur_diameter < self.min_blur_diameter:
            raise ValueError(f"Invalid range: {self.min_blur_diameter = }"
                             f" must be less or equal to {self.max_blur_diameter = }")

    @classmethod
    def auto_blend_config_2(cls, sobel_kernel_size: NumPixels, overlap: NumPixels,
                            use_vignette: bool = False) -> BlendConfig:

        DEFAULT_MAX_BLEND_RATIO = 0.95
        MAX_TO_SOBEL_FACTOR = 1.75

        def _min_blur_diam(sobel_ksize: NumPixels) -> int:
            return int(round(np.sqrt(sobel_ksize) + 1))

        def _max_blur_diam(overlap_size, sobel_ksize):
            max_blend_width = round(sobel_ksize * MAX_TO_SOBEL_FACTOR)

            # Cap it based on patch size if provided
            if overlap_size is not None:
                max_allowed = int(overlap_size * DEFAULT_MAX_BLEND_RATIO)
                max_blend_width = min(max_blend_width, max_allowed)
            return max_blend_width

        return BlendConfig(
            use_vignette=use_vignette,
            sobel_kernel_size=sobel_kernel_size,
            min_blur_diameter=_min_blur_diam(sobel_kernel_size),
            max_blur_diameter=_max_blur_diam(overlap, sobel_kernel_size)
        )

    @classmethod
    def auto_blend_config_1(cls, block_size: NumPixels, overlap: NumPixels, use_vignette: bool = False) -> BlendConfig:
        sobel_ksize = min(round(overlap / 2.0), round(block_size / 10.0))
        if sobel_ksize % 2 == 0:
            sobel_ksize += 1

        return cls.auto_blend_config_2(
            use_vignette=use_vignette,
            sobel_kernel_size=sobel_ksize,
            overlap=overlap
        )


@lru_cache(maxsize=1)
def _get_max_possible_gradient_diff(dtype_str: str, sobel_ksize: int) -> float:
    """
    Calculate maximum possible gradient difference using actual OpenCV kernel.
    Cached to avoid recomputing for the same dtype and kernel size.

    :param dtype_str: String representation of numpy dtype (e.g., 'uint8', 'float32').
        String is used instead of dtype object because dtype objects aren't hashable.
    :param sobel_ksize: Sobel kernel size
    :return: Maximum possible value gradient difference
    """
    # Convert string back to dtype
    dtype = np.dtype(dtype_str)

    if dtype == np.uint8:
        max_pixel_value = 255.0
    elif dtype in [np.float32, np.float64]:
        max_pixel_value = 1.0
    else:
        max_pixel_value = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0

    # Get actual OpenCV kernel to determine scale
    if USE_SCHARR_WHEN_KSIZE_EQUALS_3 and sobel_ksize == 3:
        sobel_ksize = cv2.FILTER_SCHARR
    kx, ky = cv2.getDerivKernels(1, 0, ksize=sobel_ksize, normalize=False)
    kernel_2d = np.outer(kx, ky)

    # Maximum response is sum of absolute values
    sobel_scale = np.sum(np.abs(kernel_2d)) / 2

    # Maximum gradient in one direction
    K = sobel_scale * max_pixel_value

    # Maximum gradient vector magnitude: sqrt(gx² + gy²)
    max_gradient_magnitude = np.sqrt(2) * K

    # Maximum difference when gradients point in opposite directions
    max_diff = 2 * max_gradient_magnitude

    return max_diff


@lru_cache(maxsize=None)
def _circular_kernel(radius: int) -> np.ndarray:
    """Cached circular structuring element."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius + 1, 2*radius + 1))


def adaptive_maximum_filter(
    radius_map: np.ndarray,
    n_levels: int = 6,
    gamma: float = 1.0,
    overreach: int = 0,
    min_radius: float | None = None,
    max_radius: float | None = None,
) -> np.ndarray:
    """
    Perform spatially adaptive morphological dilation
    (a variable-radius maximum filter).

    Each pixel expands according to its local radius value,
    producing a smooth, radius-guided maximum field.

    :param radius_map: 2D map of per-pixel radii.
    :param n_levels: Number of quantization levels.
    :param gamma: Nonlinear exponent shaping the spacing of quantized radii.
    :param overreach: Optional additive dilation to slightly extend each radius band.
    :param min_radius: Lower bound for quantization. Inferred from data if None.
    :param max_radius: Upper bound for quantization. Inferred from data if None.
    :return: Smoothed / dilated radius map.
    """
    radius_map = np.asarray(radius_map, np.float32)
    if min_radius is None:
        min_radius = float(radius_map.min())
    if max_radius is None:
        max_radius = float(radius_map.max())

    # Nonlinear quantization of radius levels
    steps = np.linspace(0, 1, n_levels + 1)[1:]
    radii = min_radius + (max_radius - min_radius) * (steps ** gamma)
    radii = np.unique(np.ceil(radii).astype(int))

    # Preallocate
    h, w = radius_map.shape
    result = np.full((h, w), min_radius, np.float32)
    processed_mask = np.zeros((h, w), np.bool_)  # tracks done pixels
    temp_mask = np.zeros_like(processed_mask)

    for r in radii:
        # Determine current active pixels: not processed yet & <= current radius
        np.less_equal(radius_map, float(r), out=temp_mask)
        active_mask = np.logical_and(temp_mask, ~processed_mask)

        # Build masked layer (float for dilation)
        layer = np.where(active_mask, radius_map, 0).astype(np.float32)

        # Perform dilation (local maximum) with circular kernel
        kernel = _circular_kernel(r + overreach)
        cv2.dilate(layer, kernel, dst=layer)

        # Merge results
        np.maximum(result, layer, out=result)

        # Mark these pixels as processed
        processed_mask |= temp_mask

    return result


def create_adaptive_blend_mask(tdiff_map: np.ndarray, mc_mask_overlap: np.ndarray,
                               blend_config: BlendConfig,
                               radii_limiter_mask: np.ndarray | None = None,
                               radii_limiter: np.ndarray | None = None,
                               dtype=np.uint8) -> np.ndarray:
    """
    Create adaptive blend mask with transition width based on gradient differences.

    :param tdiff_map: Seam gradient difference map (higher values = more difficult transition) of shape (H, W).
    :param mc_mask_overlap: Min-cut mask (0 = source side, 1 = patch side) of shape (H, W).
    :param blend_config: Params set for the generation, such as the minimum blur diameter.
    :param radii_limiter_mask: Binary mask of shape (H, W) used to compute radii_limiter.
        Should match overlapping area in most use cases.
    :param radii_limiter: (H, W) shaped numpy array that limits the maximum possible blur radius.
        Will be ignored if radii_limiter_mask is provided.
    :param dtype: Data type of source images (for calculating theoretical max)

    :return: np.ndarray (H, W): Blend mask (0 = source, 1 = patch)
    """
    min_blur_diameter, max_blur_diameter = blend_config.min_blur_diameter, blend_config.max_blur_diameter

    # Calculate theoretical maximum (cached)
    max_gradient_diff = _get_max_possible_gradient_diff(dtype.name, blend_config.sobel_kernel_size)

    # Normalize & map to func
    tdiff_norm = tdiff_map / max_gradient_diff  # normalize

    # adjusts values instead of using a linear relationship
    # the function in blend_config SHOULD clip the final values in the [0, 1] range
    blend_config.blur_size_func.inplace_func(tdiff_norm)

    # Map tdiff values to blend widths
    blend_diameters = min_blur_diameter + (max_blur_diameter - min_blur_diameter) * tdiff_norm
    max_blur_diameter_found = round(np.max(blend_diameters))
    blend_radii = np.divide(blend_diameters, 2, out=blend_diameters)

    # Limit radii with respect to the mask boundaries, to avoid having near or out bounds blur
    if radii_limiter_mask is not None:
        # create a radii limiter from a mask (usually should correspond to the overlapping area)
        radii_limiter = radii_limiter_mask.astype(np.uint8, copy=True)
        radii_limiter = cv2.distanceTransform(radii_limiter.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)

    if radii_limiter is not None:
        np.minimum(radii_limiter, blend_radii, out=blend_radii)
        blend_radii[blend_radii <= 0] = .001  # can't use zero here due to division later

    if max_blur_diameter_found > min_blur_diameter:  # if they are equal then there is nothing to dilate
        sigma = (min_blur_diameter + 1)/6
        blend_radii = adaptive_maximum_filter(
            radius_map=blend_radii,
            n_levels=blend_config.adaptive_maximum_filter_number_of_levels,
            max_radius=round(max_blur_diameter_found/2),
            overreach=min_blur_diameter//2  # ksize 10
        )
        cv2.GaussianBlur(blend_radii, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=blend_radii)

    # Calculate signed distance from transition line
    # dev note: compared 3 vs cv2.DIST_MASK_PRECISE
    #   both performed well in terms of speed and precision so far
    #   doesn't seem like something that needs to be further tested
    #   as any minor details will be blured out.
    #   I will choose simples-fastest for now
    dist_from_source = cv2.distanceTransform((mc_mask_overlap > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_from_patch = cv2.distanceTransform((mc_mask_overlap <= 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_from_patch -= dist_from_source  # compute in place
    signed_distance = dist_from_patch.astype(np.float32)
    cv2.GaussianBlur(signed_distance, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=signed_distance)

    # Create smooth transition using sigmoid function
    t = signed_distance / blend_radii  # min should be min_diameter, never zero
    cv2.GaussianBlur(t, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=t)

    # compute blend_mask in place
    # input and output clipping should be handled by the function
    blend_config.blur_shape_func.inplace_func(t)
    blend_mask = t

    np.subtract(1, blend_mask, out=blend_mask)
    return blend_mask


def gradients_differences_at_the_seam(
        sobel_ksize: int, cut_mask_overlap: np.ndarray,
        source_overlap: np.ndarray, patch_overlap: np.ndarray, patched_overlap: np.ndarray,
        grand_diff_func: Callable,
        _tmp: np.ndarray=None) -> np.ndarray:
    """
    This function does the following, albeit minimizing mem. allocations (thus being harder to read).
    1. Get the gradients for all provided textures (source, patch, and patched)
    2. Compute the difference between source & patched, and patch & patched gradients
    3. Get the highest difference from both and mask them to only get values near the cut.

    :param sobel_ksize: ksize argument when using cv2.Sobel function.
        If equal to three is replaced with cv2.FILTER_SCHARR.
    :param cut_mask_overlap: cut mask view of the area where source and patch overlap
    :param source_overlap: source view of the area which overlaps with the patch
    :param patch_overlap: patch view of the area which overlaps with the source
    :param patched_overlap: patched view of the area where source and patch overlap

    :param _tmp: If there is a patch shaped array available with no use,
    it can be re-used here to avoid an additional allocation.
    Provided there is no use for the patched_overlap array after this function is called,
    it can be passed here to slightly optimize memory usage.

    :return: a 2D array with shape equal to grad_shape[:2] that contains the gradients differences around the seam
    """
    assert (source_overlap.dtype == np.float32)
    if USE_SCHARR_WHEN_KSIZE_EQUALS_3 and sobel_ksize == 3:
        sobel_ksize = cv2.FILTER_SCHARR

    # Setup types and shape
    grad_shape = patched_overlap.shape
    _dtype, ddepth = np.float32, cv2.CV_32F

    # Allocate gradient arrays once (reuse for different images)
    gx = np.empty(grad_shape, dtype=_dtype)
    gy = np.empty(grad_shape, dtype=_dtype)

    # Compute gradients for patched_overlap
    cv2.Sobel(patched_overlap, ddepth, 1, 0, dst=gx, ksize=sobel_ksize)
    cv2.Sobel(patched_overlap, ddepth, 0, 1, dst=gy, ksize=sobel_ksize)

    # Pre-allocate diff array (will reuse)
    diff_source = np.empty(grad_shape, dtype=_dtype) if _tmp is None else _tmp

    # Compute diff_source in-place
    # diff_source = sqrt((gx_patched - gx_source)² + (gy_patched - gy_source)²)
    cv2.Sobel(source_overlap, ddepth, 1, 0, dst=diff_source, ksize=sobel_ksize)
    diff_source -= gx  # diff_source now has (gx_patched - gx_source)

    # Temporary for gy difference
    temp_gy = np.empty(grad_shape, dtype=_dtype)
    cv2.Sobel(source_overlap, ddepth, 0, 1, dst=temp_gy, ksize=sobel_ksize)
    temp_gy -= gy  # (gy_patched - gy_source)
    cv2.magnitude(diff_source, temp_gy, magnitude=diff_source)

    # Compute diff_patch (reuse temp_gy)
    diff_patch = temp_gy  # Reuse the allocation
    cv2.Sobel(patch_overlap, ddepth, 1, 0, dst=diff_patch, ksize=sobel_ksize)
    diff_patch -= gx  # (gx_patched - gx_patch)

    # Reuse gx for temporary gy computation
    cv2.Sobel(patch_overlap, ddepth, 0, 1, dst=gx, ksize=sobel_ksize)
    gx -= gy  # (gy_patched - gy_patch)
    cv2.magnitude(diff_patch, gx, magnitude=diff_patch)

    # Take maximum across channels
    if diff_source.ndim == 3:
        max_diffs_source = grand_diff_func(diff_source, axis=2, out=diff_source[:, :, 0])
        max_diffs_patch = grand_diff_func(diff_patch, axis=2, out=diff_patch[:, :, 0])
    else:
        # suppose shape == 2
        max_diffs_source = diff_source
        max_diffs_patch = diff_patch

    # Compute "transition diff. map" in-place using max_diffs_source as output buffer
    tdiff_map = max_diffs_source  # Reuse allocation
    np.multiply(max_diffs_source, 1 - cut_mask_overlap, out=tdiff_map)

    # Use max_diffs_patch as temporary
    np.multiply(max_diffs_patch, cut_mask_overlap, out=max_diffs_patch)
    np.maximum(tdiff_map, max_diffs_patch, out=tdiff_map)

    return tdiff_map


@lru_cache(maxsize=1)
def _get_radii_limiter(block_size: NumPixels, overlap: NumPixels) -> np.ndarray:
    """Blur radii limiter for standard square patches -> [[1, 2, ..., 2, 1], ...]"""
    x = np.linspace(1, overlap, overlap).reshape((1, overlap))
    x[:, -(overlap // 2):] -= overlap + 1
    x[:, -(overlap // 2):] *= -1
    return x.repeat(block_size, axis=0)
