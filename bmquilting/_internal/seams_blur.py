from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from functools import cache
import numpy as np
import cv2

from ..utils.functions import FuncWrapper, LogScalingFunc, TwoNTS
from .common import NumPixels


USE_SCHARR_WHEN_KSIZE_EQUALS_3 = True
"""When cv2.Sobel is used, use ksize=cv2.FILTER_SCHARR if the provided kernel size is equal to 3."""


def sum_squared_differences(dgx: np.ndarray, dgy: np.ndarray, dgxy: np.ndarray, out: np.ndarray=None) -> np.ndarray:
    """to be used with gradients' differences"""
    return np.einsum('ijkl,ijkl->jk', dgxy, dgxy, out=out)


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

    grad_diff_func: Callable = sum_squared_differences
    """
    Computes a seam's "errors", i.e. how much the seam is noticeable, with respect to the gradients differences.
    
    The function receives the following numpy arrays as input:
        - dgx: (H, W, C) already computed x gradients difference, per channel; view into dgxy.
        - dgy: (H, W, C) already computed y gradients difference, per channel; view into dgxy.
        - dgxy: (2, H, W, C) -> dgxy[0] = dgx; dgxy[1] = dgy; may be used instead of dgx and dgy.
        - out: (H, W) pre-allocated output array; can be used for intermediate computations.
    
    \nDefault behavior:
        Sum of Squared Differences (SSD), similar to L2 norm without the square root overhead.
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
    def auto_blend_config_2(cls, block_size: NumPixels, overlap: NumPixels, use_vignette: bool = False) -> BlendConfig:
        """
        Creates a BlendConfig by heuristically determining an adequate sobel kernel size and blur diameter bounds
        based on the provided inputs.

        :param block_size: The length of one of the patch's edges.
        :param overlap: The size of the overlapping area between patches.
        :param use_vignette: Whether to apply a vignette-like mask for additional smoothing.
        :return: A BlendConfig instance with calculated blur bounds.
        """
        sobel_ksize = min(round(overlap / 2.0), round(block_size / 10.0))
        sobel_ksize |= 1

        return cls.auto_blend_config_1(
            use_vignette=use_vignette,
            sobel_kernel_size=sobel_ksize,
            overlap=overlap
        )

    @classmethod
    def auto_blend_config_1(cls, sobel_kernel_size: NumPixels, overlap: NumPixels,
                            use_vignette: bool = False) -> BlendConfig:
        """
        Creates a BlendConfig by heuristically determining the blur diameter bounds
        based on the provided inputs.

        :param sobel_kernel_size: The kernel size used for gradient computation.
        :param overlap: The size of the overlapping area between patches.
        :param use_vignette: Whether to apply a vignette-like mask for additional smoothing.
        :return: A BlendConfig instance with calculated blur bounds.
        """

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


def __make__get_max_possible_gradient_diff():
    _cache = {}
    def _get_max_possible_gradient_diff(sobel_ksize: int, num_channels: int, norm_func: Callable) -> float:
        key = (sobel_ksize, num_channels)  # norm_func IGNORED when caching
        if key not in _cache:
            _cache[key] = ___get_max_possible_gradient_diff(sobel_ksize, num_channels, norm_func)
        return _cache[key]
    _get_max_possible_gradient_diff.cache_clear = _cache.clear
    return _get_max_possible_gradient_diff

_get_max_possible_gradient_diff = __make__get_max_possible_gradient_diff()  # cache value ignoring the passed function


def ___get_max_possible_gradient_diff(sobel_ksize: int, num_channels: int, norm_func: Callable) -> float:
    """
    Calculate max possible gradient difference by simulating a maximal gradient
    tensor and passing it through the user-defined norm function.

    :param sobel_ksize: Sobel kernel size
    :param num_channels: Number of color channels
    :param norm_func: A function that takes (grad_x, grad_y, grad_xy) and returns the "error" map
    """
    max_pixel_value = 1.0   # supposes float32

    # Get actual OpenCV kernel to determine scale
    if USE_SCHARR_WHEN_KSIZE_EQUALS_3 and sobel_ksize == 3:
        sobel_ksize = cv2.FILTER_SCHARR
    kx, ky = cv2.getDerivKernels(1, 0, ksize=sobel_ksize, normalize=False)
    kernel_2d = np.outer(kx, ky)

    sobel_scale = np.sum(np.abs(kernel_2d)) / 2     # maximum response
    K = sobel_scale * max_pixel_value               # maximum gradient in one direction
    max_diff_val = 2.0 * K                          # maximum possible difference for any single component (dx or dy)

    # Dummy tensors representing the maximum possible delta
    dummy_dgxy = np.full((2, 1, 1, num_channels), max_diff_val)
    dummy_dgx = dummy_dgxy[0]
    dummy_dgy = dummy_dgxy[1]

    return float(norm_func(dummy_dgx, dummy_dgy, dummy_dgxy))


@cache
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
                               radii_limiter: np.ndarray | None = None,
                               radii_binary_limiter: np.ndarray | None = None
                               ) -> np.ndarray:
    """
    Create adaptive blend mask with transition width based on gradient differences.

    :param tdiff_map: Seam gradient difference map (higher values = more difficult transition) of shape (H, W).
    :param mc_mask_overlap: Min-cut mask (0 = source side, 1 = patch side) of shape (H, W).
    :param blend_config: Params set for the generation, such as the minimum blur diameter.
    :param radii_limiter: (H, W) shaped numpy array that limits the maximum possible blur radius; applied pre adaptive max filter.
    :param radii_binary_limiter: (H, W) binary mask for the blur radii; applied post adaptive max filter,
    the boundaries of the limiter are not softened.
    :param dtype: Data type of source images (for calculating theoretical max)

    :return: np.ndarray (H, W): Blend mask (0 = source, 1 = patch)
    """
    min_blur_diameter, max_blur_diameter = blend_config.min_blur_diameter, blend_config.max_blur_diameter

    # Calculate theoretical maximum (cached)
    max_gradient_diff = _get_max_possible_gradient_diff(
        sobel_ksize= blend_config.sobel_kernel_size,
        num_channels=tdiff_map.shape[-1] if tdiff_map.ndim > 2 else 1,
        norm_func=blend_config.grad_diff_func)

    # Normalize & Adjust w/ blur_size_func
    tdiff_norm = np.divide(tdiff_map, max_gradient_diff, out=tdiff_map)

    # adjusts values instead of using a linear relationship
    # the function in blend_config SHOULD clip the final values in the [0, 1] range
    blend_config.blur_size_func.inplace_func(tdiff_norm)

    # Map tdiff values to blend widths
    blend_diameters = np.multiply(max_blur_diameter - min_blur_diameter, tdiff_norm, out=tdiff_norm)
    blend_diameters += min_blur_diameter
    max_blur_diameter_found = round(np.max(blend_diameters))
    blend_radii = np.divide(blend_diameters, 2, out=blend_diameters)

    effective_min_blur = min_blur_diameter
    if radii_limiter is not None:
        np.minimum(radii_limiter, blend_radii, out=blend_radii)
        effective_min_blur = 1.e-8

    if max_blur_diameter_found > effective_min_blur:  # if they are equal then there is nothing to dilate
        sigma = (min_blur_diameter + 1)/6
        blend_radii = adaptive_maximum_filter(
            radius_map=blend_radii,
            n_levels=blend_config.adaptive_maximum_filter_number_of_levels,
            max_radius=round(max_blur_diameter_found/2),
            min_radius=effective_min_blur,
            overreach=min_blur_diameter//2
        )
        cv2.GaussianBlur(blend_radii, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=blend_radii)

    if radii_binary_limiter is not None:
        blend_radii *= radii_binary_limiter

    np.maximum(blend_radii, 1.e-8, out=blend_radii)

    # Calculate signed distance from transition line
    src_zeroes = np.empty(mc_mask_overlap.shape, dtype=np.uint8)
    np.greater(mc_mask_overlap, 0, out=src_zeroes, casting='unsafe')
    dist_to_source = cv2.distanceTransform(src_zeroes, cv2.DIST_L2, cv2.DIST_MASK_3, dstType=cv2.CV_32F)  # values within patch
    pth_zeroes = cv2.bitwise_xor(src_zeroes, 1, dst=src_zeroes)
    dist_to_patch = cv2.distanceTransform(pth_zeroes, cv2.DIST_L2, cv2.DIST_MASK_3, dstType=cv2.CV_32F) # values outside patch
    signed_distance = np.subtract(dist_to_source, dist_to_patch, out=dist_to_source) # to seam; values outside patch are negative
    cv2.GaussianBlur(signed_distance, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=signed_distance)

    # Create smooth transition using sigmoid function
    t = signed_distance / blend_radii  # min should be min_diameter, never zero
    cv2.GaussianBlur(t, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=t)

    # compute blend_mask in place
    # input and output clipping should be handled by the function
    blend_config.blur_shape_func.inplace_func(t)

    return t  # blend_mask



@cache
def _get_buffers_for_graddiffs_computation(shape):
    """
    :param shape: (H, W, C)
    :return:
        dgxy (2, H, W, C): where gradients differences should be stored
        dgx (H, W, C): dgxy[0]
        dgy (H, W, C): dgxy[1]
        tg1 (H, W, C): auxiliary array to temporary store gradients
        tg2 (H, W, C): auxiliary array to temporary store gradients
        ts1 (H, W): auxiliary array to store 2D data
        ts2 (H, W): auxiliary array to store 2D data
    """
    _dtype = np.float32

    # Compute flat sizes
    s_dgxy = 2 * shape[0] * shape[1] * shape[2]  # (2, H, W, C)
    s_tg12 = s_dgxy  # (2, H, W, C)
    s_ts12 = 2 * shape[0] * shape[1]  # (2, H, W)

    total = s_dgxy + s_tg12 + s_ts12

    # Single contiguous allocation
    buf = np.empty(total, dtype=_dtype)

    # Carve out views — no copies, all contiguous
    offset = 0
    dgxy = buf[offset: offset + s_dgxy].reshape(2, *shape)
    offset += s_dgxy
    tg12 = buf[offset: offset + s_tg12].reshape(2, *shape)
    offset += s_tg12
    ts12 = buf[offset: offset + s_ts12].reshape(2, *shape[:2])
    offset += s_ts12

    dgx, dgy = dgxy[0], dgxy[1]
    tg1, tg2 = tg12[0], tg12[1]
    ts1, ts2 = ts12[0], ts12[1]

    return dgxy, dgx, dgy, tg12, tg1, tg2, ts1, ts2


def gradients_differences_at_the_seam(
        sobel_ksize: int, cut_mask_overlap: np.ndarray,
        source_overlap: np.ndarray, patch_overlap: np.ndarray, patched_overlap: np.ndarray,
        grad_diff_func: Callable,
        ) -> np.ndarray:
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

    :return: a 2D array with shape equal to grad_shape[:2] that contains the gradients differences around the seam
    """
    if USE_SCHARR_WHEN_KSIZE_EQUALS_3 and sobel_ksize == 3:
        sobel_ksize = cv2.FILTER_SCHARR

    # Setup types and shape
    _dtype, ddepth = np.float32, cv2.CV_32F

    # Get cached arrays (allocates on 1st execution)
    dgxy, dgx, dgy, tg12, tg1, tg2, ts1, ts2 =  _get_buffers_for_graddiffs_computation(patched_overlap.shape)

    # Compute gradients for patched_overlap
    cv2.Sobel(patched_overlap, ddepth, 1, 0, dst=tg1, ksize=sobel_ksize)    # tg1 -> ∇x(patched)
    cv2.Sobel(patched_overlap, ddepth, 0, 1, dst=tg2, ksize=sobel_ksize)    # tg2 -> ∇y(patched)

    # Compute norm(∇source - ∇patched)
    cv2.Sobel(source_overlap, ddepth, 1, 0, dst=dgx, ksize=sobel_ksize)
    cv2.Sobel(source_overlap, ddepth, 0, 1, dst=dgy, ksize=sobel_ksize)
    dgxy -= tg12
    diffs_source = grad_diff_func(dgx, dgy, dgxy, out=ts1)

    # Filter values near seam for source diffs
    inv_mask  = np.subtract(1, cut_mask_overlap, out=ts2)     # ts2-> 1 - mask  (inv. mask is no longer needed afterwards)
    tdiff_map = np.multiply(diffs_source, inv_mask, out=ts1)  # ts1-> tdiff_map (partial)

    # Compute norm(∇patch - ∇patched)
    cv2.Sobel(patch_overlap, ddepth, 1, 0, dst=dgx, ksize=sobel_ksize)
    cv2.Sobel(patch_overlap, ddepth, 0, 1, dst=dgy, ksize=sobel_ksize)
    dgxy -= tg12
    diffs_patch = grad_diff_func(dgx, dgy, dgxy, out=ts2)

    # Filter values near seam for patch diffs
    np.multiply(diffs_patch, cut_mask_overlap, out=diffs_patch)
    np.maximum(tdiff_map, diffs_patch, out=tdiff_map)

    return tdiff_map


@cache
def _get_radii_limiter(block_size: NumPixels, overlap: NumPixels) -> np.ndarray:
    """Blur radii limiter for standard square patches -> [[1, 2, ..., 2, 1], ...]"""
    x = np.linspace(1, overlap, overlap).reshape((1, overlap))
    x[:, -(overlap // 2):] -= overlap + 1
    x[:, -(overlap // 2):] *= -1
    return x.repeat(block_size, axis=0)
