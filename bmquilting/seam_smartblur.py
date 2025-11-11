import numpy as np
import cv2
from functools import lru_cache
from math import ceil

from .misc.dry import apply_mask
from .types import UiCoordData, GenParams, BlendConfig, num_pixels

import traceback  # for debug purposes

DEFAULT_MAX_BLEND_RATIO = 0.95  # percentage of the overlap size that can be used for blending purposes
DEFAULT_BLEND_SCALE = 1.0

USE_SCHAAR_WHEN_KSIZE_EQUALS_3 = True


def debug_resize(arr, factor=8):
    return cv2.resize(arr, (arr.shape[1] * factor, arr.shape[0] * factor))


def auto_blend_config_2(sobel_kernel_size: num_pixels, overlap: num_pixels,
                        blend_scale: float = DEFAULT_BLEND_SCALE, use_vignette: bool = False) -> BlendConfig:
    return BlendConfig(
        blend_scale=blend_scale,
        use_vignette=use_vignette,
        sobel_kernel_size=sobel_kernel_size,
        min_blur_diameter=auto_min_blend_size(sobel_kernel_size),
        max_blur_diameter=auto_max_blend_diameter(overlap, sobel_kernel_size)
    )


def auto_blend_config_1(block_size: num_pixels, overlap: num_pixels,
                        blend_scale: float = DEFAULT_BLEND_SCALE, use_vignette: bool = False) -> BlendConfig:
    return auto_blend_config_2(
        blend_scale=blend_scale,
        use_vignette=use_vignette,
        sobel_kernel_size=min(ceil(overlap / 2.0), ceil(block_size // 5 / 2.0)),
        overlap=overlap
    )


@lru_cache(maxsize=1)
def get_max_possible_gradient_diff(dtype_str, sobel_ksize) -> float:
    """
    Calculate maximum possible gradient difference using actual OpenCV kernel.
    Cached to avoid recomputing for the same dtype and kernel size.

    Parameters:
    -----------
    dtype_str : str
        String representation of numpy dtype (e.g., 'uint8', 'float32')
        We use string instead of dtype object because dtype objects aren't hashable
    sobel_ksize : int
        Sobel kernel size

    Returns:
    --------
    float : Maximum possible value gradient difference
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
    if USE_SCHAAR_WHEN_KSIZE_EQUALS_3 and sobel_ksize == 3:
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
def circular_kernel(radius: int) -> np.ndarray:
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

    Parameters
    ----------
    radius_map : np.ndarray
        2D map of per-pixel radii.
    n_levels : int
        Number of quantization levels.
    gamma : float
        Nonlinear exponent shaping the spacing of quantized radii.
    overreach : int
        Optional additive dilation to slightly extend each radius band.
    min_radius, max_radius : float, optional
        Bounds for quantization. Inferred from data if None.

    Returns
    -------
    np.ndarray
        Smoothed / dilated radius map.
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
        kernel = circular_kernel(r + overreach)
        cv2.dilate(layer, kernel, dst=layer)

        # Merge results
        np.maximum(result, layer, out=result)

        # Mark these pixels as processed
        processed_mask |= temp_mask

    return result


def create_adaptive_blend_mask(tdiff_map, mcp_mask_overlap, sobel_ksize=3,
                               blend_scale=1.0,
                               min_blur_diameter=None,
                               max_blur_diameter=None,
                               overlap_size=None,
                               max_blend_ratio=DEFAULT_MAX_BLEND_RATIO,
                               dtype=np.uint8):
    """
    Create adaptive blend mask with transition width based on gradient differences.

    Parameters:
    -----------
    tdiff_map : np.ndarray (H, W)
        Seam gradient difference map (higher values = more difficult transition)
    mcp_mask_overlap : np.ndarray (H, W)
        Min-cut mask (0 = source side, 1 = patch side)
    sobel_ksize : int
        Sobel kernel size used to compute tdiff_map
    blend_scale : float
        Exponent for mapping tdiff_map to blend widths (higher = more aggressive)
    min_blend_width : int or None
        Minimum blend width in pixels. If None, equals sobel_ksize
    max_blend_width : int or None
        Maximum blend width in pixels. If None, scales with sobel_ksize (1.75x)
        and is capped by max_blend_ratio of overlap_size
    overlap_size : int or None
        Side length of the square patch. Used to cap max_blend_width
    max_blend_ratio : float
        Maximum blend width as a fraction of patch size (default: 0.2 = 20%)
    dtype : numpy dtype
        Data type of source images (for calculating theoretical max)

    Returns:
    --------
    np.ndarray (H, W): Blend mask (0 = source, 1 = patch)
    """
    # Auto-scale blend widths based on Sobel kernel size
    # Technically they could made be independent, but this is done in order to keep things simple
    if min_blur_diameter is None:
        min_blur_diameter = auto_min_blend_size(sobel_ksize)

    if max_blur_diameter is None:
        max_blur_diameter = _auto_max_blur_diameter(max_blend_ratio, overlap_size, sobel_ksize)

    # Ensure max > min
    if max_blur_diameter <= min_blur_diameter:
        max_blur_diameter = min_blur_diameter + 1

    # Calculate theoretical maximum (cached)
    # Convert dtype to string for hashable cache key
    dtype_str = np.dtype(dtype).name
    max_gradient_diff = get_max_possible_gradient_diff(dtype_str, sobel_ksize)

    # Normalize from 0 to theoretical maximum
    #tdiff_norm = np.clip(tdiff_map / max_gradient_diff, 0, 1, dtype=tdiff_map.dtype)
    # --- 2. Normalize tdiff map (robustly) ---
    print(f"max_blend_w = {max_blur_diameter}")
    # normalize & map to func
    # these params are somewhat arbitrary; eyeballing ( <*>__<*>)
    gain = 100.0  # steeper response curve
    top = 1.0 / 2.0  # 1/3 would be when the diffs sum in all channels equals the range

    tdiff_norm = tdiff_map / max_gradient_diff
    # compute the following: np.clip(np.log1p(tdiff_norm * gain) / np.log1p(gain * top), 0, 1)
    tdiff_norm *= gain
    np.log1p(tdiff_norm, out=tdiff_norm)
    tdiff_norm /= np.log1p(gain * top)
    np.clip(tdiff_norm, 0, 1, out=tdiff_norm)

    # -- DEBUG --
    cv2.imshow("_tdiff", debug_resize(tdiff_norm / max(1e-8, np.max(tdiff_norm))))

    # Map normalized tdiff values to blend widths
    blend_diameters = min_blur_diameter + (max_blur_diameter - min_blur_diameter) * (tdiff_norm ** blend_scale)
    max_blur_diameter_found = round(np.max(blend_diameters))
    blend_radii = np.divide(blend_diameters, 2, out=blend_diameters)

    print(f"min diam, max diam, overlap, max diam found = {
    min_blur_diameter, max_blur_diameter, mcp_mask_overlap.shape[1], max_blur_diameter_found}")

    #print(f"hey!!!!! {max_blur_diameter_found, min_blur_diameter}")
    if max_blur_diameter_found > min_blur_diameter:  # if they are equal then there is nothing to dilate
        sigma = (min_blur_diameter + 1)/6
        blend_radii = adaptive_maximum_filter(
            radius_map=blend_radii,
            n_levels=3,  # TODO -> add to blend config if used in final solution!
            max_radius=round(max_blur_diameter_found/2),
            overreach=min_blur_diameter//2  # ksize 10
        )
        cv2.GaussianBlur(blend_radii, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=blend_radii)

    cv2.imshow("norm radii layered max-filt.",
               debug_resize(blend_radii / (max_blur_diameter / 2))
               )

    mcp_binary = mcp_mask_overlap

    # Calculate signed distance from transition line
    # dev note: compared 3 vs cv2.DIST_MASK_PRECISE
    #   both performed well in terms of speed and precision so far
    #   doesn't seem like something that needs to be further tested
    #   as any minor details will be blured out.
    #   I will choose simples-fastest for now
    dist_from_source = cv2.distanceTransform((mcp_binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_from_patch = cv2.distanceTransform((mcp_binary <= 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_from_patch -= dist_from_source  # compute in place
    signed_distance = dist_from_patch.astype(np.float32)
    cv2.GaussianBlur(signed_distance, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=signed_distance)

    # Create smooth transition using sigmoid function
    t = signed_distance / blend_radii  # min should be min_diameter, never zero
    cv2.GaussianBlur(t, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=t)
    #np.clip(t, -8, 8, out=t)  # required to prevent overflows
    blend_mask = 1.0 / (1.0 + np.exp(-5 * t))  # sigmoid func.

    cv2.imshow("t", debug_resize(np.abs(t) / max(np.max(np.abs(t)), 1e-8)))
    cv2.imshow("blend_mask", debug_resize(blend_mask))
    #cv2.waitKey()

    return blend_mask


def _auto_max_blur_diameter(max_blend_ratio, overlap_size, sobel_ksize):
    # Start with 1.75x the kernel size (rounded)
    max_blend_width = round(sobel_ksize * 1.75)

    print(f"HEY > {max_blend_width}")

    # Cap it based on patch size if provided
    if overlap_size is not None:
        max_allowed = int(overlap_size * max_blend_ratio)
        max_blend_width = min(max_blend_width, max_allowed)
        print(f"HEY 2 > {max_allowed, max_blend_width}")
    return max_blend_width


def auto_max_blend_diameter(overlap_size, sobel_ksize):
    return _auto_max_blur_diameter(DEFAULT_MAX_BLEND_RATIO, overlap_size, sobel_ksize)


def auto_min_blend_size(sobel_ksize):
    return round(np.sqrt(sobel_ksize) + 1)
    #return sobel_ksize // 3 * 2 + 1


def compute_adaptive_blend_mask(source: np.ndarray, patch: np.ndarray, cut_mask: np.ndarray,
                                gen_args: GenParams,
                                #overlap: int, sobel_ksize: int = 3,
                                debug: bool = False):
    """
    Compute adaptive blend mask based on gradient differences.
    Optimized for performance with pre-allocated arrays and minimal allocations.

    @param source: existing texture, to the LEFT of the patch. (rotate/flip the block for other orientations)
        The source should not have been rolled to match the overlap section on the final patched block.
    @param patch:  the patch extending the block to the RIGHT. (rotate/flip the block for other orientations)
    @param cut_mask: the mask used to produce the final patched block.
    """
    assert source.shape[0] == source.shape[1], "Source must be square"
    block_size = gen_args.block_size  # = source.shape[0]
    overlap = gen_args.overlap
    sobel_ksize = gen_args.blend_config.sobel_kernel_size
    if USE_SCHAAR_WHEN_KSIZE_EQUALS_3:
        sobel_ksize = cv2.FILTER_SCHARR

    # Setup output dimensions
    dims = np.copy(source.shape)
    dims[0] = block_size
    dims[1] = block_size * 2 - overlap

    new_img = np.zeros(dims, dtype=source.dtype)
    new_img[0:block_size, 0:block_size] = source

    # Extract overlap regions (views, not copies)
    cut_mask_overlap = cut_mask[:block_size, :overlap]
    source_overlap = source[:block_size, -overlap:]
    patch_overlap = patch[:block_size, :overlap]

    # Get patched overlap region
    patched_overlap = apply_mask(source_overlap, 1.0 - cut_mask_overlap) + apply_mask(patch_overlap, cut_mask_overlap)

    # The next section does the following, albeit minimizing mem. allocations (thus being harder to read)
    # 1. Get the gradients for all patches;
    # 2. Compute the difference between source & patched, and patch & patched
    # 3. Get the highest difference from both and mask them to only get values near the cut.

    # Pre-allocate output arrays for gradients
    # Determine output shape (handle multi-channel)
    grad_shape = patched_overlap.shape
    assert (source.dtype == np.float32)
    _dtype, ddepth = np.float32, cv2.CV_32F

    # Allocate gradient arrays once (reuse for different images)
    gx = np.empty(grad_shape, dtype=_dtype)
    gy = np.empty(grad_shape, dtype=_dtype)

    # Compute gradients for patched_overlap
    cv2.Sobel(patched_overlap, ddepth, 1, 0, dst=gx, ksize=sobel_ksize)
    cv2.Sobel(patched_overlap, ddepth, 0, 1, dst=gy, ksize=sobel_ksize)

    # Pre-allocate diff array (will reuse)
    diff_source = np.empty(grad_shape, dtype=_dtype)

    # Compute diff_source in-place
    # diff_source = sqrt((gx_patched - gx_source)² + (gy_patched - gy_source)²)
    cv2.Sobel(source_overlap, ddepth, 1, 0, dst=diff_source, ksize=sobel_ksize)
    diff_source -= gx  # diff_source now has (gx_patched - gx_source)
    diff_source *= diff_source  # Square in-place

    # Temporary for gy difference
    temp_gy = np.empty(grad_shape, dtype=_dtype)
    cv2.Sobel(source_overlap, ddepth, 0, 1, dst=temp_gy, ksize=sobel_ksize)
    temp_gy -= gy  # (gy_patched - gy_source)
    temp_gy *= temp_gy  # Square in-place

    diff_source += temp_gy  # Add squared differences
    np.sqrt(diff_source, out=diff_source)  # In-place sqrt

    # Compute diff_patch (reuse temp_gy)
    diff_patch = temp_gy  # Reuse the allocation
    cv2.Sobel(patch_overlap, ddepth, 1, 0, dst=diff_patch, ksize=sobel_ksize)
    diff_patch -= gx  # (gx_patched - gx_patch)
    diff_patch *= diff_patch  # Square

    # Reuse gx for temporary gy computation
    cv2.Sobel(patch_overlap, ddepth, 0, 1, dst=gx, ksize=sobel_ksize)
    gx -= gy  # (gy_patched - gy_patch)
    gx *= gx  # Square

    diff_patch += gx  # Add squared differences
    np.sqrt(diff_patch, out=diff_patch)  # In-place sqrt

    # Take maximum across channels
    if len(diff_source.shape) == 3:
        max_diffs_source = np.max(diff_source, axis=2)
        max_diffs_patch = np.max(diff_patch, axis=2)
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

    if debug:
        print(f"tdiff shape: {tdiff_map.shape}, max: {np.max(tdiff_map):.2f}, min: {np.min(tdiff_map):.2f}")
        print(f"tdiff mean: {np.mean(tdiff_map):.2f}, std: {np.std(tdiff_map):.2f}")

    # endregion Compute "Transition Difference Map" END

    # Create adaptive blend mask
    blended = create_adaptive_blend_mask(
        tdiff_map,
        cut_mask_overlap,
        sobel_ksize=sobel_ksize,
        min_blur_diameter=gen_args.blend_config.min_blur_diameter,
        max_blur_diameter=gen_args.blend_config.max_blur_diameter,
        blend_scale=gen_args.blend_config.blend_scale,
        overlap_size=gen_args.overlap,
        max_blend_ratio=0.15,
        dtype=source.dtype
    )

    if debug:
        min_blend = sobel_ksize
        max_proposed = int(round(sobel_ksize * 2.5))
        max_allowed = int(block_size * 0.15)
        max_actual = min(max_proposed, max_allowed)
        print(f"Blend widths: min={min_blend}px, max={max_actual}px")

    # Update min-cut mask with adaptive blend
    cut_mask[:block_size, :overlap] = 1 - blended

    if debug:
        #print(f"Final patch shape: {final_patch.shape}, dtype: {final_patch.dtype}")

        def scale(img, factor=6):
            return cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor))

        tdiff_map = scale(tdiff_map)
        # base_mask = cv2.resize(base_mask, (base_mask.shape[1]*4, base_mask.shape[0]*4))
        blended = scale(blended)
        #min_cut_patch = scale(min_cut_patch)
        source = scale(source)
        #source = cv2.cvtColor(np.uint8(source * 255), cv2.COLOR_LAB2BGR)
        #patch_overlap = cv2.cvtColor(np.uint8(patched_overlap * 255), cv2.COLOR_LAB2BGR)
        #final_patch = cv2.cvtColor(np.uint8(final_patch * 255), cv2.COLOR_LAB2BGR)
        # cv2.imshow("Source", source)
        # cv2.imshow("Patch", patch)
        cv2.imshow("Raw Patched", patched_overlap)
        #cv2.imshow("Blend Patched", ??)
        cv2.imshow("Source", source)
        cv2.imshow("tdiff_map (discontinuity)", tdiff_map / tdiff_map.max())
        cv2.imshow("Blend Mask (final)", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #return cut


if __name__ == "__main__":
    print(get_max_possible_gradient_diff("float32", 3))
