import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
from functools import lru_cache

from .misc.dry import apply_mask
from .types import UiCoordData, GenParams


DEFAULT_MAX_BLEND_RATIO = 0.28


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
    float : Maximum possible value for D
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
    kx, ky = cv2.getDerivKernels(1, 0, sobel_ksize, normalize=False)
    kernel_2d = np.outer(kx, ky)

    # Maximum response is sum of absolute values in one column (for vertical edge)
    sobel_scale = np.sum(np.abs(kernel_2d[:, -1]))

    # Maximum gradient in one direction
    K = sobel_scale * max_pixel_value

    # Maximum gradient vector magnitude: sqrt(gx² + gy²)
    max_gradient_magnitude = np.sqrt(2) * K

    # Maximum difference when gradients point in opposite directions
    max_diff = 2 * max_gradient_magnitude

    return max_diff


def create_adaptive_blend_mask_fast(tdiff_map, mcp_mask_overlap, sobel_ksize=3,
                                    blend_scale=1.0,
                                    min_blend_width=None,
                                    max_blend_width=None,
                                    patch_size=None,
                                    max_blend_ratio=DEFAULT_MAX_BLEND_RATIO,
                                    dtype=np.uint8):
    """
    Create adaptive blend mask with transition width based on gradient differences.

    Parameters:
    -----------
    tdiff_map : np.ndarray (H, W)
        Gradient difference map (higher = more difficult transition)
    mcp_mask_overlap : np.ndarray (H, W)
        Min-cut mask (0 = source side, 1 = patch side)
    sobel_ksize : int
        Sobel kernel size used to compute D
    blend_scale : float
        Exponent for mapping D to blend widths (higher = more aggressive)
    min_blend_width : int or None
        Minimum blend width in pixels. If None, equals sobel_ksize
    max_blend_width : int or None
        Maximum blend width in pixels. If None, scales with sobel_ksize (2.5x)
        and is capped by max_blend_ratio of patch_size
    patch_size : int or None
        Side length of the square patch. Used to cap max_blend_width
    max_blend_ratio : float
        Maximum blend width as a fraction of patch size (default: 0.2 = 20%)
    dtype : numpy dtype
        Data type of source images (for calculating theoretical max)

    Returns:
    --------
    np.ndarray (H, W, 1) : Blend mask (0 = source, 1 = patch)
    """

    # Auto-scale blend widths based on Sobel kernel size
    if min_blend_width is None:
        # Minimum should be at least the kernel size
        min_blend_width = sobel_ksize

    if max_blend_width is None:
        max_blend_width = _auto_max_blend_size(max_blend_ratio, patch_size, sobel_ksize)

    # Ensure max > min
    if max_blend_width <= min_blend_width:
        max_blend_width = min_blend_width + 1

    # Calculate theoretical maximum (cached)
    # Convert dtype to string for hashable cache key
    dtype_str = np.dtype(dtype).name
    max_gradient_diff = get_max_possible_gradient_diff(dtype_str, sobel_ksize)

    # Smooth tdiff to remove hard pixel transitions
    blur_sigma = max(1.0, sobel_ksize / 2.0)
    tdiff_smoothed = gaussian_filter(tdiff_map, sigma=blur_sigma)

    # Normalize from 0 to theoretical maximum
    tdiff_norm = np.clip(tdiff_smoothed / max_gradient_diff, 0, 1)

    # Map normalized tdiff values to blend widths
    blend_widths = min_blend_width + (max_blend_width - min_blend_width) * (tdiff_norm ** blend_scale)

    # Get binary mask from min-cut
    mcp_binary = (mcp_mask_overlap > 0.5).astype(float)

    # Calculate signed distance from transition line
    dist_from_source = distance_transform_edt(mcp_binary)
    dist_from_patch = distance_transform_edt(1 - mcp_binary)
    signed_distance = dist_from_patch - dist_from_source

    # Create smooth transition using sigmoid function
    t = signed_distance / (blend_widths / 2.0 + 1e-8)
    blend_mask = 1.0 / (1.0 + np.exp(-3 * t))

    return blend_mask[:, :, np.newaxis]


def _auto_max_blend_size(max_blend_ratio, patch_size, sobel_ksize):
    # Start with 2.5x the kernel size (rounded)
    max_blend_width = round(sobel_ksize * 2.5)

    # Cap it based on patch size if provided
    if patch_size is not None:
        max_allowed = int(patch_size * max_blend_ratio)
        max_blend_width = min(max_blend_width, max_allowed)
    return max_blend_width


def auto_max_blend_size(patch_size, sobel_ksize):
    return _auto_max_blend_size(DEFAULT_MAX_BLEND_RATIO, patch_size, sobel_ksize)


def _compute_adaptive_blend_mask(source: np.ndarray, #patch: np.ndarray,
                                overlap: int, sobel_ksize: int = 3, version: int = 1,
                                debug: bool = True):
    """
    Compute adaptive blend mask based on gradient differences.
    Optimized for performance with pre-allocated arrays and minimal allocations.
    """
    from bmquilting.synthesis_subroutines import get_min_cut_patch_horizontal_method

    assert source.shape[0] == source.shape[1], "Source must be square"
    block_size = source.shape[0]

    # Setup output dimensions
    dims = np.copy(source.shape)
    dims[0] = block_size
    dims[1] = block_size * 2 - overlap

    new_img = np.zeros(dims, dtype=source.dtype)
    new_img[0:block_size, 0:block_size] = source

    # Setup min-cut function
    get_min_cut_patch = get_min_cut_patch_horizontal_method(version)
    mock_gen_args = GenParams(block_size, overlap, 0, False, version)

    # For testing purposes use same texture as patch
    patch = source  # No copy needed if we're not modifying it

    # Compute min-cut patch
    min_cut_patch, mcp_mask = get_min_cut_patch(source, patch, mock_gen_args, highlight=False)
    new_img[:block_size, block_size - overlap:] = min_cut_patch

    # Extract overlap regions (views, not copies)
    patched_overlap = new_img[:block_size, block_size - overlap:block_size]
    source_overlap = patch[:block_size, block_size - overlap:block_size]
    patch_overlap = patch[:block_size, :overlap]
    mcp_mask_overlap = mcp_mask[:block_size, :overlap]

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
    np.multiply(max_diffs_source, 1 - mcp_mask_overlap, out=tdiff_map)

    # Use max_diffs_patch as temporary
    np.multiply(max_diffs_patch, mcp_mask_overlap, out=max_diffs_patch)
    np.maximum(tdiff_map, max_diffs_patch, out=tdiff_map)

    if debug:
        print(f"tdiff shape: {tdiff_map.shape}, max: {np.max(tdiff_map):.2f}, min: {np.min(tdiff_map):.2f}")
        print(f"tdiff mean: {np.mean(tdiff_map):.2f}, std: {np.std(tdiff_map):.2f}")

    # Create adaptive blend mask
    blended = create_adaptive_blend_mask_fast(
        tdiff_map,
        mcp_mask_overlap,
        sobel_ksize=sobel_ksize,
        blend_scale=10.0,
        patch_size=block_size,
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
    mcp_mask[:block_size, :overlap] = 1 - blended[:, :, 0]

    # Create final blended patch
    final_patch = np.zeros(source.shape, dtype=_dtype)
    final_patch[:, :overlap] = source[:, -overlap:]
    final_patch = apply_mask(final_patch, 1 - mcp_mask) + apply_mask(patch, mcp_mask)

    if debug:
        print(f"Final patch shape: {final_patch.shape}, dtype: {final_patch.dtype}")

        def scale(img, factor=6):
            return cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor))

        tdiff_map = scale(tdiff_map)
        # base_mask = cv2.resize(base_mask, (base_mask.shape[1]*4, base_mask.shape[0]*4))
        blended = scale(blended)
        final_patch = scale(final_patch)
        min_cut_patch = scale(min_cut_patch)
        source = scale(source)
        source = cv2.cvtColor(np.uint8(source * 255), cv2.COLOR_LAB2BGR)
        min_cut_patch = cv2.cvtColor(np.uint8(min_cut_patch * 255), cv2.COLOR_LAB2BGR)
        final_patch = cv2.cvtColor(np.uint8(final_patch * 255), cv2.COLOR_LAB2BGR)
        # cv2.imshow("Source", source)
        # cv2.imshow("Patch", patch)
        cv2.imshow("Raw Patched", min_cut_patch)
        cv2.imshow("Blend Patched", final_patch)
        cv2.imshow("Source", source)
        cv2.imshow("tdiff_map (discontinuity)", tdiff_map / tdiff_map.max())
        cv2.imshow("Blend Mask (final)", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_patch, mcp_mask



def compute_adaptive_blend_mask(source: np.ndarray, patch: np.ndarray, cut_mask: np.ndarray,
                                overlap: int, sobel_ksize: int = 3,
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
    block_size = source.shape[0]

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
    blended = create_adaptive_blend_mask_fast(
        tdiff_map,
        cut_mask_overlap,
        sobel_ksize=sobel_ksize,
        blend_scale=1,  # TODO add arg for this!
        patch_size=block_size,
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
    cut_mask[:block_size, :overlap] = 1 - blended[:, :, 0]

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
    root_dir = "test_data"
    text_idx = "18"  # 18 and 9 are good for testing purposes

    src2 = cv2.imread(f"{root_dir}/textures/t{text_idx}.png", cv2.IMREAD_COLOR_BGR)
    print(f"max src2 -> {np.max(src2)}")
    src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2LAB)
    print(f"max src2 post conv -> {np.max(src2)}")
    src2 = np.float32(src2) / 255
    print(f"max src2 post norm-> {np.max(src2)}")

    #print(f"max rev-> {np.max(cv2.cvtColor(src2*255, cv2.COLOR_LAB2BGR)) * 255 }")
    #cv2.imshow("test revert", cv2.cvtColor(np.uint8(src2*255), cv2.COLOR_LAB2BGR) )
    #cv2.waitKey()
    #quit()

    ss = np.min(src2.shape[:2])
    src2 = src2[:ss, :ss]

    _compute_adaptive_blend_mask(src2, 80, 9)
    quit()

    # patched = patch_single(src2, 50)
    # cv2.waitKey(0)
