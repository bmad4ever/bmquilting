from functools import lru_cache
import importlib.util

import cv2
import numpy as np
import cv2 as cv

from .jena2020.generate import (inf, findPatchHorizontal, findPatchVertical, findPatchBoth,
                                getMinCutPatchHorizontal, getMinCutPatchVertical, getMinCutPatchBoth)
from .types import GenParams, BlendConfig, num_pixels
from .seam_smartblur import compute_adaptive_blend_mask, auto_max_blend_diameter, auto_min_blend_size
from .misc.dry import apply_mask

from .misc.shmem_utils import SharedTextureList

epsilon = np.finfo(float).eps


def debug_resize(arr, factor=8):
    return cv2.resize(arr, (arr.shape[1] * factor, arr.shape[0] * factor))


# region   get methods by version


def get_find_patch_to_the_right_method(version: int):
    match version:
        case 0:
            def jena_right(left, source, gen_args: GenParams, rng):
                return findPatchHorizontal(left, source, gen_args.block_size, gen_args.overlap, gen_args.tolerance, rng)

            return jena_right
        case _:
            def vx_right(left_block, image, gen_args, rng):
                return find_patch_vx(left_block, None, None, None, image, gen_args, rng)

            return vx_right


def get_find_patch_below_method(version: int):
    match version:
        case 0:
            def jena_below(top, source, gen_args: GenParams, rng):
                return findPatchVertical(top, source, gen_args.block_size, gen_args.overlap, gen_args.tolerance, rng)

            return jena_below
        case _:
            def vx_below(top_block, image, gen_args, rng):
                return find_patch_vx(None, None, top_block, None, image, gen_args, rng)

            return vx_below


def get_find_patch_both_method(version: int):
    match version:
        case 0:
            def jena_both(left, top, source, gen_args: GenParams, rng):
                return findPatchBoth(left, top, source, gen_args.block_size, gen_args.overlap, gen_args.tolerance, rng)

            return jena_both
        case _:
            def vx_both(left_block, top_block, image, gen_args, rng):
                return find_patch_vx(left_block, None, top_block, None, image, gen_args, rng)

            return vx_both


def get_min_cut_patch_horizontal_method(version: int):
    if version == 0:
        def jena_cut_h(left, patch, gen_args: GenParams):
            return getMinCutPatchHorizontal(left, patch, gen_args.block_size, gen_args.overlap)

        return jena_cut_h
    return get_min_cut_patch_horizontal


def get_min_cut_patch_vertical_method(version: int):
    if version == 0:
        def jena_cut_v(top, patch, gen_args: GenParams):
            return getMinCutPatchVertical(top, patch, gen_args.block_size, gen_args.overlap)

        return jena_cut_v
    return get_min_cut_patch_vertical


def get_min_cut_patch_both_method(version: int):
    if version == 0:
        def jena_cut_both(left, top, patch, gen_args: GenParams):
            return getMinCutPatchBoth(left, top, patch, gen_args.block_size, gen_args.overlap)

        return jena_cut_both
    return get_min_cut_patch_both


def compute_errors(diffs: list[np.ndarray], version: int) -> np.ndarray:
    match version:
        case 1:
            return np.add.reduce(diffs)
        case 2:
            return np.maximum.reduce(diffs)
        case 3:
            return 1 - np.minimum.reduce(diffs)  # values from 0 to 2
        case _:
            raise NotImplementedError("Specified patch search version is not implemented.")


def get_match_template_method(version: int) -> int:
    match version:
        case 1:
            return cv.TM_SQDIFF
        case 2:
            return cv.TM_SQDIFF
        case 3:
            return cv.TM_CCOEFF_NORMED
        case _:
            raise NotImplementedError("Specified patch search version is not implemented.")


# endregion


# region  custom implementation of: patch search & min cut + auxiliary methods


def find_patch_vx(ref_block_left: np.ndarray | None,
                  ref_block_right: np.ndarray | None,
                  ref_block_top: np.ndarray | None,
                  ref_block_bottom: np.ndarray | None,
                  lookup_textures: list[np.ndarray] | SharedTextureList,
                  gen_args: GenParams,
                  rng: np.random.Generator
                  ) -> np.ndarray:
    """
    Finds the best-matching block across all textures in lookup_textures
    that satisfies the boundary constraints, applying tolerance to the
    absolute global minimum error.
    """
    block_size, overlap, tolerance = gen_args.bot
    template_method = get_match_template_method(gen_args.version)

    # List to store error matrices (Pass 1) or final candidates (Pass 2)
    err_mats: list[np.ndarray | None] = []
    global_min_error = np.inf

    mask = None
    if gen_args.vignette_on_match_template:
        mask = patch_blending_vignette(
            gen_args.block_size, gen_args.overlap,
            ref_block_left is not None,
            ref_block_right is not None,
            ref_block_top is not None,
            ref_block_bottom is not None
        ).copy()  # mind that the vignette is cached
        # invert mask inplace
        mask *= -1
        mask += 1

    # --- PASS 1: Find the absolute global minimum error ---
    for texture in lookup_textures:

        # Skip if texture is too small for a block
        if texture.shape[0] < block_size or texture.shape[1] < block_size:
            err_mats.append(None)  # Keep placeholder for alignment
            continue

        blks_diffs: list[np.ndarray] = []

        # Template Matching
        if ref_block_left is not None:
            blks_diffs.append(cv.matchTemplate(
                image=texture[:, :-block_size + overlap],
                templ=ref_block_left[:, -overlap:],
                method=template_method,
                mask=mask[:, :overlap] if mask is not None else None
            ))
        if ref_block_right is not None:
            blks_diffs.append(cv.matchTemplate(
                image=np.roll(texture, -block_size + overlap, axis=1)[:, :-block_size + overlap],
                templ=ref_block_right[:, :overlap],
                method=template_method,
                mask=mask[:, -overlap:] if mask is not None else None
            ))
        if ref_block_top is not None:
            blks_diffs.append(cv.matchTemplate(
                image=texture[:-block_size + overlap, :],
                templ=ref_block_top[-overlap:, :],
                method=template_method,
                mask=mask[:overlap, :] if mask is not None else None
            ))
        if ref_block_bottom is not None:
            blks_diffs.append(cv.matchTemplate(
                image=np.roll(texture, -block_size + overlap, axis=0)[:-block_size + overlap, :],
                templ=ref_block_bottom[:overlap, :],
                method=template_method,
                mask=mask[-overlap:, :] if mask is not None else None
            ))

        # Compute combined error matrix for this texture
        err_mat = compute_errors(blks_diffs, gen_args.version)
        err_mat = np.maximum(err_mat, 1e-8)  # clip floor to zero
        err_mats.append(err_mat)

        # Update global minimum
        global_min_error = min(global_min_error, np.min(err_mat))

    # --- Check for impossible match ---
    print(f"gme = {global_min_error}")

    if global_min_error == np.inf:
        raise ValueError("Could not find a suitable patch in any lookup texture "
                         "(all textures were too small or had matching issues).")

    # --- PASS 2: Collect all candidates within the final tolerance window ---
    final_candidates: list[tuple[int, int, int]] = []  # (texture_idx, y, x)
    acceptable_error = (1.0 + tolerance) * global_min_error

    for texture_idx, err_mat in enumerate(err_mats):
        if err_mat is None:  # Skip small textures
            continue

        # Find all positions (y, x) that satisfy the global tolerance threshold
        y_T, x_T = np.nonzero(err_mat <= acceptable_error)

        # Record all valid patch coordinates
        for y, x in zip(y_T, x_T):
            final_candidates.append((texture_idx, y, x))

    # 3. Random Selection and Patch Extraction
    if not final_candidates:
        # This occurs if the tolerance factor is so small it filters out everything,
        # but the check for global_min_error == np.inf should usually catch the
        # true impossible cases.
        raise ValueError("No patches met the final tolerance criteria.")

    # Randomly select one candidate from the final set
    c = rng.integers(len(final_candidates))
    best_texture_idx, best_y, best_x = final_candidates[c]

    # Extract the patch from the winning texture (This triggers a single read from the memmap)
    winning_texture = lookup_textures[best_texture_idx]

    return winning_texture[best_y:best_y + block_size, best_x:best_x + block_size]


def get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                           patch_block, gen_args: GenParams):
    # note -> if blurred, masks weights are scaled with respect the max mask value.
    # example: mask1: 0.2 mask2:0.5 -> max = .5 ; sum = .7 -> (.2*lb + .5*tb) * (max/sum) + patch * (1 - max)
    # likely "heavier" to compute, but does not require handling each corner individually

    block_size, overlap = gen_args.bo

    masks_list = []

    has_left = ref_block_left is not None
    has_right = ref_block_right is not None
    has_top = ref_block_top is not None
    has_bottom = ref_block_bottom is not None

    res_block = np.zeros_like(patch_block)

    def process_block(rolled_block, mask):
        if gen_args.blend_into_patch and gen_args.blend_config.use_vignette:
            vignette = patch_blending_vignette(block_size, overlap, has_left, has_right, has_top, has_bottom)
            mask *= vignette
        masks_list.append(mask)
        np.add(apply_mask(rolled_block, mask, True), res_block, out=res_block)

    if has_left:
        mask = get_min_cut_patch_mask_horizontal(ref_block_left, patch_block, block_size, overlap, gen_args.blend_config)
        #print(f"prior mask = {mask}")

        if gen_args.blend_into_patch:
            compute_adaptive_blend_mask(
                ref_block_left,
                patch_block,
                mask,
                gen_args
            )
        #print(f"post mask = {mask}")
        process_block(np.roll(ref_block_left, overlap, 1), mask)

    if has_right:
        adj_src = np.fliplr(ref_block_right)
        adj_ptc = np.fliplr(patch_block)
        mask = get_min_cut_patch_mask_horizontal(adj_src, adj_ptc, block_size, overlap, gen_args.blend_config)
        if gen_args.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                gen_args
            )
        mask = np.fliplr(mask)
        process_block(np.roll(ref_block_right, -overlap, 1), mask)

    if has_top:
        # V , >  counterclockwise rotation
        adj_src = np.rot90(ref_block_top)
        adj_ptc = np.rot90(patch_block)
        mask = get_min_cut_patch_mask_horizontal(adj_src, adj_ptc, block_size, overlap, gen_args.blend_config)
        if gen_args.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                gen_args
            )
        mask = np.rot90(mask, 3)
        process_block(np.roll(ref_block_top, overlap, 0), mask)

    if has_bottom:
        adj_src = np.fliplr(np.rot90(ref_block_bottom))
        adj_ptc = np.fliplr(np.rot90(patch_block))
        mask = get_min_cut_patch_mask_horizontal(adj_src, adj_ptc, block_size, overlap, gen_args.blend_config)
        if gen_args.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                gen_args
            )
        mask = np.rot90(np.fliplr(mask), 3)
        process_block(np.roll(ref_block_bottom, -overlap, 0), mask)

    # compute auxiliary data to weight original/patch & fix sum at the corners
    mask_s = sum(masks_list)
    masks_max = np.maximum.reduce(masks_list)
    mask_mos = np.divide(masks_max, mask_s, out=np.zeros_like(mask_s), where=mask_s != 0)

    apply_mask(res_block, mask_mos, True)  # weight res_block (also fixes sum at corners)
    patch_weight = np.subtract(1, masks_max, out=masks_max)  # note that masks_max is no longer needed
    return np.add(res_block, apply_mask(patch_block, patch_weight), out=res_block), patch_weight


@lru_cache(maxsize=4)
def patch_blending_vignette(block_size: num_pixels, overlap: num_pixels,
                            left: bool, right: bool, top: bool, bottom: bool) -> np.ndarray:
    margin = 1  # must be small !
    power = 4.5  # controls drop-off
    p = 6  # controls the shape

    def corner_distance(y, x):
        distance = ((abs(x - overlap + margin / 2) ** p + abs(y - overlap + margin / 2) ** p) ** (1 / p)) / (
                overlap - margin)
        return np.clip(distance, 0, 1)

    mask = np.ones((block_size, block_size), dtype=np.float32)
    i, j = np.meshgrid(np.arange(overlap), np.arange(overlap))
    curve_top_left_corner = 1 - corner_distance(i, j) ** power

    # Corners
    # Top left corner
    if top and left:
        mask[:overlap, :overlap] = curve_top_left_corner
    elif top:
        mask[:overlap, :overlap] = curve_top_left_corner[:, -1].reshape(-1, 1)  # Copy the last column to all columns
    elif left:
        mask[:overlap, :overlap] = curve_top_left_corner[-1, :].reshape(1, -1)  # Copy the last row to all rows

    # Top right corner
    if top and right:
        mask[:overlap, -overlap:] = np.flip(curve_top_left_corner, axis=1)
    elif top:
        mask[:overlap, -overlap:] = curve_top_left_corner[:, -1].reshape(-1, 1)
    elif right:
        mask[:overlap, -overlap:] = np.flip(curve_top_left_corner[-1, :]).reshape(1, -1)

    # Bottom left corner
    if bottom and left:
        mask[-overlap:, :overlap] = np.flip(curve_top_left_corner, axis=0)
    elif bottom:
        mask[-overlap:, :overlap] = np.flip(curve_top_left_corner[:, -1]).reshape(-1, 1)
    elif left:
        mask[-overlap:, :overlap] = curve_top_left_corner[-1, :].reshape(1, -1)

    # Bottom right corner
    if bottom and right:
        mask[-overlap:, -overlap:] = np.flip(curve_top_left_corner)
    elif bottom:
        mask[-overlap:, -overlap:] = (np.flip(curve_top_left_corner[:, -1])
                                      .reshape(-1, 1))
    elif right:
        mask[-overlap:, -overlap:] = np.flip(curve_top_left_corner[-1, :]).reshape(1, -1)  # Copy the last row flipped

    # Edges
    if top:
        mask[:overlap, overlap:block_size - overlap] = (curve_top_left_corner[:, -1]
                                                        .reshape(-1, 1))  # Copy the last column vertically
    if bottom:
        mask[-overlap:, overlap:block_size - overlap] = (np.flip(curve_top_left_corner[:, -1])
                                                         .reshape(-1, 1))  # Copy the last column flipped vertically
    if left:
        mask[overlap:block_size - overlap, :overlap] = (curve_top_left_corner[-1, :]
                                                        .reshape(1, -1))  # Copy the last row horizontally
    if right:
        mask[overlap:block_size - overlap, -overlap:] = (np.flip(curve_top_left_corner[-1, :])
                                                         .reshape(1, -1))  # Copy the last row flipped horizontally
    mask = 1 - mask
    return mask


def blur_patch_mask(src_mask: np.ndarray, vignette: np.ndarray):
    """
    Uses the distance transform to blur the patch -> dst_grad.
    This mask is composed w/ additional masks to avoid noticeable discontinuities on the generation.

    Args:
        src_mask: patch mask complement in f32; patch area as 0s and outside as 1s
        vignette: mask that fades from 1s to 0s on the edges,

    Returns: (vignette * dst_grad) + ((1 - vignette) * edge_blurred_source)
    """
    # compute dst_grad & edge_blurred
    blurred_a = np.ascontiguousarray(src_mask * 255, dtype=np.uint8)
    blurred_b = np.empty_like(blurred_a, order='c')
    cv.morphologyEx(blurred_a, cv.MORPH_ERODE, iterations=1, dst=blurred_b,
                    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    cv.blur(blurred_b, (3, 3), dst=blurred_a)
    edge_blurred = np.float32(blurred_a) / 255

    # devnote: L2 only does not work w/ uint8, only L1 works w/ uint8 I think...
    #  prior micro optimization using blurred_a & b actually removed the gradient mask, and the blured was being used instead
    dst_grad = cv.distanceTransform(blurred_b, cv.DIST_L2, maskSize=0)
    dst_grad_aux = cv.blur(dst_grad, (5, 5))
    np.divide(dst_grad_aux, max(np.max(dst_grad_aux), 1), out=dst_grad)

    # compute formula re-using already allocated memory
    #   formula: (vignette * dst_grad) + ((1 - vignette) * edge_blurred)
    weighted_blurred = np.multiply(vignette, dst_grad, out=dst_grad)
    one_minus_vignette = 1 - vignette  # vignette is cached, can't edit it
    weighted_src = np.multiply(edge_blurred, one_minus_vignette, out=one_minus_vignette)
    result = np.add(weighted_blurred, weighted_src, out=weighted_blurred)
    return np.clip(result, 0, 1, out=result)  # better safe than sorry


def get_min_cut_patch_mask_horizontal_jena2020(block1, block2, block_size: num_pixels, overlap: num_pixels):
    """
    @param block1: block to the left, with the overlap on its right edge
    @param block2: block to the right, with the overlap on its left edge
    @return: ONLY the mask (not the patched overlap section)
    """
    err = ((block1[:, -overlap:] - block2[:, :overlap]) ** 2).mean(2)
    # maintain minIndex for 2nd row onwards and
    min_index = []
    E = [list(err[0])]
    for i in range(1, err.shape[0]):
        # Get min values and args, -1 = left, 0 = middle, 1 = right
        e = [inf] + E[-1] + [inf]
        e = np.array([e[:-2], e[1:-1], e[2:]])
        # Get minIndex
        min_arr = e.min(0)
        min_arg = e.argmin(0) - 1
        min_index.append(min_arg)
        # Set Eij = e_ij + min_
        Eij = err[i] + min_arr
        E.append(list(Eij))

    # Check the last element and backtrack to find path
    path = []
    min_arg = np.argmin(E[-1])
    path.append(min_arg)

    # Backtrack to min path
    for idx in min_index[::-1]:
        min_arg = min_arg + idx[min_arg]
        path.append(min_arg)
    # Reverse to find full path
    path = path[::-1]
    mask = np.zeros((block_size, block_size), dtype=block1.dtype)
    for i in range(len(path)):
        mask[i, :path[i] + 1] = 1
    return mask


def update_seams_map_view(seams_map_view, gen_args, patch_weights):
    seam_map_block = get_seam_mask_from_patch_weights(patch_weights, gen_args)
    clear_seam_overlapped_by_patch(seams_map_view, patch_weights)
    #seams_map_view += seam_map_block
    np.maximum(seams_map_view, seam_map_block, out=seams_map_view)


if importlib.util.find_spec("pyastar2d") is not None:
    import pyastar2d


    def get_min_cut_patch_mask_horizontal_astar(
            block1, block2,
            block_size: num_pixels, overlap: num_pixels,
            blend_config: BlendConfig = None
    ):
        """
        @param block1: block to the left, with the overlap on its right edge (should be normalized!)
        @param block2: block to the right, with the overlap on its left edge

        @return: ONLY the mask (not the patched overlap section)
        """
        # get min and max safety radii
        if blend_config is not None:
            min_safe_rad = blend_config.min_blur_diameter//2
            max_safe_rad = blend_config.max_blur_diameter//2
        else:
            min_safe_rad = 0
            max_safe_rad = 0

        # compute error matrix
        block1_safe_overlap_view = block1[:, -overlap:]
        block2_safe_overlap_view = block2[:, :overlap]
        if min_safe_rad > 0:
            # trim to avoid unneeded computations
            block1_safe_overlap_view = block1_safe_overlap_view[:, min_safe_rad:-min_safe_rad]
            block2_safe_overlap_view = block2_safe_overlap_view[:, min_safe_rad:-min_safe_rad]

        err = block1_safe_overlap_view - block2_safe_overlap_view
        err **= 2
        err = err.mean(2)

        # adjust safety_radius
        safety_radius = min(
            # try to get more real estate if errors are small instead of defaulting to max possible radius
            # mind that this guess is heuristic, err here does not translate directly to the seam diff map
            # meaning that: without opting for the max blur radius
            #   there is always the risk of getting a blur near the edges
            (round(min_safe_rad + (max_safe_rad-min_safe_rad) * np.max(err))+max_safe_rad)//2,
            # safeguard against big save_radius values
            round(overlap / 3)
        )
        print((max_safe_rad-min_safe_rad) * np.max(err))
        print(min_safe_rad)
        print(max_safe_rad)
        print(safety_radius)
        # trim the err matrix so that values within the adjusted safety radius are not used
        to_trim = safety_radius - min_safe_rad
        if to_trim > 0:
            err = err[:, to_trim:-to_trim]

        err = err ** 2  # make penalty func steeper

        # scale to integer color range and offset by 1 (works for both RGB and LAB)
        #   this is done so that the distance from 0 to the smallest possible error
        #   keeps the same proportion relative to other error values
        #   otherwise there would be a relative penalty mismatch for pixels with error equal to zero
        #   when offsetting the errors, which is required for both pyastar2d (min. value accepted as weight is 1)
        err *= 255
        err += 1

        # make the lowest value big enough for 1 to be negligible and pad extremes with weight 1
        #   this is done so that the extremes work as 'free corridors';
        #   to make the start and end points arbitrary
        err *= block_size ** 2
        err = np.pad(err, ((1, 1), (0, 0)), 'constant', constant_values=(1, 1))

        start = (0, err.shape[1] // 2)
        end = (err.shape[0] - 1, err.shape[1] // 2)

        path = pyastar2d.astar_path(err, start, end, allow_diagonal=True)
        mask = np.ones((block_size, block_size), dtype=block1.dtype)
        shape_m2 = err.shape[0] - 2

        start_index = 0  # find start index to avoid checking 0 < i every iteration
        for idx, (i, j) in enumerate(path):
            if 0 < i:
                start_index = idx
                break

        for i, j in path[start_index:]:  # draw path
            mask[i - 1, j + 1 + safety_radius] = 0
            if i >= shape_m2:
                break

        cv.floodFill(mask, None, (mask.shape[0] - 1, mask.shape[1] - 1), (0,))
        return mask


    get_min_cut_patch_mask_horizontal = get_min_cut_patch_mask_horizontal_astar
else:
    get_min_cut_patch_mask_horizontal = get_min_cut_patch_mask_horizontal_jena2020


# endregion


# region min cut patch aliases

def get_min_cut_patch_horizontal(left_block, patch_block, gen_args: GenParams):
    return get_4way_min_cut_patch(
        left_block, None, None, None,
        patch_block, gen_args
    )


def get_min_cut_patch_vertical(top_block, patch_block, gen_args: GenParams):
    return get_4way_min_cut_patch(
        None, None, top_block, None,
        patch_block, gen_args
    )


def get_min_cut_patch_both(left_block, top_block, patch_block, gen_args: GenParams):
    return get_4way_min_cut_patch(
        left_block, None, top_block, None,
        patch_block, gen_args
    )


# endregion


def get_seam_mask_from_patch_weights(patch_weights: np.ndarray, gen_args: GenParams) -> np.ndarray:
    if gen_args.blend_into_patch:
        # Has blending, compute distance to 0.5
        #cv.imshow("patch_weights", debug_resize(patch_weights))
        seam_mask = 1 - np.abs(.5 - patch_weights) * 2
        #cv.imshow("seam mask", debug_resize(seam_mask))
        #cv.waitKey()
        np.clip(seam_mask, 0, 1, out=seam_mask)  # just in case
        return seam_mask
    else:
        # No blending, find seam line
        gx = cv.Sobel(patch_weights, cv.CV_32F, 1, 0, ksize=cv.FILTER_SCHARR)
        gy = cv.Sobel(patch_weights, cv.CV_32F, 0, 1, ksize=cv.FILTER_SCHARR)
        np.multiply(gx, gx, out=gx)
        np.multiply(gy, gy, out=gy)
        gx += gy
        mags = gx / 256
        np.clip(mags, 0, 1, out=mags)
        return mags


def clear_seam_overlapped_by_patch(seam_map_view: np.ndarray, patch_weights: np.ndarray):
    seam_map_view *= 1 - patch_weights
