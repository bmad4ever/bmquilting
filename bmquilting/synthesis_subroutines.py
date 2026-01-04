from functools import lru_cache
import importlib.util
from typing import Any

import numpy as np
import cv2 as cv
from numpy import dtype, ndarray
from numpy.random import Generator

from .types import GenParams, NumPixels, SquarePatchingBlendConfig, PatchIdx, _2D_Slice
from .seam_smartblur import compute_adaptive_blend_mask
from .misc.shmem_utils import SharedTextureList
from .misc.dry import apply_mask

from .jena2020.generate import inf, getMinCutPatchHorizontal, getMinCutPatchVertical, getMinCutPatchBoth

epsilon = np.finfo(float).eps

type FindPatchSlices = tuple[
    tuple[_2D_Slice, _2D_Slice],
    tuple[_2D_Slice, _2D_Slice],
    tuple[_2D_Slice, _2D_Slice],
    tuple[_2D_Slice, _2D_Slice]
]


def debug_resize(arr, factor=8):
    return cv.resize(arr, (arr.shape[1] * factor, arr.shape[0] * factor))


# region   get methods by version


def get_find_patch_to_the_right_method():
    def vx_right(ref_block, image, gen_params, rng):
        return find_patch_vx(True, False, False, False, ref_block, image, gen_params, rng)

    return vx_right


def get_find_patch_below_method():
    def vx_below(ref_block, image, gen_params, rng):
        return find_patch_vx(False, False, True, False, ref_block, image, gen_params, rng)

    return vx_below


def get_find_patch_both_method():
    def vx_both(ref_block, image, gen_params, rng):
        return find_patch_vx(True, False, True, False, ref_block, image, gen_params, rng)

    return vx_both


def get_min_cut_patch_horizontal_method(version: int):
    if version == 0:
        def jena_cut_h(ref_block, patch, gen_params: GenParams):
            return getMinCutPatchHorizontal(ref_block, patch, gen_params.block_size, gen_params.overlap)

        return jena_cut_h
    return get_min_cut_patch_horizontal


def get_min_cut_patch_vertical_method(version: int):
    if version == 0:
        def jena_cut_v(ref_block, patch, gen_params: GenParams):
            return getMinCutPatchVertical(ref_block, patch, gen_params.block_size, gen_params.overlap)

        return jena_cut_v
    return get_min_cut_patch_vertical


def get_min_cut_patch_both_method(version: int):
    if version == 0:
        def jena_cut_both(ref_block, patch, gen_params: GenParams):
            return getMinCutPatchBoth(ref_block, patch, gen_params.block_size, gen_params.overlap)

        return jena_cut_both
    return get_min_cut_patch_both


def compute_errors(diffs: list[np.ndarray], version: int) -> np.ndarray:
    match version:
        case 0:
            # kept somewhat similar to jena2020 (mean is just the sum scaled down by the number of channels)
            return np.add.reduce(diffs)
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
        case 0:
            # kept somewhat similar to jena2020 used in prior v0
            return cv.TM_SQDIFF
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

def find_patch_vx(overlaps_left: bool,
                  overlaps_right: bool,
                  overlaps_top: bool,
                  overlaps_bottom: bool,
                  ref_block: np.ndarray,  # gen. texture view at the patch to generate location
                  lookup_textures: list[np.ndarray] | SharedTextureList,
                  gen_params: GenParams,
                  rng: np.random.Generator
                  ) -> np.ndarray:
    """
    Calls find_patch_vx_idx and returns the patch instead of its indices.
    See find_patch_vx_idx documentation.

    :return: a block sized texture patch.
    """
    best_texture_idx, best_y, best_x = find_patch_vx_idx(
        overlaps_left, overlaps_right, overlaps_top, overlaps_bottom, ref_block, lookup_textures, gen_params, rng)

    # Extract the patch from the winning texture
    winning_texture = lookup_textures[best_texture_idx]
    block_size = gen_params.block_size
    return winning_texture[best_y:best_y + block_size, best_x:best_x + block_size]


@lru_cache(maxsize=2)
def get_slice_metadata_for_find_patch(block_size, overlap) -> FindPatchSlices:
    """
    Auxiliary function for the template matching used in the find_patch_vx_idx function.
    Computes and caches the slice objects for the texture and for the template & mask respectively.
    """
    bmo = block_size - overlap
    return (
        (np.s_[:, :-bmo], np.s_[:, :overlap] ),  # Left
        (np.s_[:, bmo: ], np.s_[:, -overlap:]),  # Right
        (np.s_[:-bmo, :], np.s_[:overlap, : ]),  # Top
        (np.s_[bmo:, : ], np.s_[-overlap:, :])   # Bottom
    )


def find_patch_vx_idx(overlaps_left: bool,
                      overlaps_right: bool,
                      overlaps_top: bool,
                      overlaps_bottom: bool,
                      ref_block: np.ndarray,  # gen. texture view at the patch to generate location
                      lookup_textures: list[np.ndarray] | SharedTextureList,
                      gen_params: GenParams,
                      rng: np.random.Generator
                      ) -> PatchIdx:
    """
    Finds the best-matching block across all textures in lookup_textures
    that satisfies the boundary constraints, applying tolerance to the
    absolute global minimum error.

    From the returned tuple, the corresponding patch can be obtained in the following way:
    lookup_textures[best_texture_idx][best_y:best_y+block_size, best_x:best_x+block_size]

    :param ref_block: roi on the texture where the patch will be placed over.
        overlapping regions should have been already filled prior to calling this method.

    :param gen_params: besides the block and overlap size, gen_params provides the tolerance (percentage) which will
        define what is the acceptable range for the patch selection with respect to the best possible patch errors.
        The higher the tolerance, the more leeway the function has to select a "worse" patch.

    :return: best_texture_idx, best_y, best_x
    """
    block_size, overlap, tolerance = gen_params.bot
    template_method = get_match_template_method(gen_params.version)

    # List to store error matrices (Pass 1) or final candidates (Pass 2)
    err_mats: list[np.ndarray | None] = []
    global_min_error = np.inf

    # Get Vignette Mask depending on Generation Params
    mask = None
    if gen_params.vignette_on_match_template:
        mask = patch_blending_vignette(
            gen_params.block_size, gen_params.overlap,
            overlaps_left,
            overlaps_right,
            overlaps_top,
            overlaps_bottom
        )
        mask = 1 - mask  # invert mask, mind that vignette is cached (original shouldn't be edited)

    # Define Conditions list and 'Bookmarks' (static metadata, very cheap)
    #  done to avoid repeating if conditions for each overlapping area, making the code slightly more compact
    conditions = [overlaps_left, overlaps_right, overlaps_top, overlaps_bottom]
    slice_definitions = get_slice_metadata_for_find_patch(block_size, overlap)  # (texture, template & mask)

    # --- PASS 1: Compute errors for each texture & find the absolute global minimum error ---
    for texture in lookup_textures:

        # Skip if texture is too small for a block
        if texture.shape[0] < block_size or texture.shape[1] < block_size:
            err_mats.append(None)  # Keep placeholder for alignment
            continue

        blks_diffs: list[np.ndarray] = []

        # Iterate Lazily
        for has_overlap, (tex_s, tmpl_s) in zip(conditions, slice_definitions):
            if has_overlap:
                blks_diffs.append(cv.matchTemplate(
                    image=texture[tex_s],
                    templ=ref_block[tmpl_s],
                    method=template_method,
                    mask=mask[tmpl_s] if mask is not None else None
                ))

        # Compute combined error matrix for this texture
        err_mat = compute_errors(blks_diffs, gen_params.version)
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
    final_candidates = _filter_candidate_patches(err_mats, global_min_error, tolerance)
    best_texture_idx, best_x, best_y = _select_a_random_patch(final_candidates, rng)
    return best_texture_idx, best_y, best_x


def _select_a_random_patch(final_candidates: list[tuple[int, int, int]], rng: Generator) -> tuple[int, int, int]:
    """
    :param final_candidates: the list of patches' metadata in the following tuple format: ( texture index, y-coord, x-coord )
    :param rng: random generator used to select the patch.
    :return: tuple with the selected patch metadata: ( texture index, y-coord, x-coord )
    """
    if not final_candidates:
        # This occurs if the tolerance factor is so small it filters out everything,
        # but the check for global_min_error == np.inf should usually catch the
        # true impossible cases.
        raise ValueError("No patches met the final tolerance criteria.")
    c = rng.integers(len(final_candidates))
    best_texture_idx, best_y, best_x = final_candidates[c]
    return best_texture_idx, best_x, best_y


def _filter_candidate_patches(err_mats: list[ndarray | None],
                              global_min_error: float , tolerance: float) -> list[tuple[int, int, int]]:
    """

    :param err_mats: error matrices resulting from template matching.
        NOTE: the error matrices indices should match the texture indices.
              if a texture is too small, or has some other problem that invalidates fetching a patch from it,
              its' corresponding err_mat can be set as None to preserve the indices correspondence.
    :param global_min_error: the min error found from all the provided matrices.
        NOTE: the iteration used to obtain the err_mats should also compute the global minimum error,
        there is no need to iterate the list an additional time to obtain this value prior to filtering the candidates.
    :param tolerance: how high, percentage wise, can the error be above the global minimum error to pass as a candidate.
    :return: the list of candidates in the following tuple format: ( index in err_mats, y-coord, x-coord )
    """
    final_candidates: list[tuple[int, int, int]] = []  # (texture_idx, y, x)
    acceptable_error = (1.0 + tolerance) * global_min_error

    for texture_idx, err_mat in enumerate(err_mats):
        if err_mat is None:  # skip small or invalid textures
            continue
        y_t, x_t = np.nonzero(err_mat <= acceptable_error)  # filter positions with respect to the tolerance threshold
        for y, x in zip(y_t, x_t):   # record all valid patch coordinates
            final_candidates.append((texture_idx, y, x))
    return final_candidates


def get_4way_min_cut_patch(
        overlaps_left: bool,
        overlaps_right: bool,
        overlaps_top: bool,
        overlaps_bottom: bool,
        ref_block: np.ndarray,  # gen. texture view at the patch to generate location
        patch_block, gen_params: GenParams) -> tuple[np.ndarray, np.ndarray]:
    """
    :param ref_block: roi on the texture where the patch will be placed over.
        overlapping regions should have been already filled prior to calling this method.
    :param patch_block: new block to be placed over the ref_block roi.
    :return: patch, mask
    """

    block_size, overlap = gen_params.bo
    masks_max = np.zeros(patch_block.shape[:2], dtype=np.float32)

    def process_block(mask):
        if gen_params.blend_into_patch and gen_params.blend_config.use_vignette:
            vignette = patch_blending_vignette(
                block_size, overlap, overlaps_left, overlaps_right, overlaps_top, overlaps_bottom)
            mask *= vignette
            # dev note:
            #   Might seem strange to multiply these instead of choosing the maximum here;
            #   the thing about the vignette is that, although initially devised as a workaround
            #   to avoid having the seams' blur reaching the edges of the overlap which would
            #   create potential "visual seam" artifacts at the edge of a new patch, this variant had an unforeseen
            #   effect that I did not think of when implementing it:
            #       It roughly aligns the generated patches' seams into a grid, which somewhat mitigates the
            #   problem of having loose seam ends creating the occasional visual discontinuity.
            #       The results seamed interesting, so I eventually settled on leaving this solution as it is.
            #       Despite not preventing the seams' blur from reaching the edge in the current implementation,
            #   the effect seemed better than the alternative. 
            #       I've also noticed that using two vignettes,
            #   one more steep for the maximum and the other for seam alignment, seems to create
            #   more visual artifacts than it solves.
        np.maximum(mask, masks_max, out=masks_max)

    if overlaps_left:
        mask = get_min_cut_patch_mask_horizontal(ref_block, patch_block, block_size, overlap,
                                                 gen_params.blend_config)

        if gen_params.blend_into_patch:
            compute_adaptive_blend_mask(
                ref_block,
                patch_block,
                mask,
                gen_params
            )
        process_block(mask)

    if overlaps_right:
        adj_src = np.fliplr(ref_block)
        adj_ptc = np.fliplr(patch_block)
        mask = get_min_cut_patch_mask_horizontal(adj_src, adj_ptc, block_size, overlap, gen_params.blend_config)
        if gen_params.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                gen_params
            )
        mask = np.fliplr(mask)
        process_block(mask)

    if overlaps_top:
        # V , >  counterclockwise rotation
        adj_src = np.rot90(ref_block)
        adj_ptc = np.rot90(patch_block)
        mask = get_min_cut_patch_mask_horizontal(adj_src, adj_ptc, block_size, overlap, gen_params.blend_config)
        if gen_params.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                gen_params
            )
        mask = np.rot90(mask, 3)
        process_block(mask)

    if overlaps_bottom:
        adj_src = np.fliplr(np.rot90(ref_block))
        adj_ptc = np.fliplr(np.rot90(patch_block))
        mask = get_min_cut_patch_mask_horizontal(adj_src, adj_ptc, block_size, overlap, gen_params.blend_config)
        if gen_params.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                gen_params
            )
        mask = np.rot90(np.fliplr(mask), 3)
        process_block(mask)

    # compute:   mask * patch + (1-mask) * ref_block
    res = apply_mask(ref_block, masks_max)
    res += apply_mask(patch_block, np.subtract(1, masks_max, out=masks_max))
    return res, masks_max


@lru_cache(maxsize=4)
def patch_blending_vignette(block_size: NumPixels, overlap: NumPixels,
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


def get_min_cut_patch_mask_horizontal_jena2020(block1, block2, block_size: NumPixels, overlap: NumPixels):
    """
    :param block1: block to the left, with the overlap on its right edge
    :param block2: block to the right, with the overlap on its left edge
    :return: ONLY the mask (not the patched overlap section)
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


def update_seams_map_view(seams_map_view: np.ndarray, patch_weights: np.ndarray, blends_into_patch: bool):
    seam_map_block = get_seam_mask_from_patch_weights(patch_weights, blends_into_patch)
    clear_seam_overlapped_by_patch(seams_map_view, patch_weights)
    #seams_map_view += seam_map_block
    np.maximum(seams_map_view, seam_map_block, out=seams_map_view)


if importlib.util.find_spec("pyastar2d") is not None:
    import pyastar2d


    def get_min_cut_patch_mask_horizontal_astar(
            ref_block, patch_block,
            block_size: NumPixels, overlap: NumPixels,
            blend_config: SquarePatchingBlendConfig = None
    ):
        """
        :param ref_block: block in the source texture being generated (should be normalized)
        :param patch_block: patch that will be pasted over the ref_block

        :return: ONLY the mask (not the patched overlap section)
        """
        # get min and max safety radii
        safety_radius = 0  # suppose zero until computed
        if blend_config is not None and blend_config.use_blur_radii_guess_pathfind_limiter:
            min_safe_rad = blend_config.min_blur_diameter // 2
            max_safe_rad = blend_config.max_blur_diameter // 2
        else:
            min_safe_rad = 0
            max_safe_rad = 0

        # compute error matrix
        block1_safe_overlap_view = ref_block[:, :overlap]
        block2_safe_overlap_view = patch_block[:, :overlap]
        if min_safe_rad > 0:
            # pre-trim to avoid unneeded computations
            block1_safe_overlap_view = block1_safe_overlap_view[:, min_safe_rad:-min_safe_rad]
            block2_safe_overlap_view = block2_safe_overlap_view[:, min_safe_rad:-min_safe_rad]

        err = avg_squared_diff(block1_safe_overlap_view, block2_safe_overlap_view)

        if max_safe_rad > 0:
            # compute safety_radius & trim the remaining area
            safety_radius = min(
                # try to get more real estate if errors are small instead of defaulting to max possible radius
                # mind that this guess is heuristic, err here does not translate directly to the seam diff map
                # meaning that: without opting for the max blur radius
                #   there is always the risk of getting a blur near the edges
                (round(min_safe_rad + (max_safe_rad - min_safe_rad) * np.max(err)) + max_safe_rad) // 2,
                # safeguard against big save_radius values
                overlap // 3
            )

            # trim the err matrix so that values within the adjusted safety radius are not used
            to_trim = safety_radius - min_safe_rad
            if to_trim > 0:
                err = err[:, to_trim:-to_trim]

        adjust_errors_func_inplace(err)
        adjust_errors_for_pystar2d_inplace(err, block_size)

        # Create 'free-corridors' at both extremes, so that the X starting point is arbitrary
        err = np.pad(err, ((1, 1), (0, 0)), 'constant', constant_values=(1, 1))

        start = (0, err.shape[1] // 2)
        end = (err.shape[0] - 1, err.shape[1] // 2)

        path = pyastar2d.astar_path(err, start, end, allow_diagonal=True)
        mask = np.ones((block_size, block_size), dtype=ref_block.dtype)
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
    get_min_cut_patch_mask_horizontal = lambda b1, b2, bs, os, bc: (  # ignore blend config (last arg)
        get_min_cut_patch_mask_horizontal_jena2020(b1, b2, bs, os))


def avg_squared_diff(block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
    err = block1 - block2
    err **= 2
    err = err.mean(2)
    return err


def adjust_errors_func_inplace(errors: np.ndarray) -> None:
    # make penalty func steeper
    # TODO consider making this a custom function, or adding params in the config to control it
    errors **= 2


def adjust_errors_for_pystar2d_inplace(errors: np.ndarray, block_len: NumPixels) -> None:
    """Adjust the errors so that 1s can be used to create 'free-corridors'"""

    # scale to integer color range and offset by 1 (works for both RGB and LAB)
    #   this is done so that the distance from 0 to the smallest possible error
    #   keeps the same proportion relative to other error values
    #   otherwise there would be a relative penalty mismatch for pixels with error equal to zero
    #   when offsetting the errors, which is required for both pyastar2d (min. value accepted as weight is 1)
    errors *= 255
    errors += 1

    # make the lowest value big enough for 1 to be negligible (paths created w/ 1s become "free-corridors")
    errors *= block_len ** 2


# endregion


# region min cut patch aliases

def get_min_cut_patch_horizontal(ref_block, patch_block, gen_params: GenParams):
    return get_4way_min_cut_patch(
        True, False, False, False,
        ref_block, patch_block, gen_params
    )


def get_min_cut_patch_vertical(ref_block, patch_block, gen_params: GenParams):
    return get_4way_min_cut_patch(
        False, False, True, False,
        ref_block, patch_block, gen_params
    )


def get_min_cut_patch_both(ref_block, patch_block, gen_params: GenParams):
    return get_4way_min_cut_patch(
        True, False, True, False,
        ref_block, patch_block, gen_params
    )


# endregion


def get_seam_mask_from_patch_weights(patch_weights: np.ndarray, blends_into_patch: bool) -> np.ndarray:
    if blends_into_patch:
        # Has blending, compute distance to 0.5
        seam_mask = 1 - np.abs(.5 - patch_weights) * 2
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
