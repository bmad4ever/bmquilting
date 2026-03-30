from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from functools import lru_cache
from enum import Enum
import numpy as np
import pyastar2d
import cv2

from .common import (
    avg_squared_diff, adjust_errors_for_pystar2d_inplace,
    _filter_candidate_patches, _select_a_random_patch, blend_with_mask, update_seams_map_view)
from .seams_blur import BlendConfig, gradients_differences_at_the_seam, create_adaptive_blend_mask
from .common import NumPixels, Percentage, PatchIdx
from .shmem_utils import SharedTextureList


# region ==== CONFIG DATACLASSES ====

class AstarVariant(Enum):
    """Options for the minimum cut patch search algorithm."""
    DEFAULT = "default"
    """Uses the default pyastar2d with diagonals set to true."""
    ORTHO_Y = "ortho_y"
    """Uses pyastar2d with ORTHOGONAL_Y heuristic override and diagonals set to true."""
    NONE = "none"
    """Bypasses seam computation."""


@dataclass(frozen=True, slots=True)
class CircularPatchParams:
    """
    Parameters for defining a circular patch with pre-calculated dimensions.

    note: values are set up to work with cv2.circle.

    :ivar diameter: The total width/height of the patch (must be odd).
    :ivar overlap_ratio: Percentage of the radius within the overlapping area (0.0 to 1.0).

    :ivar radius: The radius of the patch, calculated as ``diameter // 2``.
    :ivar overlap_radius: The radial distance that overlaps with adjacent patches.
    :ivar non_overlap_radius: The radius of the patch excluding the overlap area.
    :ivar warped_len: The length of the patch along the y-axis when warped.
    """

    diameter: NumPixels
    overlap_ratio: Percentage

    # These are default-initialized but overwritten in __post_init__
    overlap_radius: NumPixels = 0
    non_overlap_radius: NumPixels = 0
    radius: NumPixels = 0
    warped_len: NumPixels = 0

    def __post_init__(self):
        if self.diameter % 2 != 1:
            raise ValueError(f"diameter={self.diameter} must be odd to ensure a symmetric center.")

        r_val = self.diameter // 2
        ov_r_val = round(r_val * self.overlap_ratio)

        object.__setattr__(self, "radius", r_val)
        object.__setattr__(self, "overlap_radius", ov_r_val)
        object.__setattr__(self, "non_overlap_radius", r_val - ov_r_val)
        object.__setattr__(self, "warped_len", round(r_val * 2 * np.pi))

    @property
    def block_size(self) -> NumPixels:
        """
        The size of the square bounding box containing the circular patch.
        :return: The diameter of the patch.
        """
        return self.diameter

    @property
    def center(self) -> NumPixels:
        """
        :return: The pixel index of the center (usable with cv2.circle).
        """
        return self.radius

    @property
    def center_2d_f(self) -> tuple[float, float]:
        """
        :return: The pixel indices of the center as a floats (usable with cv2.warpPolar).
        """
        center = float(self.center)
        return center, center


@dataclass(frozen=True, slots=True)
class CircularPatchingConfig:
    patch_params: CircularPatchParams
    blend_config: BlendConfig | None
    tolerance: Percentage
    outer_corners_weighted_template_matching: bool

    spacing_factor: Percentage
    """
    Spacing between the centers of two horizontal adjacent patches in the lattice with respect to the radius size.

    May affect reproducibility of some algorithms when set to ~1.12 or lower.
    """

    astar_heur: int
    """
    Heuristic for the astar algorithm used when computing seams.

        Available options via the **advanced** class method:
      - `DEFAULT`: Uses Heuristic.DEFAULT.
      - `ORTHO_Y`: Uses Heuristic.ORTHOGONAL_Y.
      - `NONE`: No seams are computed.
    """

    _error_func: Callable = field(init=True, repr=False)
    """
    Function used to compute the errors between overlapping patches, used to compute the seams.

    A custom method may be provided via the **advanced** class method.
    """

    @classmethod
    def with_seams(cls, patch_params: CircularPatchParams, tolerance: Percentage, spacing_factor: Percentage
                   ) -> CircularPatchingConfig:
        blend_config = BlendConfig.auto_blend_config_1(
            patch_params.block_size,
            patch_params.overlap_radius,
            True)
        return cls(
            patch_params=patch_params,
            blend_config=blend_config,
            tolerance=tolerance,
            outer_corners_weighted_template_matching=False,
            spacing_factor=spacing_factor,
            astar_heur=pyastar2d.Heuristic.DEFAULT,
            _error_func=avg_squared_diff
        )

    @classmethod
    def with_feathering(cls, patch_params: CircularPatchParams, tolerance: Percentage, spacing_factor: Percentage
                        ) -> CircularPatchingConfig:
        blend_config = BlendConfig.auto_blend_config_1(
            patch_params.block_size,
            patch_params.overlap_radius,
            True)
        return cls(
            patch_params=patch_params,
            blend_config=blend_config,
            tolerance=tolerance,
            outer_corners_weighted_template_matching=False,
            spacing_factor=spacing_factor,
            astar_heur=3,
            _error_func=avg_squared_diff
        )

    @classmethod
    def advanced(cls, patch_params: CircularPatchParams, blend_config: BlendConfig,
                 tolerance: Percentage, spacing_factor: Percentage,
                 a_star_variant: AstarVariant,
                 outer_corners_weighted_template_matching: bool = False,
                 custom_error_func: Callable = None
                 ) -> CircularPatchingConfig:
        astar_variant_map = {
            AstarVariant.DEFAULT: pyastar2d.Heuristic.DEFAULT,
            AstarVariant.ORTHO_Y: pyastar2d.Heuristic.ORTHOGONAL_Y,
            AstarVariant.NONE: 3
        }
        return cls(
            patch_params=patch_params,
            blend_config=blend_config,
            tolerance=tolerance,
            outer_corners_weighted_template_matching=outer_corners_weighted_template_matching,
            spacing_factor=spacing_factor,
            astar_heur=astar_variant_map[a_star_variant],
            _error_func=custom_error_func if custom_error_func is not None else avg_squared_diff
        )

    def __post_init__(self):
        if 0.0 > self.tolerance:
            raise ValueError(f"{self.tolerance = } tolerance should be greater than 0.")

    @property
    def blend_into_patch(self) -> bool:
        return self.blend_config is not None

    @property
    def spacing(self) -> int:
        return round(self.patch_params.radius * self.spacing_factor)

    @property
    def use_seams(self) -> bool:
        return self.astar_heur <= 2

# endregion ==== CONFIG DATACLASSES ====


@lru_cache(maxsize=2)
def _get_annular_mask(patch_params: CircularPatchParams, dtype = np.float32) -> np.ndarray:
    """
    :return: Binary annular mask for the circular patch overlapping region.
    """
    block_size, center = patch_params.block_size, patch_params.center
    radius, non_overlap_radius = patch_params.radius, patch_params.non_overlap_radius
    annulus_roi_block = np.zeros((block_size, block_size), dtype=dtype, order='C')
    cv2.circle(annulus_roi_block, (center, center), radius, (1,), -1)
    cv2.circle(annulus_roi_block, (center, center), round(non_overlap_radius), (0,), -1)
    return annulus_roi_block


@lru_cache(maxsize=1)
def _get_circle_mask(patch_params: CircularPatchParams) -> np.ndarray:
    """
    :return: float32 circle binary mask
    """
    block_size, center = patch_params.block_size, patch_params.center
    circle_mask = np.zeros((block_size, block_size), dtype=np.float32, order='C')
    cv2.circle(circle_mask, (center, center), patch_params.radius, (1.0,), -1)
    return circle_mask


def _compute_radial_seam_mask(circ_patching_config: CircularPatchingConfig,
                              roi: np.ndarray, block: np.ndarray, patch: np.ndarray, _tmp: np.ndarray) -> np.ndarray:
    pp = circ_patching_config.patch_params
    f_radius = float(pp.radius)

    # Compute Errors prior to warping
    errors = circ_patching_config._error_func(block, patch)

    # Find Seam using Polar Coordinates
    center = circ_patching_config.patch_params.center_2d_f
    warped_len = circ_patching_config.patch_params.warped_len
    warp_flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST

    # Warp errors and overlap ROI to polar coordinates & Find "optimal" seam
    polar_errors, polar_roi = (
        cv2.warpPolar(i, (pp.radius, warped_len), center, f_radius, warp_flags)
        for i in [errors, roi])
    mask = _min_cut_circ(polar_errors, polar_roi, pp.non_overlap_radius)

    # Warp mask back to cartesian coords & Compute patched result
    mask = cv2.warpPolar(mask, roi.shape[:2], center, f_radius, cv2.WARP_INVERSE_MAP | warp_flags)
    patched = blend_with_mask(block, patch, mask, out=_tmp)
    # mind that even if blend into patch option is not set, patched variable is re-used for the vignette later

    # (optional) Blend mask with respect to gradients diff.
    if circ_patching_config.blend_into_patch:
        tdiff_map = gradients_differences_at_the_seam(circ_patching_config.blend_config.sobel_kernel_size,
                                                      mask, block, patch, patched,
                                                      circ_patching_config.blend_config.grad_diff_func,
                                                      _tmp=patched)

        if circ_patching_config.blend_config.use_blur_radii_limiter:
            radii_limiter = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3,
                                                  dstType=cv2.CV_32F)
            cv2.GaussianBlur(radii_limiter, (5, 5), 5, dst=radii_limiter)
            # note: alternatively, I could cache a limiter using annular mask, this would be more efficient;
            #       however, such solution may have a blur derived seam at the transition between the
            #       "filled" and "unfilled" sections of the patch.
        else:
            # mask out external area while allowing blurring on the overlapping region
            radii_limiter = roi * circ_patching_config.patch_params.diameter

        mask = create_adaptive_blend_mask(
            tdiff_map=tdiff_map,
            mc_mask_overlap=mask,
            blend_config=circ_patching_config.blend_config,
            radii_limiter_mask=radii_limiter,
            dtype=block.dtype
        )
    return mask

def get_bbox_idx(x: int, y: int, patch_params: CircularPatchParams) -> tuple[slice, slice]:
    radius = patch_params.radius
    return np.s_[y - radius:y + radius + 1, x - radius:x + radius + 1]


def process_patch_at_location(image: np.ndarray, filled_mask: np.ndarray, seams_map: np.ndarray,
                              lookup_textures: list[np.ndarray] | SharedTextureList,
                              x: int, y: int,
                              config: CircularPatchingConfig,
                              rng: np.random.Generator) -> tuple[PatchIdx, np.ndarray]:
    """
    Finds and applies a single circular patch at location (x, y).
    Updates image, filled_mask, and seams_map arrays.

    :param image: Image being patched or generated.
        Should be of type float32, with values in the interval [0, 1].
    :param filled_mask: Mask that keeps track of the area already filled.
        Should be of type float32, with values in the interval [0, 1].
    :param seams_map: Seams map, should be of type float32 with values in the interval [0, 1].
    :param x: x position on the image array (not the index in a lattice).
    :param y: y position on the image array (not the index in a lattice).
    :param lookup_textures: source images used for patch extraction.
        Should be of type float32, sharing the same type as the provided image.

    :return: (patch indices, mask)
    """
    # --- Fetch Config Parameters ---
    pp = config.patch_params
    bbox_idx = get_bbox_idx(x, y, pp)  # bbox for the area being patched

    # Extract the current image block and the region of interest (ROI)
    block = image[bbox_idx]
    roi = filled_mask[bbox_idx]
    seams = seams_map[bbox_idx]

    # Get the annular mask used for this patch
    annulus_roi_block = _get_annular_mask(pp)

    roi = annulus_roi_block * roi  # confines the ROI to the annulus (overlap) region
    tmpl_mask = roi                # mask used w/ template matching

    # (optional) Weighted Template Matching Setup
    if config.outer_corners_weighted_template_matching:
        corners_mask = _find_annulus_outer_corners(roi, pp.non_overlap_radius, pp.radius)
        if corners_mask is not None:
            tmpl_mask = _distance_to_points(corners_mask)
            tmpl_mask *= roi
        del corners_mask

    # Find the best matching patch from the lookup texture
    best_text_idx, best_y, best_x = _find_circular_patch(lookup_textures, block, tmpl_mask, config, rng)
    patch: np.ndarray = lookup_textures[best_text_idx][best_y:best_y+pp.block_size, best_x:best_x+pp.block_size]

    # AUX may be used for the patched block (w/ raw seam, no blending) and for the vignette
    #   no seams + no vignette is not an expected option combination; it only makes sense for debug purposes
    #   so the is no need prevent the allocation with a conditional here
    aux = np.empty_like(patch)

    # Get mask to blend source and patch (pre vignette)
    mask = _compute_radial_seam_mask(config, roi, block, patch, _tmp=aux) if config.use_seams \
      else _get_circle_mask(pp).copy()


    # (optional) Apply Vignette
    if config.blend_into_patch and config.blend_config.use_vignette:
        vignette = _setup_vignette(roi, pp, dst=aux.reshape(aux.shape[0], -1))
        np.minimum(mask, vignette, out=mask)

    # Update Seams & Filled Mask State
    update_seams_map_view(seams, mask, config.blend_into_patch)
    np.maximum(mask, filled_mask[bbox_idx], out=filled_mask[bbox_idx])

    # Paste into the Source Image
    np.subtract(1, mask, out=mask)  # mask <- (1 - mask) so that image[bbox_idx] is sent as fg
    blend_with_mask(patch, image[bbox_idx], mask, out=image[bbox_idx])

    return (best_text_idx, best_y, best_x), mask


# region "Min Cut" Related Functions  ____START

def _min_cut_circ(errors: np.ndarray, roi: np.ndarray, non_overlap_radius: NumPixels,
                  heur_override:pyastar2d.Heuristic=0) -> np.ndarray:
    """
    @param errors: warped matrix with the pre-computed errors
    @param roi: warped binary mask of the region of interest
    """
    adjust_errors_for_pystar2d_inplace(errors, errors.shape[0])

    offset_row = _find_first_2adjacent_all_zero_rows(roi[:, non_overlap_radius:])
    if offset_row is not None:
        errors = np.roll(errors, -offset_row, axis=0)
        roi = np.roll(roi, -offset_row, axis=0)

    start_x = end_x = errors.shape[1] - 1
    if offset_row is None:
        # when no empty row is found, search for the cut endpoints at the top & bottom so that the path is not broken
        start_x, end_x = _find_min_cut_circ_endpoints(errors, roi, heur_override)

    errors[:, -1] = roi[:, -1] * errors[:, -1] + (1 - roi[:, -1])  # bottom holes escape path
    errors[:, :-1][roi[:, :-1] == 0] = np.inf  # don't travel outside the mask, unless via the escape path

    start = (0, start_x)
    end = (errors.shape[0] - 1, end_x)

    path = pyastar2d.astar_path(errors, start, end, allow_diagonal=True, heuristic_override=heur_override)
    mask = np.zeros_like(roi)

    for i, j in path:  # draw path
        mask[i, j] = 1

    cv2.floodFill(mask, None, (0, 0), (1,))
    if offset_row is not None:
        mask = np.roll(mask, offset_row, axis=0)

    return mask


def _find_first_2adjacent_all_zero_rows(mask:np.ndarray) -> int | None:
    """:return: the index of the second all zero row, or None if none found"""
    prev = False
    for i, row in enumerate(mask):
        current = all(pixel == 0 for pixel in row)
        if prev and current:
            return i
        prev = current
    return None


def _x_squared_distance_1d(a_1d: np.ndarray, b_1d: np.ndarray) -> np.ndarray:
    """Computes the squared distance matrix based on 1D arrays of X-coordinates."""
    # Reshape for broadcasting: (N, 1) and (1, M)
    a_coords = a_1d.reshape(-1, 1)
    b_coords = b_1d.reshape(1, -1)
    squared_dist_matrix = (a_coords - b_coords) ** 2  # gets rid of the sign
    return squared_dist_matrix


def _find_min_cut_circ_endpoints(errors: np.ndarray, roi: np.ndarray, heur_override: pyastar2d.Heuristic=0) -> tuple[int, int]:
    """
    Consider the roi as the following sections: | A | B | C | D |
    This functions rolls & crops it to get the section: | D | A |
    Then finds the best path to go from the start of D to the end of A.
    The points in the middle of the path, where D and A meet, are the points of interest.

    There may be multiple points at this boarder, since A* can noodle around.
    The coordinates of the pair of points nearest each other at y_center and y_center -1 is returned.
    If multiple pairs have the same minimum distance, only one is returned.

    :param errors: polar unwrapped errors
    :param roi: polar unwrapped overlap roi
    :return: a tuple containing the top and bottom row endpoints (column idx) for the min cut
    """
    import pyastar2d

    err_len_div4 = errors.shape[0] // 4
    errors = np.roll(errors, -err_len_div4, axis=0)[-err_len_div4*2:, :]
    roi = np.roll(roi, -err_len_div4, axis=0)[-err_len_div4*2:, :]

    maze = np.ones((errors.shape[0] + 2, errors.shape[1]), dtype=np.float32)
    maze_area = maze[1:-1, :]
    maze_area[:, :] = errors

    maze[0, :] = 1
    maze[-1, :] = 1
    maze[1:-1, -1] = roi[:, -1] * maze[1:-1, -1] + (1 - roi[:, -1])  # bottom holes escape path
    maze[1:-1, :-1][roi[:, :-1] == 0] = np.inf  # don't travel outside the mask, unless via the escape path

    start = (0, maze.shape[1] - 1)
    end = (maze.shape[0] - 1, maze.shape[1] - 1)

    path = pyastar2d.astar_path(maze, start, end, allow_diagonal=True, heuristic_override=heur_override)

    # Compute the distance matrix between points (y is ignored since it is fixed)
    err_len_div4_minus1 = err_len_div4 - 1
    points_of_interest_top = [x for (y, x) in path if y - 1 == err_len_div4]
    points_of_interest_bottom = [x for (y, x) in path if y - 1 == err_len_div4_minus1]
    dist_matrix = _x_squared_distance_1d(np.array(points_of_interest_top), np.array(points_of_interest_bottom))

    # Find the indices of the minimum distance
    min_index = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    top_point = points_of_interest_top[min_index[0]]
    bottom_point = points_of_interest_bottom[min_index[1]]

    return top_point, bottom_point

# endregion "Min Cut" Related Functions  ____END


def _find_circular_patch(lookup_textures: list[np.ndarray] | SharedTextureList,
                         block: np.ndarray, mask: np.ndarray,
                         params: CircularPatchingConfig, rng: np.random.Generator) -> PatchIdx:
    """

    :param lookup_textures: textures used for patch searching.
    :param block: 2D slice of the texture being patched containing the overlapping region.
        The array should be of type float.
    :param mask: Mask of the overlapping region. It should be of type float, having values in the interval [0, 1].
        The mask can be weighted to prioritize or neglect certain spots within the overlapping region.
    :param rng: random generator for results reproducibility.
    :return: texture index and (y, x) top-left coordinates of the block sized patch containing the overlapping region.
    """
    block_size, tolerance = params.patch_params.block_size, params.tolerance
    err_mats: list[np.ndarray | None] = []
    global_min_error = np.inf

    # --- PASS 1: Compute errors for each texture & find the absolute global minimum error ---
    for texture in lookup_textures:

        # Skip if texture is too small for a block
        if texture.shape[0] < block_size or texture.shape[1] < block_size:
            err_mats.append(None)  # Keep placeholder for alignment
            continue

        # Compute errors & add to errors list
        # notes: only TM_SQDIFF and TM_CCORR_NORMED accept a mask, which is required even if not weighted
        err_mat = cv2.matchTemplate(image=texture, templ=block, mask=mask, method=cv2.TM_SQDIFF)
        err_mat = np.maximum(err_mat, 1e-8)  # clip floor to zero
        err_mats.append(err_mat)

        global_min_error = min(global_min_error, np.min(err_mat))  # update minimum

    # --- Check for impossible match ---
    if global_min_error == np.inf:
        raise ValueError("Could not find a suitable patch in any lookup texture "
                         "(all textures were too small or had matching issues).")

    # --- PASS 2: Collect all candidates within the final tolerance window ---
    final_candidates = _filter_candidate_patches(err_mats, global_min_error, tolerance)
    return _select_a_random_patch(final_candidates, rng)

# region Weight Template Matching Auxiliary Funcs ____START

def _find_annulus_outer_corners(mask: np.ndarray, inner_radius: int, outer_radius: int) -> np.ndarray | None:
    """
    :param mask: the mask w/ the annulus
    :param inner_radius: inner radius of the annulus
    :param outer_radius: outer radius of the annulus
    :return: uint8 mask with the painted corners if any were found; or None, if no corners were found
    """
    margin: int = 4
    extended_mask = np.zeros((margin * 2 + mask.shape[0], margin * 2 + mask.shape[1]), dtype=mask.dtype)
    extended_mask[4:-4, 4:-4] = mask

    dst = cv2.cornerHarris(extended_mask, blockSize=2, ksize=5, k=0.06)

    # Mask out the center
    center = (outer_radius + margin, outer_radius + margin)
    cut_off_radius = (outer_radius + inner_radius) // 2
    cv2.circle(dst, center, cut_off_radius, 0, -1)

    # Crop to mask region
    dst = dst[4:-4, 4:-4]

    # Threshold
    threshold = 1e-4
    bool_mask = dst > threshold

    if not bool_mask.any():
        return None

    return bool_mask.astype(np.uint8)


def _distance_to_points(points_mask: np.ndarray) -> np.ndarray:
    """
    :param points_mask: uint8 binary mask with the points to set the distance to.
    :return: a float32 mask with a gaussian blurred inverse normalized distance transform, i.e.
        the closer to the points the closer to one.
    """
    np.subtract(1, points_mask, out=points_mask)
    distance_map = cv2.distanceTransform(points_mask, cv2.DIST_L2, 5, dst=points_mask)
    ksize = min(min(distance_map.shape)//4 + 1, 15)
    cv2.GaussianBlur(distance_map, (ksize, ksize), sigmaX=0, sigmaY=0, dst=distance_map)
    dst_norm = distance_map.astype(np.float32)
    cv2.normalize(dst_norm, dst_norm, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    np.subtract(1, dst_norm, out=dst_norm)
    return dst_norm

# endregion Weight Template Matching Auxiliary Funcs ____END


def set_random_patch_at_location(image: np.ndarray, filled_mask: np.ndarray,
                                 lookup_textures: list[np.ndarray] | SharedTextureList,
                                 x: int, y: int,
                                 config: CircularPatchingConfig,
                                 rng: np.random.Generator) -> tuple[PatchIdx, np.ndarray]:
    """
        Note: despite not updating seams, this function is prepared to handle non-empty image sources so
        that it can be repurposed for other use cases besides selecting the 1st patch when generating a texture.

        :return: patch_idxs, mask
    """
    radius = config.patch_params.radius
    center = config.patch_params.center
    block_size = config.patch_params.block_size
    y1, y2, x1, x2 = y - radius, y + radius + 1, x - radius, x + radius + 1

    # Random patch selection
    rnd_text_idx = int(rng.integers(len(lookup_textures)))
    rnd_lookup = lookup_textures[rnd_text_idx]
    h, w = rnd_lookup.shape[:2]
    rand_h = int(rng.integers(h - block_size))
    rand_w = int(rng.integers(w - block_size))
    start_block = rnd_lookup[rand_h:rand_h + block_size, rand_w:rand_w + block_size]

    # Create circular mask
    mask = np.zeros((block_size, block_size), dtype=filled_mask.dtype)
    cv2.circle(mask, (center, center), radius, (1,), -1)

    # Update image & Filled mask
    bbox_idx = np.s_[y1:y2, x1:x2]
    np.maximum(mask, filled_mask[bbox_idx], out=filled_mask[bbox_idx])
    np.subtract(1, mask, out=mask)
    blend_with_mask(start_block, image[bbox_idx], mask, out=image[bbox_idx])

    return (rnd_text_idx, rand_h, rand_w), mask


def _setup_vignette(roi: np.ndarray, patch_params: CircularPatchParams, dst: np.ndarray = None) -> np.ndarray:
    center = (patch_params.center, patch_params.center)
    blur_k = max(3, patch_params.radius // 4 + 1)

    # Patch only region ( roi complement + center )
    aux = (roi>0).astype(np.uint8)      # set as uint8 for usage w/ distanceTransform
    aux = cv2.bitwise_xor(aux, _get_annular_mask(patch_params, dtype=np.uint8), dst=aux)
    cv2.circle(aux, center, patch_params.non_overlap_radius, (1,), -1)
    np.bitwise_xor(aux, 1, out=aux)     # invert for usage w/ distanceTransform

    # Apply radius sized blur
    vignette = cv2.distanceTransform(aux, cv2.DIST_L2, 3, dst=dst, dstType=cv2.CV_32F)
    limit = patch_params.overlap_radius
    np.clip(vignette, 0, limit, out=vignette)
    np.subtract(limit, vignette, out=vignette)
    np.divide(vignette, limit, out=vignette)

    # Blur vignette but keep patch only regions w/ values equal to 1
    cv2.blur(vignette, (blur_k, blur_k), dst=vignette)
    vignette *= aux ; aux -= 1 ; vignette += aux              # result should be the same as: vignette[aux<1] = 1
    np.clip(vignette, 0, 1, out=vignette)

    return vignette
