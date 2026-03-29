from enum import Enum

from .seams_blur import (
    auto_blend_config_1, auto_blend_config_2,
    gradients_differences_at_the_seam, create_adaptive_blend_mask, _get_radii_limiter,
)
from ..common_types import NumPixels, PatchIdx, Percentage, BlendConfig
from .shmem_utils import SharedTextureList
from .mask_utils import blend_with_mask


from dataclasses import dataclass, field, asdict
from collections.abc import Callable
from functools import lru_cache
from numpy.random import Generator
import numpy as np
import pyastar2d
import cv2 as cv




import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

epsilon = np.finfo(float).eps


def debug_resize(arr, factor=8):
    return cv.resize(arr, (arr.shape[1] * factor, arr.shape[0] * factor))


# region ==== CONFIG DATACLASSES ====

type CallableSeamsAlgorithm = Callable[[np.ndarray, np.ndarray, int, int, BlendConfig], np.ndarray]
"""Algorithm that computes the seam mask. Its arguments are: ref. block, patch block, block_size, overlap, and blend config."""

class SeamsAlgorithm(Enum):
    """Options for the minimum cut patch search algorithm."""
    ASTAR = "astar"
    """Implemented using pyastar2d; check documentation for more information."""
    MIN_CUT = "min_cut"
    """Implemented using numpy; adapted from jena2020."""
    NONE = "none"
    """Bypasses seam computation."""


@dataclass(frozen=True, slots=True)
class SquarePatchingBlendConfig(BlendConfig):
    use_blur_radii_guess_pathfind_limiter: bool = True
    """
    Attempts to mitigate seam blurring artifacts BEFORE seam computation.
    Prior to computing the seam, make an educated guess of the potential max blur radius.
    When computing the seam the overlapping area is further constrained with respect to this guess to avoid having 
    the seam go near the edges of the overlapping area.

    Only applicable when using pyastar2d to compute the seam.
    """


@dataclass(frozen=True, slots=True)
class SquarePatchingConfig:
    """
    Data used across multiple quilting subroutines.
    Used in quilting.py and make_seamless.py
    """
    block_size: NumPixels
    overlap: NumPixels
    tolerance: Percentage
    blend_config: SquarePatchingBlendConfig | None
    vignette_on_match_template: bool
    """whether to use the blending vignette as a mask when searching for a matching patch"""

    _compute_seam_callable: Callable = field(init=True, repr=False)
    """
    Internal callable with the algorithm that computes a seam.

    Available options via the **advanced** class method:
      - `ASTAR`: A* grid based solution. Seams can backtrack. (C backend)
      - `MIN_CUT`: Purist solution; seams cannot backtrack.
      - `NONE`: No seams are computed.

    A custom method may be provided via the **custom** class method.
    """

    _error_func: Callable = field(init=True, repr=False)
    """
    Function used to compute the errors between overlapping patches, used to compute the seams.
    
    A custom method may be provided via the **advanced** class method.
    """

    _mt_error_adjust: Callable[[float], float] = field(init=False)
    """
    Function to adjust the errors with respect to the match_template_method selected.
    Not meant to be set by the user; it is set automatically via __post_init__.
    """

    match_template_method: int = cv.TM_SQDIFF
    """
    Match template method used when searching for a patch with a matching overlap section.
    Only TM_SQDIFF and TM_CCOEFF_NORMED are supported.
    """


    @classmethod
    def with_seams(cls, block_size: NumPixels, overlap: NumPixels, tolerance: Percentage) -> "SquarePatchingConfig":
        """
        Uses seams whose blur size bounds are heuristically determined by the provided argument;
        they are complemented with a vignette-like mask.

        Uses A* to compute the seams, where the error is the channels' averaged squared difference squared
        (^4; you read that correctly, it is not a typo).
        """
        blend_config = auto_blend_config_1(block_size, overlap, True)
        blend_config = SquarePatchingBlendConfig(**asdict(blend_config))

        return cls(
            block_size=block_size,
            overlap=overlap,
            tolerance=tolerance,
            blend_config=blend_config,
            vignette_on_match_template=False,
            match_template_method=cv.TM_SQDIFF,
            _compute_seam_callable=get_seam_mask_horizontal_astar,
            _error_func=avg_squared_diff
        )

    @classmethod
    def with_feathering(cls, block_size: NumPixels, overlap: NumPixels, tolerance: Percentage) -> "SquarePatchingConfig":
        """Does not compute seams, the patch is blended using a vignette-like mask."""
        blend_config = auto_blend_config_1(block_size, overlap, True)
        blend_config = SquarePatchingBlendConfig(**asdict(blend_config))

        return cls(
            block_size=block_size,
            overlap=overlap,
            tolerance=tolerance,
            blend_config=blend_config,
            vignette_on_match_template=False,
            match_template_method=cv.TM_SQDIFF,
            _compute_seam_callable=get_seam_mask_none,
            _error_func=avg_squared_diff
        )

    @classmethod
    def advanced(cls, block_size, overlap, tolerance, blend_config,
                 seam_algorithm: SeamsAlgorithm = SeamsAlgorithm.ASTAR,
                 vignette_on_match_template: bool = False,
                 match_template_method=cv.TM_SQDIFF,
                 custom_error_func: Callable = None
                 ) -> "SquarePatchingConfig":
        seams_algo_map = {
            SeamsAlgorithm.ASTAR: get_seam_mask_horizontal_astar,
            SeamsAlgorithm.MIN_CUT: get_seam_mask_horizontal_min_cut,
            SeamsAlgorithm.NONE: get_seam_mask_none,
        }
        return cls(
            block_size=block_size,
            overlap=overlap,
            tolerance=tolerance,
            blend_config=blend_config,
            vignette_on_match_template=vignette_on_match_template,
            match_template_method=match_template_method,
            _compute_seam_callable=seams_algo_map[seam_algorithm],
            _error_func= custom_error_func if custom_error_func is not None else avg_squared_diff
        )

    @classmethod
    def custom(cls, block_size, overlap, tolerance, blend_config,
               custom_func: CallableSeamsAlgorithm,
               vignette_on_match_template: bool = False,
               match_template_method=cv.TM_SQDIFF
               ) -> "SquarePatchingConfig":
        """Create a config using a user‑supplied min‑cut function."""
        return cls(
            block_size=block_size,
            overlap=overlap,
            tolerance=tolerance,
            blend_config=blend_config,
            vignette_on_match_template=vignette_on_match_template,
            match_template_method=match_template_method,
            _compute_seam_callable=custom_func,
            _error_func=lambda _: 0  # IGNORED; supposes that the custom seam function does not access it
        )

    def __post_init__(self):
        if not (0.0 <= self.tolerance <= 1.0):
            raise ValueError(f"{self.tolerance = } tolerance should be in the [0,1] range.")

        # Bypass the frozen restriction to setup errors adjust function with respect to template matching method
        if self.match_template_method == cv.TM_SQDIFF:
            adjuster = lambda e: e
        elif self.match_template_method == cv.TM_CCOEFF_NORMED:  # [-1, 1] , where 1 is the best possible match
            adjuster = lambda e: 1 - e  # adjust to only positive values where smaller values mean a better match
        else:
            raise ValueError(f"{self.match_template_method = } is not supported.\n"
                             f"Only TM_SQDIFF and TM_CCOEFF_NORMED are supported.")

        object.__setattr__(self, '_mt_error_adjust', adjuster)

    def _compute_seam(self, source: np.ndarray, patch: np.ndarray) -> np.ndarray:
        return self._compute_seam_callable(source, patch, self.block_size, self.overlap, self)

    @property
    def blend_into_patch(self) -> bool:
        return self.blend_config is not None

    @property
    def bo(self) -> tuple[NumPixels, NumPixels]:
        return self.block_size, self.overlap

    @property
    def bot(self) -> tuple[NumPixels, NumPixels, Percentage]:
        return self.block_size, self.overlap, self.tolerance


# endregion ==== CONFIG DATACLASSES ====


# region   PATCH SEARCH & COMPUTE SEAMS + related auxiliary methods

def find_patch_vx(overlaps_left: bool,
                  overlaps_right: bool,
                  overlaps_top: bool,
                  overlaps_bottom: bool,
                  ref_block: np.ndarray,  # gen. texture view at the patch to generate location
                  lookup_textures: list[np.ndarray] | SharedTextureList,
                  patching_config: SquarePatchingConfig,
                  rng: np.random.Generator
                  ) -> np.ndarray:
    """
    Calls find_patch_vx_idx and returns the patch instead of its indices.
    See find_patch_vx_idx documentation.

    :return: a block sized texture patch.
    """
    best_texture_idx, best_y, best_x = find_patch_vx_idx(
        overlaps_left, overlaps_right, overlaps_top, overlaps_bottom, ref_block, lookup_textures, patching_config, rng)

    # Extract the patch from the winning texture
    winning_texture = lookup_textures[best_texture_idx]
    block_size = patching_config.block_size
    return winning_texture[best_y:best_y + block_size, best_x:best_x + block_size]


@lru_cache(maxsize=4)
def _get_overlap_mask(block_size: NumPixels, overlap: NumPixels,
                      overlaps_left: bool, overlaps_right: bool,
                      overlaps_top: bool, overlaps_bottom: bool) -> np.ndarray:
    mask = np.zeros((block_size, block_size), dtype=np.float32)

    if overlaps_left:
        mask[:, :overlap] = 1.0
    if overlaps_right:
        mask[:, -overlap:] = 1.0
    if overlaps_top:
            mask[:overlap, :] = 1.0
    if overlaps_bottom:
            mask[-overlap:, :] = 1.0

    return mask

@lru_cache(maxsize=4)
def _get_vignetted_overlap_mask(block_size: NumPixels, overlap: NumPixels,
                                overlaps_left: bool, overlaps_right: bool,
                                overlaps_top: bool, overlaps_bottom: bool) -> np.ndarray:
    overlap_mask = _get_overlap_mask(block_size, overlap, overlaps_left, overlaps_right, overlaps_top, overlaps_bottom)
    mask = _patch_blending_vignette(
        block_size, overlap,
        overlaps_left,
        overlaps_right,
        overlaps_top,
        overlaps_bottom
    )

    mask = .5 - mask  # mind that the above mask can't be edited directly because it is cached
    mask *= 2.0
    np.clip(mask, 0.0, 1.0, out=mask)
    np.subtract(1.0, mask, out=mask)
    mask *= overlap_mask
    return mask


def find_patch_vx_idx(overlaps_left: bool,
                      overlaps_right: bool,
                      overlaps_top: bool,
                      overlaps_bottom: bool,
                      ref_block: np.ndarray,  # gen. texture view at the patch to generate location
                      lookup_textures: list[np.ndarray] | SharedTextureList,
                      patching_config: SquarePatchingConfig,
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

    :param patching_config: besides the block and overlap size, it provides the tolerance (percentage) which will
        define what is the acceptable range for the patch selection with respect to the best possible patch errors.
        The higher the tolerance, the more leeway the function has to select a "worse" patch.

    :return: best_texture_idx, best_y, best_x
    """
    block_size, overlap, tolerance = patching_config.bot

    # List to store error matrices (Pass 1) or final candidates (Pass 2)
    err_mats: list[np.ndarray | None] = []
    global_min_error = np.inf

    # Get Overlap Mask
    mask = (
        _get_vignetted_overlap_mask(block_size, overlap, overlaps_left, overlaps_right, overlaps_top, overlaps_bottom)
        if patching_config.vignette_on_match_template
        else _get_overlap_mask(block_size, overlap, overlaps_left, overlaps_right, overlaps_top, overlaps_bottom)
    )

    # --- PASS 1: Compute errors for each texture & find the absolute global minimum error ---
    for texture in lookup_textures:

        # Skip if texture is too small for a block
        if texture.shape[0] < block_size or texture.shape[1] < block_size:
            err_mats.append(None)  # Keep placeholder for alignment
            continue

        errs = cv.matchTemplate(image=texture, templ=ref_block, method=patching_config.match_template_method, mask=mask)

        # Adjust errors
        if patching_config.match_template_method == cv.TM_CCOEFF_NORMED:
            errs = 1.0 - errs

        errs = np.maximum(errs, 1e-8)  # clip floor to zero
        err_mats.append(errs)

        # Update global minimum
        text_min, _, _, _ = cv.minMaxLoc(errs)
        global_min_error = min(global_min_error, text_min)

    # --- Check for impossible match ---
    if global_min_error == np.inf:
        logger.warning(f"Global minimum error is {global_min_error}")
        raise ValueError("Could not find a suitable patch in any lookup texture "
                         "(all textures were too small or had matching issues).")

    # --- PASS 2: Collect all candidates within the final tolerance window ---
    final_candidates = _filter_candidate_patches(err_mats, global_min_error, tolerance)
    best_texture_idx, best_y, best_x = _select_a_random_patch(final_candidates, rng)
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
    return final_candidates[c]


def _filter_candidate_patches(err_mats: list[np.ndarray | None],
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


def get_4way_seam_patched(
        overlaps_left: bool,
        overlaps_right: bool,
        overlaps_top: bool,
        overlaps_bottom: bool,
        ref_block: np.ndarray,  # gen. texture view at the patch to generate location
        patch_block, patching_config: SquarePatchingConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    :param ref_block: roi on the texture where the patch will be placed over.
        overlapping regions should have been already filled prior to calling this method.
    :param patch_block: new block to be placed over the ref_block roi.
    :return: patch, mask
    """

    block_size, overlap = patching_config.bo
    masks_max = np.zeros(patch_block.shape[:2], dtype=np.float32)

    def process_block(mask):
        if patching_config.blend_into_patch and patching_config.blend_config.use_vignette:
            vignette = _patch_blending_vignette(
                block_size, overlap, overlaps_left, overlaps_right, overlaps_top, overlaps_bottom)
            np.maximum(mask, vignette, out=mask)  # mind that the mask is a view that may be flipped or rotated
        np.maximum(mask, masks_max, out=masks_max)

    if overlaps_left:
        mask = patching_config._compute_seam(ref_block, patch_block)

        if patching_config.blend_into_patch:
            compute_adaptive_blend_mask(
                ref_block,
                patch_block,
                mask,
                patching_config
            )
        process_block(mask)

    if overlaps_right:
        adj_src = np.fliplr(ref_block)
        adj_ptc = np.fliplr(patch_block)
        mask = patching_config._compute_seam(adj_src, adj_ptc)
        if patching_config.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                patching_config
            )
        mask = np.fliplr(mask)
        process_block(mask)

    if overlaps_top:
        # V , >  counterclockwise rotation
        adj_src = np.rot90(ref_block)
        adj_ptc = np.rot90(patch_block)
        mask = patching_config._compute_seam(adj_src, adj_ptc)
        if patching_config.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                patching_config
            )
        mask = np.rot90(mask, 3)
        process_block(mask)

    if overlaps_bottom:
        adj_src = np.fliplr(np.rot90(ref_block))
        adj_ptc = np.fliplr(np.rot90(patch_block))
        mask = patching_config._compute_seam(adj_src, adj_ptc)
        if patching_config.blend_into_patch:
            compute_adaptive_blend_mask(
                adj_src,
                adj_ptc,
                mask,
                patching_config
            )
        mask = np.rot90(np.fliplr(mask), 3)
        process_block(mask)

    res = blend_with_mask(patch_block, ref_block, masks_max)
    return res, np.subtract(1, masks_max, out=masks_max)


@lru_cache(maxsize=4)
def _patch_blending_vignette(block_size: NumPixels, overlap: NumPixels,
                             left: bool, right: bool, top: bool, bottom: bool) -> np.ndarray:
    margin = 1  # must be small !
    power = 2.5  # controls drop-off
    p = 4  # controls the shape

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

    return np.subtract(1, mask, out=mask)


def get_seam_mask_horizontal_min_cut(ref_block, patch_block, block_size: NumPixels,
                                     overlap: NumPixels, config: SquarePatchingConfig):
    """
    Adapted from jena2020 solution.
    :return: ONLY the mask (not the patched overlap section)
    """
    inf = float('inf')
    err =config._error_func(ref_block[:, :overlap], patch_block[:, :overlap])
    # maintain minIndex for 2nd row onwards and
    min_index = []
    e = [list(err[0])]
    for i in range(1, err.shape[0]):
        # Get min values and args, -1 = left, 0 = middle, 1 = right
        _e = [inf] + e[-1] + [inf]
        _e = np.array([_e[:-2], _e[1:-1], _e[2:]])
        # Get minIndex
        min_arr = _e.min(0)
        min_arg = _e.argmin(0) - 1
        min_index.append(min_arg)
        # Set Eij = e_ij + min_
        e_ij = err[i] + min_arr
        e.append(list(e_ij))

    # Check the last element and backtrack to find path
    path = []
    min_arg = np.argmin(e[-1])
    path.append(min_arg)

    # Backtrack to min path
    for idx in min_index[::-1]:
        min_arg = min_arg + idx[min_arg]
        path.append(min_arg)
    # Reverse to find full path
    path = path[::-1]
    mask = np.zeros((block_size, block_size), dtype=ref_block.dtype)
    for i in range(len(path)):
        mask[i, :path[i] + 1] = 1
    return mask


def update_seams_map_view(seams_map_view: np.ndarray, patch_weights: np.ndarray, blends_into_patch: bool):
    seam_map_block = get_seam_mask_from_patch_weights(patch_weights, blends_into_patch)
    clear_seam_overlapped_by_patch(seams_map_view, patch_weights)
    #seams_map_view += seam_map_block
    np.maximum(seams_map_view, seam_map_block, out=seams_map_view)


def get_seam_mask_horizontal_astar(
        ref_block, patch_block,
        block_size: NumPixels, overlap: NumPixels,
        config: SquarePatchingConfig
        #blend_config: SquarePatchingBlendConfig = None
) -> np.ndarray:
    """
    :param ref_block: block in the source texture being generated (should be normalized)
    :param patch_block: patch that will be pasted over the ref_block

    :return: ONLY the mask (not the patched overlap section)
    """
    # get min and max safety radii
    blend_config = config.blend_config

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

    err = config._error_func(block1_safe_overlap_view, block2_safe_overlap_view)
    #avg_squared_diff(block1_safe_overlap_view, block2_safe_overlap_view)

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


def get_seam_mask_none(
        __, ___,
        block_size: NumPixels, overlap: NumPixels,
        ____
) -> np.ndarray:
    return np.zeros((block_size, block_size), dtype=np.float32)


def avg_squared_diff(block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
    err = block1 - block2
    err **= 2
    err = err.mean(2)
    return err


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


# region   PATCH SEARCH METHODS ALIASES WITH RESPEC TO DIRECTIONS

def get_find_patch_to_the_right_method():
    def vx_right(ref_block, image, patching_config, rng):
        return find_patch_vx(True, False, False, False, ref_block, image, patching_config, rng)

    return vx_right


def get_find_patch_below_method():
    def vx_below(ref_block, image, patching_config, rng):
        return find_patch_vx(False, False, True, False, ref_block, image, patching_config, rng)

    return vx_below


def get_find_patch_both_method():
    def vx_both(ref_block, image, patching_config, rng):
        return find_patch_vx(True, False, True, False, ref_block, image, patching_config, rng)

    return vx_both

# endregion


# region SEAM PATCHING METHODS ALIASES WITH RESPECT TO DIRECTION

def get_seam_patched_horizontal(ref_block, patch_block, patching_config: SquarePatchingConfig):
    return get_4way_seam_patched(
        True, False, False, False,
        ref_block, patch_block, patching_config
    )


def get_seam_patched_vertical(ref_block, patch_block, patching_config: SquarePatchingConfig):
    return get_4way_seam_patched(
        False, False, True, False,
        ref_block, patch_block, patching_config
    )


def get_seam_patched_both(ref_block, patch_block, patching_config: SquarePatchingConfig):
    return get_4way_seam_patched(
        True, False, True, False,
        ref_block, patch_block, patching_config
    )


# endregion


def get_seam_mask_from_patch_weights(patch_weights: np.ndarray, blends_into_patch: bool) -> np.ndarray:
    """
    Seam Mask here refers to mask containing the boundary or blending area highlighted;
    it is not the same the mask used to merge the source with the patch.
    """
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


def compute_adaptive_blend_mask(source: np.ndarray, patch: np.ndarray, cut_mask: np.ndarray,
                                patching_config: SquarePatchingConfig) -> None:
    """
    Compute adaptive blend mask based on gradient differences.
    The output is copied to cut_mask to avoid additional allocations.

    :param source: existing texture block where the patch will be placed, where the overlap area is [:, :overlap].
                    (rotate/flip the block in order to process other orientations)
    :param patch:  the patch to be placed over the source, where the overlap area is [:, :overlap].
                    (rotate/flip the block in order to process other orientations)
    :param cut_mask: the mask used to produce the final patched block.
    :param patching_config: generation parameters, which should contain the blend_config.
    """
    block_size = patching_config.block_size  # = source.shape[0]
    overlap = patching_config.overlap
    sobel_ksize = patching_config.blend_config.sobel_kernel_size

    # Setup output dimensions
    dims = np.copy(source.shape)
    dims[0] = block_size
    dims[1] = block_size * 2 - overlap

    new_img = np.zeros(dims, dtype=source.dtype)
    new_img[0:block_size, 0:block_size] = source

    # Extract overlap regions (views, not copies)
    cut_mask_overlap = cut_mask[:block_size, :overlap]
    source_overlap = source[:block_size, :overlap]
    patch_overlap = patch[:block_size, :overlap]
    patched_overlap = blend_with_mask(patch_overlap, source_overlap, cut_mask_overlap)

    # Compute Gradients Difference & Create Adaptive Blend Mask for the overlap section
    tdiff_map = gradients_differences_at_the_seam(sobel_ksize, cut_mask_overlap,
                                                  source_overlap, patch_overlap, patched_overlap,
                                                  patching_config.blend_config.grad_diff_func,
                                                  _tmp=patched_overlap)
    blended = create_adaptive_blend_mask(
        tdiff_map=tdiff_map,
        mc_mask_overlap=cut_mask_overlap,
        blend_config=patching_config.blend_config,
        radii_limiter=_get_radii_limiter(block_size, overlap) if patching_config.blend_config.use_blur_radii_limiter else None,
        dtype=source.dtype
    )

    # Update min-cut mask with adaptive blend
    cut_mask[:block_size, :overlap] = blended
