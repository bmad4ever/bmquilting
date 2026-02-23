import cv2
import numpy as np
import math

from numpy import random, ndarray
from dataclasses import dataclass
from functools import lru_cache

try:
    from .synthesis_subroutines import (
        apply_mask, avg_squared_diff, adjust_errors_func_inplace, adjust_errors_for_pystar2d_inplace,
        _filter_candidate_patches, _select_a_random_patch, blend_with_mask,
        get_seam_mask_from_patch_weights, clear_seam_overlapped_by_patch, update_seams_map_view)
    from .types import NumPixels, Percentage, BlendConfig
    from .seam_smartblur import gradients_differences_at_the_seam, create_adaptive_blend_mask, auto_blend_config_2
    from .misc.shmem_utils import SharedTextureList
except:
    from bmquilting.synthesis_subroutines import (
        apply_mask, avg_squared_diff, adjust_errors_func_inplace, adjust_errors_for_pystar2d_inplace,
        _filter_candidate_patches, _select_a_random_patch, blend_with_mask,
        get_seam_mask_from_patch_weights, clear_seam_overlapped_by_patch, update_seams_map_view)
    from bmquilting.types import NumPixels, Percentage, BlendConfig
    from bmquilting.seam_smartblur import gradients_differences_at_the_seam, create_adaptive_blend_mask, auto_blend_config_2
    from bmquilting.misc.shmem_utils import SharedTextureList

TEMP_BLEND_CONFIG = auto_blend_config_2(9, 40,  True)
# quilting using Circular Patches over a Hexagonal Lattice (CPHL)


# TODO
# -> test: test w/ additional textures
# -> cleanup: remove comments & unused
# -> expose patch "hole" function
# -> implement comfyui node
# -> implement "radial" parallel texture generation using this quilting variant

#region __________ TESTING PURPOSES __________

kernelx3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelx5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernelx15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, img)


def create_mock_mask_with_polygon(width, height, polygon_points):
    """
    Create a white mask of given dimensions and draw a black polygon on it.

    Args:
    - width (int): The width of the mask.
    - height (int): The height of the mask.
    - polygon_points (list of tuples): List of (x, y) tuples representing the vertices of the polygon.

    Returns:
    - mask (numpy.ndarray): The mask with the black polygon drawn on it.
    """
    mask = np.ones((height, width), dtype=np.uint8)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_points], (0, 0, 0))
    return mask


def create_mock_mask(shape):
    # Define dimensions of the mask
    height, width = shape[:2]  #500, 500

    # Define points of the 5-sided polygon (pentagon)
    polygon_points = [
        (1 / 2, 1 / 5),  # Top
        (4 / 5, 2 / 5),  # Bottom-right
        (35 / 50, 35 / 50),  # Bottom
        (15 / 50, 35 / 50),  # Bottom-left
        (1 / 5, 2 / 5)  # Top-left
    ]
    polygon_points = [(x * width, y * height) for (x, y) in polygon_points]

    # Create the mask
    mask = create_mock_mask_with_polygon(width, height, polygon_points)
    return mask


def get_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Get the bounding box of the black shape in the mask.
    :param mask: The mask with the black shape.
    :return: bounding_box (tuple): The bounding box of the black shape in the format (x, y, w, h).
    """
    mask = 1 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    else:
        return 0, 0, 0, 0  # edge case: no contours




#endregion __________ TESTING PURPOSES __________


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
    spacing_factor: float

    no_seams: bool
    """computes no seams, use this with vignette directly blend the patch in"""

    def __post_init__(self):
        if 0.0 > self.tolerance:
            raise ValueError(f"{self.tolerance = } tolerance should be greater than 0.")

    @property
    def blend_into_patch(self) -> bool:
        return self.blend_config is not None


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

    showInMovedWindow('Corners', bool_mask.astype(np.uint8)*255, 50 + bool_mask.shape[0] , 25)
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

    showInMovedWindow('Weighted Mask', dst_norm, 50 + distance_map.shape[0] * 2, 25)
    return dst_norm

# endregion Weight Template Matching Auxiliary Funcs ____END


def blur_annulus_mask(mask: np.ndarray, inner_radius: NumPixels) -> np.ndarray:
    """
    UNUSED, likely not needed...
    TODO is this needed / salvageable ?
    """
    height, width = mask.shape

    border_size = round((width - inner_radius) / 2)
    extended_mask = cv2.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)

    blur_kernel_size_h = (border_size // 2, border_size // 2)
    blur_kernel_size = (border_size, border_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, blur_kernel_size_h)
    extended_mask = cv2.morphologyEx(extended_mask, cv2.MORPH_ERODE, kernel)
    blurred_extended_mask = cv2.GaussianBlur(extended_mask, blur_kernel_size, sigmaX=0, sigmaY=0)

    blurred_mask = blurred_extended_mask[border_size:border_size + height, border_size:border_size + width]

    blurred_mask = cv2.normalize(blurred_mask.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    return blurred_mask


def reverse_blured_circle_mask(mask: np.ndarray, overlap_radius):
    """
    :param mask: circle mask as uint8 w/ values from [0 to 255]
    TODO is this needed / salvageable?
    """
    _, width = mask.shape[:2]
    ksize = width // 2 + 1
    ksize = (ksize, ksize)
    border_size = round(width / 2)
    blured_mask = cv2.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
    blured_mask[border_size:-border_size, border_size:-border_size] = mask
    np.subtract(255, blured_mask, out=blured_mask)
    blured_mask = cv2.morphologyEx(blured_mask, cv2.MORPH_DILATE, kernelx3, iterations=overlap_radius // 2)
    blured_mask = cv2.GaussianBlur(blured_mask, ksize, sigmaX=0, sigmaY=0)
    blured_mask = blured_mask[border_size:-border_size, border_size:-border_size]
    blured_mask = blured_mask.astype(np.float32) / 255
    return blured_mask


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


def process_patch_at_location(image: np.ndarray, filled_mask: np.ndarray, seams_map: np.ndarray,
                              lookup_textures: list[np.ndarray] | SharedTextureList,
                              x: int, y: int,
                              config: CircularPatchingConfig,
                              rng: np.random.Generator) -> None:
    """
    Finds and applies a single circular patch at location (x, y).

    :param image: Image being patched or generated.
        Should be of type float, with values in the interval [0, 1].
    :param filled_mask: Mask that keeps track of the area already filled.
        Should be of type float, with values in the interval [0, 1].

    Updates image block and the blending mask at the provided (x, y) position.
    """
    # --- Fetch Config Parameters ---
    radius = config.patch_params.radius
    pp = config.patch_params
    bbox_idx = np.s_[y - radius:y + radius + 1, x - radius:x + radius + 1]  # bbox for the area being patched

    # Extract the current image block and the region of interest (ROI)
    block = image[bbox_idx]
    roi = filled_mask[bbox_idx]
    seams = seams_map[bbox_idx]

    # Get the annular mask used for this patch
    annulus_roi_block = _get_annular_mask(config.patch_params)

    roi = annulus_roi_block * roi  # confines the ROI to the annulus (overlap) region
    tmpl_mask = roi                # mask used w/ template matching

    # (optional) Weighted Template Matching Setup
    if config.outer_corners_weighted_template_matching:
        corners_mask = _find_annulus_outer_corners(roi, pp.non_overlap_radius, radius)
        if corners_mask is not None:
            tmpl_mask = _distance_to_points(corners_mask)
            tmpl_mask *= roi
        del corners_mask

    # Find the best matching patch from the lookup texture
    patch: np.ndarray = _find_circular_patch(lookup_textures, block, tmpl_mask, config, rng)

    # AUX may be used for the patched block (w/ raw seam, no blending) and for the vignette
    #   no seams + no vignette is not an expected option combination; it only makes sense for debug purposes
    #   so the is no need prevent the allocation with a conditional here
    aux = np.empty_like(patch)

    # Get mask to blend source and patch (pre vignette)
    mask = _get_circle_mask(config.patch_params).copy() if config.no_seams else \
           _compute_radial_seam_mask(config, roi, block, patch, _tmp=aux)

    # (optional) Apply Vignette
    if config.blend_into_patch and config.blend_config.use_vignette:
        vignette = _setup_vignette(roi, config.patch_params, dst=aux.reshape(aux.shape[0], -1))
        np.minimum(mask, vignette, out=mask)

        showInMovedWindow("vignette", vignette, 50 + radius * 2 * 5, 25)

    showInMovedWindow("mask", mask, 50 + radius * 2 * 4, 25)

    # Update Seams & Filled Mask State
    update_seams_map_view(seams, mask, config.blend_into_patch)
    np.maximum(mask, filled_mask[bbox_idx], out=filled_mask[bbox_idx])

    # Paste into the Source Image
    np.subtract(1, mask, out=mask)  # mask <- (1 - mask) so that image[bbox_idx] is sent as fg
    blend_with_mask(patch, image[bbox_idx], mask, out=image[bbox_idx])

    # TODO REMOVE DEBUG LATER
    showInMovedWindow("patched single", image, 1920 - image.shape[1], 25)
    cv2.waitKey(0)

    # TODO REMOVE DEBUG LATER
    cv2.imshow("filled state", filled_mask)
    cv2.waitKey(0)


def _compute_radial_seam_mask(circ_patching_config: CircularPatchingConfig,
                              roi: ndarray, block: ndarray, patch: ndarray, _tmp: ndarray) -> np.ndarray:
    pp = circ_patching_config.patch_params
    f_radius = float(pp.radius)

    # Compute Errors prior to warping
    errors = avg_squared_diff(block, patch)
    adjust_errors_func_inplace(errors)

    # Find Seam using Polar Coordinates
    center = circ_patching_config.patch_params.center_2d_f
    warped_len = circ_patching_config.patch_params.warped_len
    warp_flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST

    # Warp errors and overlap ROI to polar coordinates & Find "optimal" seam
    polar_errors, polar_roi = (
        cv2.warpPolar(i, (pp.radius, warped_len), center, f_radius, warp_flags)
        for i in [errors, roi])
    mask = min_cut_circ(polar_errors, polar_roi, pp.non_overlap_radius)

    # Warp mask back to cartesian coords & Compute patched result
    mask = cv2.warpPolar(mask, roi.shape[:2], center, f_radius, cv2.WARP_INVERSE_MAP | warp_flags)
    patched = blend_with_mask(block, patch, mask, out=_tmp)
    # mind that even if blend into patch option is not set, patched variable is re-used for the vignette later

    # (optional) Blend mask with respect to gradients diff.
    if circ_patching_config.blend_into_patch:
        tdiff_map = gradients_differences_at_the_seam(circ_patching_config.blend_config.sobel_kernel_size, mask, block, patch,
                                                      patched, _tmp=patched)

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

        showInMovedWindow("radii limiter", radii_limiter / max(np.max(radii_limiter), 1e-5), 50 + pp.radius * 2 * 4, 25)
        mask = create_adaptive_blend_mask(
            tdiff_map=tdiff_map,
            mc_mask_overlap=mask,
            blend_config=circ_patching_config.blend_config,
            radii_limiter_mask=radii_limiter,
            dtype=block.dtype
        )
    return mask


def circle_quilt(image: np.ndarray, roi_mask: np.ndarray,
                 lookup_textures: list[np.ndarray] | SharedTextureList,
                 config: CircularPatchingConfig,
                 rng: np.random.Generator,
                 debug_img: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies texture synthesis to the ROI by tiling circular patches on a hexagonal lattice
    and blending them using min-cut in polar coordinates, driven by a configuration object.
    """

    # --- 0. Fetch Config Parameters ---
    radius = config.patch_params.radius
    diameter = config.patch_params.diameter
    spacing_factor = config.spacing_factor

    # --- 1. Initial Setup and Padding ---

    # Pad the image and mask
    # NOTE: Since the mask is slightly eroded, diameter is used instead of the radius,
    #       so that there is enough leeway in worst case scenario.
    image = cv2.copyMakeBorder(image, diameter, diameter, diameter, diameter, borderType=cv2.BORDER_REPLICATE)
    roi_mask = cv2.copyMakeBorder(roi_mask, diameter, diameter, diameter, diameter, borderType=cv2.BORDER_REPLICATE)
    roi_mask_i = roi_mask.astype(np.uint8)
    roi_mask = roi_mask.astype(np.float32)
    roi_mask /= np.max(roi_mask)

    seams_map = np.zeros(image.shape[:2], dtype=np.float32)

    # Get the bounding box of the ROI (in the padded image coordinates)
    min_x, min_y, w, h = get_bounding_box(roi_mask_i)
    max_x, max_y = min_x + w, min_y + h
    del roi_mask_i

    # filled_mask tracks which areas have already been patched (used for overlap region)
    filled_mask = roi_mask.copy()

    # Erode the ROI mask to prevent against cases where there is little nearby texture to grab for template matching
    kernel_size = 2 * radius//2 + 1
    circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cv2.morphologyEx(roi_mask, cv2.MORPH_ERODE, kernel=circular_kernel, dst=roi_mask)
    del kernel_size, circular_kernel
    cv2.imshow("eroded roi_mask", roi_mask * 255)

    # --- 2. Hexagonal Lattice Iteration ---
    # Calculates the spacing for the patch centers based on config.spacing_factor
    x_spacing = int(radius * 2 / 1.5 * math.cos(math.pi / 6) * spacing_factor)
    y_spacing = int(radius * spacing_factor)

    for y in range(int(min_y), int(max_y + y_spacing), y_spacing):
        for x in range(int(min_x), int(max_x + x_spacing), x_spacing):

            if (y // y_spacing) % 2 != 0:  # offset every odd row to create a hexagonal pattern
                x += x_spacing // 2

            if roi_mask[y, x] > 0:         # skip if the patch center is outside the eroded ROI
                continue

            if debug_img is not None:      # debug lattice, TODO remove later
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.circle(debug_img, (int(x - diameter), int(y - diameter)), radius, color, -1)
                cv2.circle(debug_img, (int(x - diameter), int(y - diameter)), 5, (0, 0, 0), -1)

            process_patch_at_location(image, filled_mask, seams_map, lookup_textures, x, y, config, rng)

    # --- 3. Final Cleanup ---

    # Debug visualization for the bounding box
    if debug_img is not None:           # TODO remove debug
        cv2.rectangle(debug_img, (int(min_x - diameter), int(min_y - diameter)), (int(max_x - diameter), int(max_y - diameter)),
                      (255, 255, 255), 2)

    # Remove the padding from the final image before returning
    return image[diameter:-diameter, diameter:-diameter], seams_map


def _find_first_2adjacent_all_zero_rows(mask:np.ndarray) -> int | None:
    """:return: the index of the second all zero row, or None if none found"""
    prev = False
    for i, row in enumerate(mask):
        current = all(pixel == 0 for pixel in row)
        if prev and current:
            return i
        prev = current
    return None


def min_cut_circ(errors: np.ndarray, roi: np.ndarray, non_overlap_radius: NumPixels) -> np.ndarray:
    """
    @param errors: warped matrix with the pre-computed errors
    @param roi: warped binary mask of the region of interest
    """
    import pyastar2d

    # devnote:
    #  narrow cuts, that barely have any width, can still allow for narrow paths;
    #   such cases can leave unfilled "cuts" in the texture when patching.
    #  To avoid the aforementioned problem, the size or spacing of the patches can be changed.
    #  I've attempted to solve the problem using morphological operations;
    #   however, this approach comes with some new problems.
    #  I've thus decided to keep the solution simple.

    adjust_errors_for_pystar2d_inplace(errors, errors.shape[0])

    offset_row = _find_first_2adjacent_all_zero_rows(roi[:, non_overlap_radius:])
    if offset_row is not None:
        errors = np.roll(errors, -offset_row, axis=0)
        roi = np.roll(roi, -offset_row, axis=0)

    showInMovedWindow("rolled polar roi", roi * 255, 100, 200)

    errors[roi == 0] = 0  # TODO only for DEBUG purposes, can be removed later
    showInMovedWindow("errors", errors / max(np.max(errors), 1e-5), 100 + (8 + roi.shape[1]) * 2, 200)

    start_x = end_x = errors.shape[1] - 1
    if offset_row is None:
        # when no empty row is found, search for the cut endpoints at the top & bottom so that the path is not broken
        start_x, end_x = _find_min_cut_circ_endpoints(errors, roi)

    errors[:, -1] = roi[:, -1] * errors[:, -1] + (1 - roi[:, -1])  # bottom holes escape path
    errors[:, :-1][roi[:, :-1] == 0] = np.inf  # don't travel outside the mask, unless via the escape path

    start = (0, start_x)
    end = (errors.shape[0] - 1, end_x)
    #cv2.waitKey(0)

    path = pyastar2d.astar_path(errors, start, end, allow_diagonal=True)
    mask = np.zeros_like(roi)

    print(f"mask.shape={mask.shape}")
    for i, j in path:  # draw path
        mask[i, j] = 1

    showInMovedWindow("cut path", mask * 255, 100 + (8 + roi.shape[1]) * 3, 200)

    print(f"seed point={(mask.shape[0] - 1, mask.shape[1] - 1)}")
    cv2.floodFill(mask, None, (0, 0), (1,))
    if offset_row is not None:
        mask = np.roll(mask, offset_row, axis=0)
    showInMovedWindow("cut mask", mask * 255, 100 + (8 + roi.shape[1]) * 4, 200)
    #cv2.waitKey(0)

    return mask


def _x_squared_distance_1d(a_1d: np.ndarray, b_1d: np.ndarray) -> np.ndarray:
    """Computes the squared distance matrix based on 1D arrays of X-coordinates."""
    # Reshape for broadcasting: (N, 1) and (1, M)
    a_coords = a_1d.reshape(-1, 1)
    b_coords = b_1d.reshape(1, -1)
    squared_dist_matrix = (a_coords - b_coords) ** 2  # gets rid of the sign
    return squared_dist_matrix


def _find_min_cut_circ_endpoints(errors: np.ndarray, roi: np.ndarray) -> tuple[int, int]:
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
    showInMovedWindow("rolled roi section", roi * 255, 100 + (8 + roi.shape[1]) * 5, 200)

    maze = np.ones((errors.shape[0] + 2, errors.shape[1]), dtype=np.float32)
    maze_area = maze[1:-1, :]
    maze_area[:, :] = errors

    maze[0, :] = 1
    maze[-1, :] = 1
    maze[1:-1, -1] = roi[:, -1] * maze[1:-1, -1] + (1 - roi[:, -1])  # bottom holes escape path
    maze[1:-1, :-1][roi[:, :-1] == 0] = np.inf  # don't travel outside the mask, unless via the escape path

    start = (0, maze.shape[1] - 1)
    end = (maze.shape[0] - 1, maze.shape[1] - 1)

    path = pyastar2d.astar_path(maze, start, end, allow_diagonal=True)

    # Compute the distance matrix between points (y is ignored since it is fixed)
    err_len_div4_minus1 = err_len_div4 - 1
    points_of_interest_top = [x for (y, x) in path if y - 1 == err_len_div4]
    points_of_interest_bottom = [x for (y, x) in path if y - 1 == err_len_div4_minus1]
    dist_matrix = _x_squared_distance_1d(np.array(points_of_interest_top), np.array(points_of_interest_bottom))

    # Find the indices of the minimum distance
    min_index = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    top_point = points_of_interest_top[min_index[0]]
    bottom_point = points_of_interest_bottom[min_index[1]]
    print(f"points of interest = {(top_point, bottom_point)}")

    # TODO debug code, not actually needed, remove later
    mask = np.zeros_like(maze)
    for i, j in path:
        mask[i, j] = 1
        if i - 1 == err_len_div4:
            mask[i, :max(0, j-4):4] = 1

    showInMovedWindow("aux_cut", mask, 100 + (8 + roi.shape[1]) * 6, 200)

    return top_point, bottom_point


def _find_circular_patch(lookup_textures: list[np.ndarray] | SharedTextureList,
                         block: np.ndarray, mask: np.ndarray,
                         params: CircularPatchingConfig, rng: np.random.Generator) -> np.ndarray:
    """

    :param lookup_textures: textures used for patch searching.
    :param block: 2D slice of the texture being patched containing the overlapping region.
        The array should be of type float.
    :param mask: Mask of the overlapping region. It should be of type float, having values in the interval [0, 1].
        The mask can be weighted to prioritize or neglect certain spots within the overlapping region.
    :param rng: random generator for results reproducibility.
    :return: block sized patch containing the overlapping region.
    """
    block_size, tolerance = params.patch_params.block_size, params.tolerance
    err_mats: list[np.ndarray | None] = []
    global_min_error = np.inf

    showInMovedWindow("roi used in match template", mask, 50 + block_size * 3, 25)

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

        _min, _, _, _ = cv2.minMaxLoc(err_mat)
        global_min_error = min(global_min_error, _min)  # update minimum

    # --- Check for impossible match ---
    if global_min_error == np.inf:
        raise ValueError("Could not find a suitable patch in any lookup texture "
                         "(all textures were too small or had matching issues).")

    # --- PASS 2: Collect all candidates within the final tolerance window ---
    final_candidates = _filter_candidate_patches(err_mats, global_min_error, tolerance)
    best_texture_idx, best_x, best_y = _select_a_random_patch(final_candidates, rng)
    return lookup_textures[best_texture_idx][best_y:best_y+block_size, best_x:best_x+block_size]


def _setup_vignette_radial(
        roi: np.ndarray, patch_params: CircularPatchParams, dst:np.ndarray=None,
        _tmp:np.ndarray=None) -> np.ndarray:
    """DEPRECATED"""
    center = (patch_params.center, patch_params.center)
    dilate_kernel_size = max(3, patch_params.radius//8 + 1)
    blur_kernel_size = max(3, patch_params.radius//4 + 1)

    radial = (roi>0).astype(np.uint8)  # copy & set dtype for dist. transf
    _radial_extrapolate(radial, patch_params.radius - 2, radial)

    # Patch the transition going from the overlapping area to the non overlapping area
    patch_edge = cv2.distanceTransform(radial, cv2.DIST_L2, cv2.DIST_MASK_3, dst=_tmp, dstType=cv2.CV_32F)
    patch_edge[patch_edge > patch_params.overlap_radius] = patch_params.overlap_radius
    np.subtract(patch_params.overlap_radius, patch_edge, out=patch_edge)
    cv2.normalize(patch_edge, patch_edge, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Create the vignette fading into the prior, already placed, texture
    np.subtract(1, radial, out=radial)
    cv2.circle(radial, center, patch_params.radius, (1,), -1)
    vignette = cv2.distanceTransform(radial, cv2.DIST_L2, cv2.DIST_MASK_3, dst=dst, dstType=cv2.CV_32F)
    vignette[vignette > patch_params.overlap_radius] = patch_params.overlap_radius
    cv2.normalize(vignette, vignette, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    cv2.blur(vignette, (blur_kernel_size, blur_kernel_size), dst=vignette)

    np.maximum(vignette, patch_edge, out=vignette)

    cv2.dilate(vignette, np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.float32), dst=vignette)
    np.clip(vignette, 0, 1, out=vignette)
    vignette **= 1/5
    return vignette


def _setup_vignette_(roi: np.ndarray, patch_params: CircularPatchParams, dst:np.ndarray=None,
                    _tmp:np.ndarray=None) -> np.ndarray:
    center = (patch_params.center, patch_params.center)
    dilate_kernel_size = max(3, patch_params.radius//8 + 1)
    blur_kernel_size = max(3, patch_params.radius//4 + 1)

    # patch transition into non-overlapping section
    aux = (roi>0).astype(np.uint8)  # copy & set dtype for dist. transf
    np.subtract(1, aux, out=aux)
    aux *= _get_annular_mask(patch_params).astype(np.uint8)
    np.subtract(1, aux, out=aux)
    cv2.circle(aux, center, patch_params.non_overlap_radius, (1,), -1)
    cv2.imshow("CPT", aux*255)

    patch_edge = cv2.distanceTransform(aux, cv2.DIST_L2, cv2.DIST_MASK_3, dst=_tmp, dstType=cv2.CV_32F)
    patch_edge[patch_edge > patch_params.overlap_radius//2] = patch_params.overlap_radius//2
    np.subtract(patch_params.overlap_radius, patch_edge, out=patch_edge)
    cv2.normalize(patch_edge, patch_edge, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    cv2.imshow("PE", patch_edge)

    # Create the vignette fading into the prior, already placed, texture
    #np.subtract(1, radial, out=radial)
    bin_roi = np.zeros_like(aux, dtype=np.uint8)
    cv2.circle(bin_roi, center, patch_params.radius, (1,), -1)
    vignette = cv2.distanceTransform(bin_roi, cv2.DIST_L2, cv2.DIST_MASK_3, dst=dst, dstType=cv2.CV_32F)
    vignette[vignette > patch_params.overlap_radius] = patch_params.overlap_radius
    cv2.normalize(vignette, vignette, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    cv2.imshow("grad", vignette)
    cv2.blur(vignette, (blur_kernel_size, blur_kernel_size), dst=vignette)

    np.maximum(vignette, patch_edge, out=vignette)

    cv2.dilate(vignette, np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.float32), dst=vignette)
    np.clip(vignette, 0, 1, out=vignette)

    #vignette **= 1/2
    vignette **= 2

    cv2.imshow("VIG", vignette)
    cv2.waitKey(0)
    return vignette


def _setup_vignette_2(roi: np.ndarray, patch_params: CircularPatchParams, dst: np.ndarray = None,
                    _tmp: np.ndarray = None) -> np.ndarray:
    center = (patch_params.center, patch_params.center)
    # Kernels
    blur_k = max(3, patch_params.radius // 4 + 1)

    # 1. Complement to the overlapping region
    aux = (roi>0).astype(np.uint8)  # copy & set dtype for dist. transf
    aux = cv2.bitwise_xor(aux, _get_annular_mask(patch_params).astype(np.uint8), dst=aux)
    # Invert back to get the 'background' for distanceTransform
    cv2.bitwise_xor(aux, 1, dst=aux)
    cv2.imshow("CPT", aux*255)
    cv2.circle(aux, center, patch_params.non_overlap_radius, (1,), -1)


    # 2. Optimized patch_edge calculation
    patch_edge = cv2.distanceTransform(aux, cv2.DIST_L2, 3, dst=_tmp, dstType=cv2.CV_32F)
    limit = patch_params.overlap_radius // 2
    np.clip(patch_edge, 0, limit, out=patch_edge)
    np.subtract(limit, patch_edge, out=patch_edge)
    cv2.multiply(patch_edge, 1.0 / limit, dst=patch_edge)

    # 3. Create the vignette
    # Re-use aux to save allocation if possible, or use a zeros_like
    bin_roi = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(bin_roi, center, patch_params.radius, (1,), -1)

    vignette = cv2.distanceTransform(bin_roi, cv2.DIST_L2, 3, dst=dst, dstType=cv2.CV_32F)
    v_limit = float(patch_params.overlap_radius)
    np.clip(vignette, 0, v_limit, out=vignette)
    np.multiply(vignette, 1.0 / v_limit, out=vignette)

    cv2.blur(vignette, (blur_k, blur_k), dst=vignette)

    # 4. Final Composition
    np.maximum(vignette, patch_edge, out=vignette)

    # Use a simple integer for anchor if possible, or pre-calculate kernel
    dilate_sz = max(3, patch_params.radius // 8 + 1)
    kernel = np.ones((dilate_sz, dilate_sz), dtype=np.uint8)
    cv2.dilate(vignette, kernel, dst=vignette)

    np.clip(vignette, 0, 1, out=vignette)
    cv2.imshow("VIG", aux * 255)

    return vignette


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



@lru_cache(maxsize=2)
def _get_radial_map(h: NumPixels, w: NumPixels, radius: NumPixels) -> tuple[np.ndarray, np.ndarray]:
    """:return: (pixel coordinates: CV_16SC2, interpolation coefficients: CV_16UC1)"""
    cx, cy = w // 2, h // 2

    # Allocate one contiguous block of memory so that coords & coeffs are adjacent in memory
    total_elements = h * w * 3
    raw_data = np.empty(total_elements, dtype=np.uint16)

    # Create views
    coords = raw_data[:h * w * 2].view(np.int16).reshape(h, w, 2)
    coeffs = raw_data[h * w * 2:].reshape(h, w)

    # Compute map in float32
    y, x = np.indices((h, w))
    dx, dy = x - cx, y - cy
    theta = np.arctan2(dy, dx)

    tmp_float_map = np.empty((h, w, 2), dtype=np.float32)
    tmp_float_map[..., 0] = cx + radius * np.cos(theta)
    tmp_float_map[..., 1] = cy + radius * np.sin(theta)

    # Convert to integer coordinates + coefficients
    # noinspection PyTypeChecker
    cv2.convertMaps(tmp_float_map, None, cv2.CV_16SC2,
                    dstmap1=coords, dstmap2=coeffs)

    return coords, coeffs



def _radial_extrapolate(mask: np.ndarray, radius: NumPixels, dst:np.ndarray=None) -> np.ndarray:
    coords, coeffs = _get_radial_map(mask.shape[0], mask.shape[1], radius)
    return cv2.remap(mask, coords, coeffs, cv2.INTER_NEAREST, dst, cv2.BORDER_REPLICATE)



def set_random_patch_at_location(image: np.ndarray, filled_mask: np.ndarray,
                                 lookup_textures: list[np.ndarray] | SharedTextureList,
                                 x: int, y: int,
                                 config: CircularPatchingConfig,
                                 rng: np.random.Generator):
    """
        Note: despite not updating seams, this function is prepared to handle non-empty image sources so
        that it can be repurposed for other use cases besides selecting the 1st patch when generating a texture.
    """
    radius = config.patch_params.radius
    center = config.patch_params.center
    block_size = config.patch_params.block_size
    y1, y2, x1, x2 = y - radius, y + radius + 1, x - radius, x + radius + 1

    # Random patch selection
    rnd_lookup = lookup_textures[rng.integers(len(lookup_textures))]
    h, w = rnd_lookup.shape[:2]
    rand_h = rng.integers(h - block_size)
    rand_w = rng.integers(w - block_size)
    start_block = rnd_lookup[rand_h:rand_h + block_size, rand_w:rand_w + block_size]

    # Create circular mask
    mask = np.ones((block_size, block_size), dtype=filled_mask.dtype)
    cv2.circle(mask, (center, center), radius, (0,), -1)

    # Update image & Filled mask
    apply_mask(image[y1:y2, x1:x2], mask, True)
    np.subtract(1, mask, out=mask)
    np.maximum(mask, filled_mask[y1:y2, x1:x2], out=filled_mask[y1:y2, x1:x2])
    image[y1:y2, x1:x2] += apply_mask(start_block, mask)



if __name__ == "__main__":
    text_idx = "18"
    src = cv2.imread(f"test_data/results/t{text_idx}.png", cv2.IMREAD_COLOR)
    src2 = cv2.imread(f"test_data/textures/t{text_idx}.png", cv2.IMREAD_COLOR)
    src = np.float32(src) / 255
    src2 = np.float32(src2) / 255

    img = cv2.resize(src, [d * 2 for d in src.shape[:2][::-1]])
    lookup = cv2.resize(src2, [d * 2 for d in src2.shape[:2][::-1]])

    mask = create_mock_mask(img.shape)
    print(f"shapes {(img.shape, mask.shape)}")
    img[mask == 0] = (0, 0, 0)

    #TEMP_BLEND_CONFIG.use_vignette = False
    config = CircularPatchingConfig(
        patch_params=CircularPatchParams(diameter=161, overlap_ratio=.7),
        spacing_factor=.68,
        tolerance=0.0,
        outer_corners_weighted_template_matching=True,
        blend_config=TEMP_BLEND_CONFIG,
        no_seams=True
    )

    rng = np.random.default_rng(seed=0)
    debug_img = np.stack((mask.copy(),) * 3, axis=-1)
    img, seams = circle_quilt(image=img, roi_mask=mask, lookup_textures=[lookup], debug_img=debug_img,
                       config=config, rng=rng)
    cv2.imshow("Result", img)
    cv2.imshow("Seams", seams)
    cv2.imshow("Patches Visualizer", debug_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
