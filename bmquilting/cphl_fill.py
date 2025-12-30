import cachetools
import cv2
import numpy as np
import math
from numpy import random
from dataclasses import dataclass
from functools import cached_property, lru_cache

try:
    from .synthesis_subroutines import (
        apply_mask, avg_squared_diff, adjust_errors_func_inplace, adjust_errors_for_pystar2d_inplace)
    from .types import NumPixels, Percentage, BlendConfig
    from .seam_smartblur import gradients_differences_at_the_seam, create_adaptive_blend_mask, auto_blend_config_2
except:
    from bmquilting.synthesis_subroutines import (
        apply_mask, avg_squared_diff, adjust_errors_func_inplace, adjust_errors_for_pystar2d_inplace)
    from bmquilting.types import NumPixels, Percentage, BlendConfig
    from bmquilting.seam_smartblur import gradients_differences_at_the_seam, create_adaptive_blend_mask, auto_blend_config_2


TEMP_BLEND_CONFIG = auto_blend_config_2(9, 40, 1, True)
# quilting using Circular Patches over a Hexagonal Lattice (CPHL)


# TODO
# -> test: test w/ additional textures
# -> cleanup: remove comments & unused
# -> tolerance must be an argument
# -> blend into patch feature
# -> expose patch "hole" function
# -> implement comfyui node
# -> implement "radial" parallel texture generation using this quilting variant

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
    tolerance: Percentage
    outer_corners_weighted_template_matching: bool
    spacing_factor: float

    def __post_init__(self):
        if 0.0 > self.tolerance:
            raise ValueError(f"{self.tolerance = } tolerance should be greater than 0.")



def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, img)


def find_annulus_outer_corners(mask: np.ndarray, inner_radius: int, outer_radius: int) -> list[tuple[int, int]]:
    """
    Args:
        mask: the mask w/ the annulus
        inner_radius: inner radius of the annulus
        outer_radius: outer radius of the annulus

    Returns:
        list of points in the following tuple format: (y, x)
    """
    margin: int = 4
    extended_mask = np.zeros((margin * 2 + mask.shape[0], margin * 2 + mask.shape[1]), dtype=mask.dtype)
    extended_mask[4:-4, 4:-4] = mask

    dst = cv2.cornerHarris(extended_mask, blockSize=2, ksize=5, k=0.06)
    corners = np.nonzero(dst > int(0.01 * dst.max()))
    corners = zip(corners[1], corners[0])  # generator to iterate all corners; also swaps y & x
    corners = ((c[0] - margin, c[1] - margin) for c in corners)  # offset w/ margin

    # ___ filter out inner corners __________
    center = (outer_radius + margin, outer_radius + margin)
    sqr_cut_off_radius = ((outer_radius + inner_radius) / 2) ** 2

    def corner_filter_predicate(corner):
        nonlocal center, sqr_cut_off_radius
        return sqr_cut_off_radius < (corner[0] - center[0]) ** 2 + (corner[1] - center[1]) ** 2

    outer_corners = [c for c in corners if corner_filter_predicate(c)]
    return outer_corners


def distance_to_points(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    """
    Args:
        shape: output mask shape
        points: list of points in the format: (y, x).

    Returns:
        a float32 mask with the distance transform
    """
    corners = np.zeros(shape, dtype=np.uint8)
    for p in points:  # paint points
        # points are drawn using cv.circle to avoid out of bound errors at the edges
        cv2.circle(corners, p, 3, (255,), -1)

    showInMovedWindow("annulus corners", corners, 50 + shape[0] * 1, 25)

    distance_map = cv2.distanceTransform(255 - corners, cv2.DIST_L2, 5, dst=corners)
    cv2.GaussianBlur(distance_map, (15, 15), sigmaX=0, sigmaY=0,
                     dst=distance_map)  # TODO this requires a min radius of ~8
    dst_norm = np.empty_like(distance_map, dtype=np.float32)
    dst_norm = cv2.normalize(distance_map, dst_norm, 0, 1, cv2.NORM_MINMAX)
    dst_norm = 1 - dst_norm

    showInMovedWindow('Weighted Mask', dst_norm, 50 + shape[0] * 2, 25)
    return dst_norm


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


kernelx3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelx5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernelx15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


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


@lru_cache(maxsize=1)
def _get_annular_mask(patch_params: CircularPatchParams, dtype_name: str= "float32") -> np.ndarray:
    """:return: Binary annular mask for the circular patch overlapping region."""
    block_size, center = patch_params.block_size, patch_params.center
    radius, non_overlap_radius = patch_params.radius, patch_params.non_overlap_radius
    annulus_roi_block = np.zeros((block_size, block_size), dtype=np.dtype(dtype_name))
    cv2.circle(annulus_roi_block, (center, center), radius, (1,), -1)
    cv2.circle(annulus_roi_block, (center, center), round(non_overlap_radius), (0,), -1)
    return annulus_roi_block


# Helper function: Signature updated to accept only config
def _process_patch_at_location(image: np.ndarray, filled_mask: np.ndarray,
                               lookup: np.ndarray,  # TODO replace with list | SharedTextList
                               x: int, y: int,
                               config: CircularPatchingConfig,
                               rng: np.random.Generator, debug_img):
    """
    Finds and applies a single circular patch at location (x, y).

    :param image: Image being patched or generated.
        Should be of type float, with values in the interval [0, 1].
    :param filled_mask: Mask that keeps track of the area already filled.
        Should be of type float, with values in the interval [0, 1].

    Updates image block and the blending mask.
    """
    # --- Fetch Config Parameters ---
    radius = config.patch_params.radius
    f_radius = float(radius)
    non_overlap_radius = config.patch_params.non_overlap_radius

    # 1. Calculate the bounding box for the current circular patch
    y1, y2, x1, x2 = y - radius, y + radius + 1, x - radius, x + radius + 1

    # 2. Extract the current image block and the region of interest (ROI)
    block = image[y1:y2, x1:x2]
    roi = filled_mask[y1:y2, x1:x2]

    # Get the annular mask used for this patch
    annulus_roi_block = _get_annular_mask(config.patch_params, dtype_name=roi.dtype.name)

    roi = annulus_roi_block * roi  # confines the ROI to the annulus (overlap) region
    tmpl_mask = roi                # mask used w/ template matching

    # 3. Weighted Template Matching Setup
    # The 'outer_corners_weighted_template_matching' flag can be used here later
    if config.outer_corners_weighted_template_matching:
        corners = find_annulus_outer_corners(roi, non_overlap_radius, radius)
        if len(corners) > 0:
            tmpl_mask = distance_to_points(roi.shape[:2], corners)
            tmpl_mask *= roi
        del corners

    # 4. Find the best matching patch from the lookup texture
    # Now using the config's tolerance
    patch: np.ndarray = _find_circular_patch(lookup, block, tmpl_mask, config, rng)

    # 5. Compute Errors prior to warping
    errors = avg_squared_diff(block, patch)
    adjust_errors_func_inplace(errors)

    # 6. Find Seam using Polar Coordinates
    center = config.patch_params.center_2d_f
    warped_len = config.patch_params.warped_len
    warp_flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST

    # Warp current block, new patch, and overlap ROI to polar coordinates
    polar_errors, polar_roi = (
        cv2.warpPolar(i, (radius, warped_len), center, f_radius, warp_flags)
        for i in [errors, roi])

    # Find the "optimal" seam
    mask = min_cut_circ(polar_errors, polar_roi, non_overlap_radius)

    # Warp the seam mask back to Cartesian coordinates
    mask = cv2.warpPolar(mask, roi.shape[:2], center, f_radius,
                         cv2.WARP_INVERSE_MAP | warp_flags)

    showInMovedWindow("mask", mask * 255, 50 + radius * 2 * 4, 25)

    # 7. Compute patched result
    patched = apply_mask(image[y1:y2, x1:x2], (1 - mask)) + apply_mask(patch, mask)

    # TODO CURRENT START

    tdiff_map = gradients_differences_at_the_seam(TEMP_BLEND_CONFIG.sobel_kernel_size,  # TODO remove hardcoding
                                                  mask, block, patch, patched)

    blended = create_adaptive_blend_mask(
        tdiff_map=tdiff_map,
        mc_mask_overlap=mask,
        blend_config=TEMP_BLEND_CONFIG, # TODO replace with actual configs
        radii_limiter_mask=roi,
        dtype=block.dtype
    )

    #showInMovedWindow("blended", blended * 255, 50 + radius * 2 * 5, 25)

    # TODO END

    # 8. Paste into the source image
    image[y1:y2, x1:x2] = apply_mask(image[y1:y2, x1:x2], blended) + apply_mask(patch, 1-blended)
    showInMovedWindow("patched single", image, 1920 - image.shape[1], 25)
    cv2.waitKey(0)

    # 9. Update the filled mask state
    np.maximum(mask, filled_mask[y1:y2, x1:x2], out=filled_mask[y1:y2, x1:x2])
    cv2.imshow("filled state", filled_mask * 255)
    cv2.waitKey(0)

    return image[y1:y2, x1:x2], mask


def circle_quilt(image: np.ndarray, roi_mask: np.ndarray,
                 lookup: np.ndarray,  # TODO change to list / SharedTextList
                 config: CircularPatchingConfig,
                 rng: np.random.Generator,
                 debug_img: np.ndarray = None) -> np.ndarray:
    """
    Applies texture synthesis to the ROI by tiling circular patches on a hexagonal lattice
    and blending them using min-cut in polar coordinates, driven by a configuration object.
    """

    # --- 0. Fetch Config Parameters ---
    radius = config.patch_params.radius
    spacing_factor = config.spacing_factor

    # --- 1. Initial Setup and Padding ---

    # Pad the image and mask
    image = cv2.copyMakeBorder(image, radius, radius, radius, radius, borderType=cv2.BORDER_REFLECT_101)
    roi_mask = cv2.copyMakeBorder(roi_mask, radius, radius, radius, radius, borderType=cv2.BORDER_REFLECT_101)
    roi_mask_i = roi_mask.astype(np.uint8)
    roi_mask = roi_mask.astype(np.float32)
    roi_mask /= np.max(roi_mask)

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

    # Debug visualization for the bounding box
    if debug_img is not None:           # TODO remove debug
        cv2.rectangle(debug_img, (int(min_x - radius), int(min_y - radius)), (int(max_x - radius), int(max_y - radius)),
                      (1.0, 0.0, 0.0), 2)

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
                cv2.circle(debug_img, (int(x - radius), int(y - radius)), radius, color, -1)
                cv2.circle(debug_img, (int(x - radius), int(y - radius)), 5, (0, 0, 0), -1)

            _process_patch_at_location(image, filled_mask, lookup, x, y, config, rng, debug_img)

    # --- 3. Final Cleanup ---

    # Remove the padding from the final image before returning
    return image[radius:-radius, radius:-radius]


def find_first_2adjacent_all_zero_rows(mask):
    """return the index of the second all zero row"""
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

    offset_row = find_first_2adjacent_all_zero_rows(roi[:, non_overlap_radius:])
    if offset_row is not None:
        errors = np.roll(errors, -offset_row, axis=0)
        roi = np.roll(roi, -offset_row, axis=0)

    showInMovedWindow("rolled polar roi", roi * 255, 100, 200)

    errors[roi == 0] = 0  # TODO only for DEBUG purposes, can be removed later
    showInMovedWindow("errors", errors / np.max(errors), 100 + (8 + roi.shape[1]) * 2, 200)

    start_x = end_x = errors.shape[1] - 1
    if offset_row is None:
        # when no empty row is found, search for the cut endpoints at the top & bottom so that the path is not broken
        start_x, end_x = find_min_cut_circ_endpoints(errors, roi)

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


def _x_squared_distance_1D(a_1d: np.ndarray, b_1d: np.ndarray) -> np.ndarray:
    """
    Computes the squared distance matrix based on 1D arrays of X-coordinates.
    """
    # Reshape for broadcasting: (N, 1) and (1, M)
    a_coords = a_1d.reshape(-1, 1)
    b_coords = b_1d.reshape(1, -1)
    squared_dist_matrix = (a_coords - b_coords) ** 2  # gets rid of the sign
    return squared_dist_matrix


def find_min_cut_circ_endpoints(errors: np.ndarray, roi: np.ndarray) -> tuple[int, int]:
    """
    Consider the roi as the following sections: | A | B | C | D |
    This functions rolls & crops it to get the section: | D | A |
    Then finds the best path to go from the start of D to the end of A.
    The points in the middle of the path, where D and A meet, are the points of interest.

    There may be multiple points at this boarder, since A* can noodle around.
    The coordinates of the pair of points nearest each other at y_center and y_center -1 is returned.
    If multiple pairs have the same minimum distance, only one is returned.

    Args:
        errors: polar unwrapped errors
        roi: polar unwrapped overlap roi
    Returns: a tuple containing the top and bottom row endpoints (column idx) for the min cut
    """
    import pyastar2d

    err_len_div4 = errors.shape[0] // 4
    errors = np.roll(errors, -err_len_div4, axis=0)[err_len_div4 * 2:, :]
    roi = np.roll(roi, -err_len_div4, axis=0)[err_len_div4*2:, :]
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
    dist_matrix = _x_squared_distance_1D(np.array(points_of_interest_top), np.array(points_of_interest_bottom))

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


def _find_circular_patch(lookup,
                         block: np.ndarray, mask: np.ndarray,
                         params: CircularPatchingConfig, rng: np.random.Generator):
    """

    :param lookup: TODO -> should be a list or SharedTextList
    :param block: 2D slice of the texture being patched containing the overlapping region.
        The array should be of type float.
    :param mask: Mask of the overlapping region. It should of type float, having values in the interval [0, 1].
        The mask can be weighted to prioritize or neglect certain spots within the overlapping region.
    :param tolerance:
    :param block_size:
    :param rng:
    :return:
    """
    block_size, tolerance = params.patch_params.block_size, params.tolerance

    # texture is for output
    # image is for patch search
    showInMovedWindow("roi used in match template", mask, 50 + block_size * 3, 25)

    # according to documentation only TM_SQDIFF and TM_CCORR_NORMED accept a mask
    # also, if given as a float it can be weighted... may come in handy later
    err_mat = cv2.matchTemplate(image=lookup, templ=block, mask=mask, method=cv2.TM_SQDIFF)

    #err_mat = 1 - ncs  # for TM_CCOEFF_NORMED
    if tolerance > 0:
        # attempt to ignore zeroes in order to apply tolerance, but mind edge case (e.g., blank image)
        min_val = np.min(pos_vals) if (pos_vals := err_mat[err_mat > 0]).size > 0 else 0
    else:
        min_val = np.min(err_mat)
    print(f"{min_val = }    err_threshhold = {(1.0 + tolerance) * min_val}      {np.any(min_val <= (1.0 + tolerance) * min_val)}")
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return lookup[y:y + block_size, x:x + block_size]


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

    config = CircularPatchingConfig(
        patch_params=CircularPatchParams(diameter=161, overlap_ratio=.5),
        spacing_factor=0.9,
        tolerance=0.0,
        outer_corners_weighted_template_matching=True
    )

    rng = np.random.default_rng(seed=0)
    debug_img = np.stack((mask.copy(),) * 3, axis=-1)
    img = circle_quilt(image=img, roi_mask=mask, lookup=lookup, debug_img=debug_img,
                       config=config, rng=rng)
    cv2.imshow("Result", img)
    cv2.imshow("Patches Visualizer", debug_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
