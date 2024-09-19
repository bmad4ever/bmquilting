import cv2
import numpy as np
import math
from numpy import random
from .synthesis_subroutines import apply_mask

# quilting using Circular Patches over a Hexagonal Lattice (CPHL)


# TODO
# -> test: test w/ additional textures
# -> cleanup: remove comments & unused
# -> tolerance must be an argument
# -> blend into patch feature
# -> expose patch "hole" function
# -> implement comfyui node
# -> implement "radial" parallel texture generation using this quilting variant


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)


def find_donut_outer_corners(mask: np.ndarray, inner_radius: int, outer_radius: int) -> list[tuple[int, int]]:
    """
    yes, this is a function name; and this, is its respective documentation...
    Args:
        mask: the mask w/ the donut
        inner_radius: inner radius of the donut
        outer_radius: outer radius of the donut

    Returns:
        list of points in the following tuple format: (y, x)
    """
    margin: int = 4
    extended_mask = np.zeros((margin*2 + mask.shape[0], margin*2 + mask.shape[1]), dtype=mask.dtype)
    extended_mask[4:-4, 4:-4] = mask

    dst = cv2.cornerHarris(extended_mask, blockSize=2, ksize=5, k=0.06)
    corners = np.where(dst > int(0.01 * dst.max()))  # return [[xs], [ys]]
    corners = zip(corners[1], corners[0])  # generator to iterate all corners; also swaps y & x
    corners = ((c[0] - margin, c[1] - margin) for c in corners)  # offset w/ margin

    # ___ filter out inner corners __________
    center = (outer_radius+margin, outer_radius+margin)
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

    showInMovedWindow("donut corners", corners, 50+shape[0]*1, 25)

    distance_map = cv2.distanceTransform(255 - corners, cv2.DIST_L2, 3, dst=corners)
    dst_norm = np.empty_like(distance_map, dtype=np.float32)
    dst_norm = cv2.normalize(distance_map, dst_norm, 0, 1, cv2.NORM_MINMAX)
    dst_norm = cv2.GaussianBlur(dst_norm, (15, 15), sigmaX=0, sigmaY=0)  # TODO this requires a min radius of ~8
    dst_norm = 1 - dst_norm

    showInMovedWindow('Weighted Mask', dst_norm, 50+shape[0]*2, 25)
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


def get_bounding_box(mask):
    """
    Get the bounding box of the black shape in the mask.

    Args:
    - mask (numpy.ndarray): The mask with the black shape.

    Returns:
    - bounding_box (tuple): The bounding box of the black shape in the format (x, y, w, h).
    """
    mask = 1 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    else:
        return (0, 0, 0, 0)  # edge case: no contours


kernelx3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#kernelx5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernelx5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def blur_donut_mask(mask, inner_radius):
    """
    UNUSED, likely not needed...
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


def reverse_blured_circle_mask(mask, overlap_radius):
    """
    Args:
        mask: circle mask as uint8 w/ values from [0 to 255]
    """
    _, width = mask.shape[:2]
    ksize = width//2 + 1
    ksize = (ksize, ksize)
    border_size = round(width / 2)
    blured_mask = cv2.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
    blured_mask[border_size:-border_size, border_size:-border_size] = mask
    np.subtract(255, blured_mask, out=blured_mask)
    blured_mask = cv2.morphologyEx(blured_mask, cv2.MORPH_DILATE, kernelx3, iterations=overlap_radius//2)
    blured_mask = cv2.GaussianBlur(blured_mask, ksize, sigmaX=0, sigmaY=0)
    blured_mask = blured_mask[border_size:-border_size, border_size:-border_size]
    blured_mask = blured_mask.astype(np.float32) / 255
    return blured_mask


def circle_quilt(img, roi_mask, lookup, radius=80, spacing_factor=1.0, overlap_r=40, debug_img=None):
    # for a spacing factor of .9, non-overlap must be at least half the radius
    # so overlap beyond half the radius may leave holes in the generation
    # instead of limiting the overlap though, it may be preferable to circle_quilt each hole individually recursively
    # not entirely sure this would lead to a better generation... something to think about

    img = cv2.copyMakeBorder(img, radius, radius, radius, radius, borderType=cv2.BORDER_REFLECT_101)
    roi_mask = cv2.copyMakeBorder(roi_mask, radius, radius, radius, radius, borderType=cv2.BORDER_REFLECT_101)

    f_radius = float(radius)

    min_x, min_y, w, h = get_bounding_box(roi_mask)
    max_x, max_y = min_x + w, min_y + h

    filled_mask = roi_mask.copy()

    cv2.morphologyEx(roi_mask, cv2.MORPH_ERODE, kernel=kernelx3, iterations=radius // 2, dst=roi_mask)

    cv2.rectangle(debug_img, (int(min_x - radius), int(min_y - radius)), (int(max_x - radius), int(max_y - radius)),
                  (255, 0, 0), 2)  # TODO remove debug

    #circ_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(radius * 2), int(radius * 2)))
    #hole_mask = np.zeros_like(circ_roi)
    overlap_r_kernel_adj = overlap_r +1 if overlap_r%2 == 0 else overlap_r
    overlap_kernel = (overlap_r_kernel_adj, overlap_r_kernel_adj)
    del overlap_r_kernel_adj

    non_overlap_radius = radius - overlap_r
    circ_roi = np.zeros((int(radius*2), int(radius*2)), dtype=np.uint8)
    cv2.circle(circ_roi, (radius, radius), radius, (1,), -1)
    donut_roi = circ_roi.copy()  # used as mask for template matching
    cv2.circle(donut_roi, (radius, radius), round(non_overlap_radius), (0,), -1)

    weighted_circle_roi = reverse_blured_circle_mask(circ_roi*255, overlap_r)  # weights less near the center
    #showInMovedWindow("wc", weighted_circle_roi, 500, 25)

    #cv2.imshow("donut_roi", circ_roi); cv2.waitKey(0); quit()

    center = [d / 2 - .5 for d in circ_roi.shape]
    polar_unwrapped_size = (radius, round(radius * 2 * np.pi))

    # iterate hexagonal lattice
    x_spacing = int(radius * 2 / 1.5 * math.cos(math.pi / 6) * spacing_factor)  # Horizontal spacing
    y_spacing = int(radius * spacing_factor)  # Vertical spacing
    for y in range(int(min_y), int(max_y + y_spacing), y_spacing):
        for x in range(int(min_x), int(max_x + x_spacing), x_spacing):
            if (y // y_spacing) % 2 != 0:
                x += x_spacing // 2

            if roi_mask[y, x] == 0:  # point within delimited roi

                # TODO likely a good idea to extract the code below into a method!

                if debug_img is not None:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.circle(debug_img, (int(x - radius), int(y - radius)), radius, color, -1)
                    cv2.circle(debug_img, (int(x - radius), int(y - radius)), 5, (0, 0, 0), -1)

                y1, y2, x1, x2 = y - radius, y + radius, x - radius, x + radius
                block, roi = img[y1:y2, x1:x2], filled_mask[y1:y2, x1:x2]
                roi = donut_roi*roi
                corners = find_donut_outer_corners(roi, non_overlap_radius, radius)
                weighted_roi = distance_to_points(roi.shape[:2], corners)

                # prior solutions, TODO remove later
                #weighted_roi = weighted_circle_roi * weighted_roi * roi
                #weighted_roi = np.maximum(weighted_circle_roi, weighted_roi) * roi

                weighted_roi = np.maximum(weighted_circle_roi, weighted_roi)
                weighted_roi = cv2.GaussianBlur(weighted_roi, overlap_kernel, sigmaX=0, sigmaY=0)
                weighted_roi = weighted_roi * roi
                showInMovedWindow("weighted roi", weighted_roi, 50+radius*2*3, 25)

                print(f"shapes  block={block.shape}  roi={roi.shape}")
                patch: np.ndarray = find_circular_patch(lookup, block, roi, 0.0 ,
                                                        radius * 2)  # TODO tolerance should not be hardcoded

                #cv2.imshow("block", block)
                cv2.imshow("roi", roi*255)
                #cv2.imshow("Patch", apply_mask(patch, circ_roi));
                #cv2.waitKey(0); quit()

                warp_flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST
                polar_block, polar_patch, polar_roi = [
                    cv2.warpPolar(i, (radius, round(radius * 2 * np.pi)), center, f_radius, warp_flags)
                    for i in [block, patch, roi]]
                #cv2.imshow("n", polar_patch);
                #cv2.waitKey(0); quit()
                mask = min_cut_circ(polar_block, polar_patch, polar_roi, non_overlap_radius)
                mask = cv2.warpPolar(mask, roi.shape[:2], center, f_radius,
                                     cv2.WARP_INVERSE_MAP | warp_flags)
                #mask = np.maximum(mask, (1 - roi))
                #mask *= circ_roi

                #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernelx3)
                #mask = cv2.blur(mask, (3, 3))
                showInMovedWindow("mask", mask * 255, 50+radius*2*4, 25)
                #cv2.imshow("roi", roi * 255)
                #cv2.imshow("Patch", apply_mask(patch, mask))
                #cv2.waitKey(0); quit()

                img[y1:y2, x1:x2] = apply_mask(img[y1:y2, x1:x2], (1 - mask)) + apply_mask(patch, mask)
                showInMovedWindow("patched single", img, 1920 - img.shape[1], 25)
                cv2.waitKey(0)  #;quit()

                #update filled mask
                #np.maximum(circ_roi, filled_mask[y1:y2, x1:x2], out=filled_mask[y1:y2, x1:x2])
                np.maximum(mask, filled_mask[y1:y2, x1:x2], out=filled_mask[y1:y2, x1:x2])
                #cv2.imshow("filled state", filled_mask*255); cv2.waitKey(0)
    return img[radius:-radius, radius:-radius]


def find_first_2adjacent_all_zero_rows(mask):
    """return the index of the second all zero row"""
    prev = False
    for i, row in enumerate(mask):
        current = all(pixel == 0 for pixel in row)
        if prev and current:
            return i
        prev = current
    return None


def min_cut_circ(block, patch, roi, non_overlap_radius):
    """
    Args:
        block: polar unwrapped block
        patch: polar unwrapped patch
        roi:  polar unwrapped overlap roi ( stored in filled mask )
        non_overlap_radius: inner radius where overlap is not considered ( could use the donut roi, but this is faster )
    Returns:

    """
    import pyastar2d
    # forced debug case, remove the following line later TODO
    #roi[40:50, :] = 0  # force all zeros row

    errs = ((patch - block) ** 2).mean(2)
    #empty_row = find_first_all_zero_row(roi)
    offset_row = find_first_2adjacent_all_zero_rows(roi[:, non_overlap_radius:])
    #offset_row = find_edge_escape_path_with_at_least_2pixels(roi)
    if offset_row is not None:
        errs = np.roll(errs, -offset_row, axis=0)
        roi = np.roll(roi, -offset_row, axis=0)
        block = np.roll(block, -offset_row, axis=0)


    showInMovedWindow("rolled polar roi", roi * 255, 100, 200)
    showInMovedWindow("rolled polar block", block, 100 + 8 + roi.shape[1], 200)
    errs[roi == 0] = 0
    showInMovedWindow("errs", errs/np.max(errs), 100 +(8+roi.shape[1])*2, 200)

    start_x = end_x = errs.shape[1] - 1
    if offset_row is None:
        # when no empty row is found, search for the cut endpoints at the top & bottom so that the path is not broken
        start_x, end_x = find_min_cut_circ_endpoints(errs, roi, errs.shape[0]//4)

    maze = errs.copy()
    maze *= errs.shape[0] ** 3
    maze += 1
    maze *= errs.shape[0] ** 3
    maze[:, -1] = roi[:, -1] * maze[:, -1] + (1 - roi[:, -1])  # bottom holes escape path
    maze[:, :-1][roi[:, :-1] == 0] = np.inf  # don't travel outside the mask, unless via the escape path

    start = (0, start_x)
    end = (maze.shape[0] - 1, end_x)
    cv2.waitKey(0)

    path = pyastar2d.astar_path(maze, start, end, allow_diagonal=True)
    mask = np.ones((errs.shape[0], errs.shape[1] + 1), dtype=roi.dtype)

    print(f"mask.shape={mask.shape}")
    for i, j in path:  # draw path
        mask[i, j] = 0

    showInMovedWindow("cut path", mask * 255,  100 + (8+roi.shape[1])*3, 200)

    print(f"seed point={(mask.shape[0] - 1, mask.shape[1] - 1)}")
    cv2.floodFill(mask, None, (mask.shape[1] - 1, mask.shape[0] - 1), (0,))
    #cv2.floodFill(mask, None, (0, 0), (0,))
    if offset_row is not None:
        mask = np.roll(mask, offset_row, axis=0)
    showInMovedWindow("cut mask", mask * 255, 100 + (8 + roi.shape[1])*4, 200)
    #cv2.waitKey(0)

    return mask[:, :-2]


def find_min_cut_circ_endpoints(errs, roi, half_section_size):
    """

    Args:
        errs:
        roi: polar unwrapped overlap roi
        section_size: number of rows to use in the "mock" sample
    Returns: a tuple containing the top and bottom row endpoints (column idx) for the min cut
    """
    import pyastar2d
    from scipy.spatial.distance import cdist

    errs = np.roll(errs, -half_section_size, axis=0)[half_section_size*2:, :]
    roi = np.roll(roi, -half_section_size, axis=0)[half_section_size*2:, :]
    showInMovedWindow("rolled roi section", roi*255,  100 + (8 + roi.shape[1])*5, 200)

    maze = np.ones((errs.shape[0] + 2, errs.shape[1]), dtype=np.float32)
    maze_area = maze[1:-1, :]
    maze_area[:, :] = errs
    #maze_area[roi == 0] = 1 #errs.shape[1]
    maze *= errs.shape[0] ** 3
    maze += 1
    maze *= errs.shape[0] ** 3
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[1:-1, -1] = roi[:, -1] * maze[1:-1, -1] + (1 - roi[:, -1])  # bottom holes escape path
    maze[1:-1, :-1][roi[:,:-1]==0] = np.inf  # don't travel outside the mask, unless via the escape path


    start = (0, maze.shape[1] - 1)
    end = (maze.shape[0] - 1, maze.shape[1] - 1)

    path = pyastar2d.astar_path(maze, start, end, allow_diagonal=True)
    mask = np.ones((errs.shape[0], errs.shape[1] + 1), dtype=roi.dtype)

    print(f"mask.shape={mask.shape}")
    half_section_size_m1 = half_section_size - 1
    points_of_interest_top = [(y-1, x) for (y, x) in path if y-1 == half_section_size]
    points_of_interest_bottom = [(y-1, x) for (y, x) in path if y-1 == half_section_size_m1]

    # Compute the distance matrix between points in la and lb
    dist_matrix = cdist(points_of_interest_top, points_of_interest_bottom, metric='euclidean')

    # Find the indices of the minimum distance
    min_index = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    top_point = points_of_interest_top[min_index[0]]
    bottom_point = points_of_interest_bottom[min_index[1]]
    #min_distance = dist_matrix[min_index]

    print(f"points of interest = {(top_point, bottom_point)}")

    # TODO debug code, not actually needed, remove later
    mask = np.zeros_like(maze)
    for i, j in path:
        mask[i, j] = 1
    showInMovedWindow("aux_cut", mask,  100 + (8 + roi.shape[1])*6, 200)

    return top_point[1], bottom_point[1]



def find_circular_patch(lookup, block, mask, tolerance, block_size, rng=None):  # TODO rng
    # texture is for output
    # image is for patch search

    # according to documentation only TM_SQDIFF and TM_CCORR_NORMED accept a mask
    # also, if given as a float it can be weighted... may come in handy later
    err_mat = cv2.matchTemplate(image=lookup, templ=block, mask=mask, method=cv2.TM_SQDIFF)

    #err_mat = 1 - ncs  # for TM_CCOEFF_NORMED
    if tolerance > 0:
        # attempt to ignore zeroes in order to apply tolerance, but mind edge case (e.g., blank image)
        min_val = np.min(pos_vals) if (pos_vals := err_mat[err_mat > 0]).size > 0 else 0
    else:
        min_val = np.min(err_mat)
    print(f"{min_val}    {(1.0 + tolerance) * min_val}      {np.any(min_val <= (1.0 + tolerance) * min_val)}")
    print(np.argwhere(err_mat <= 0))
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = np.random.randint(len(y))  # rng.integers(len(y)) # TODO replace w/ generator
    y, x = y[c], x[c]
    return lookup[y:y + block_size, x:x + block_size]


if __name__ == "__main__":
    text_idx = "18"
    src = cv2.imread(f"results/t{text_idx}.png", cv2.IMREAD_COLOR)
    src2 = cv2.imread(f"textures/t{text_idx}.png", cv2.IMREAD_COLOR)
    src = np.float32(src) / 255
    src2 = np.float32(src2) / 255

    img = cv2.resize(src, [d * 2 for d in src.shape[:2][::-1]])
    lookup = cv2.resize(src2, [d * 2 for d in src2.shape[:2][::-1]])

    mask = create_mock_mask(img.shape)
    print(f"shapes {(img.shape, mask.shape)}")
    img[mask == 0] = (0, 0, 0)

    debug_img = np.stack((mask.copy(),) * 3, axis=-1)
    img = circle_quilt(img=img, roi_mask=mask, lookup=lookup, debug_img=debug_img)
    cv2.imshow("Result", img)
    cv2.imshow("Patches Visualizer", debug_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
