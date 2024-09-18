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
    cv2.imshow(winname,img)


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


def circle_quilt(img, roi_mask, lookup, radius=64, spacing_factor=1.1, overlap_r=32, debug_img=None):
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
    non_overlap_radius = radius - overlap_r
    circ_roi = np.zeros((int(radius*2), int(radius*2)), dtype=np.uint8)
    cv2.circle(circ_roi, (radius, radius), radius, (1,), -1)
    donut_roi = circ_roi.copy()  # used as mask for template matching
    cv2.circle(donut_roi, (radius, radius), round(non_overlap_radius), (0,), -1)

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

                if debug_img is not None:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.circle(debug_img, (int(x - radius), int(y - radius)), radius, color, -1)
                    cv2.circle(debug_img, (int(x - radius), int(y - radius)), 5, (0, 0, 0), -1)

                y1, y2, x1, x2 = y - radius, y + radius, x - radius, x + radius
                block, roi = img[y1:y2, x1:x2], filled_mask[y1:y2, x1:x2]
                roi = donut_roi*roi
                #roi = circ_roi*roi
                # TODO to investigate...
                #   areas near "donut" breaks should be considered more important to match (I'm guessing)
                #   consider using float mask weighted w/ the above in mind
                print(f"shapes  block={block.shape}  roi={roi.shape}")
                patch: np.ndarray = find_circular_patch(lookup, block, roi, 0.0,
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
                mask = np.maximum(mask, (1 - roi))
                mask *= circ_roi
                #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernelx3)
                #mask = cv2.blur(mask, (3, 3))
                #cv2.imshow("mask", mask * 255)
                #cv2.imshow("roi", roi * 255)
                #cv2.imshow("Patch", apply_mask(patch, mask))
                #cv2.waitKey(0); quit()

                img[y1:y2, x1:x2] = apply_mask(img[y1:y2, x1:x2], (1 - mask)) + apply_mask(patch, mask)
                cv2.imshow("patched single", img);
                cv2.waitKey(0)  #;quit()

                #update filled mask
                np.maximum(circ_roi, filled_mask[y1:y2, x1:x2], out=filled_mask[y1:y2, x1:x2])
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
    empty_row = find_first_2adjacent_all_zero_rows(roi[:, non_overlap_radius:])
    if empty_row is not None:
        errs = np.roll(errs, -empty_row, axis=0)
        roi = np.roll(roi, -empty_row, axis=0)
        block = np.roll(block, -empty_row, axis=0)

    showInMovedWindow("rolled polar roi", roi * 255, 100, 200)
    showInMovedWindow("rolled polar block", block, 100 + 8 + roi.shape[1], 200)
    errs[roi == 0] = 0
    showInMovedWindow("errs", errs/np.max(errs), 100 +(8+roi.shape[1])*2, 200)

    maze = np.ones((errs.shape[0] + 2, errs.shape[1]), dtype=np.float32)
    maze_area = maze[1:-1, :]
    maze_area[:, :] = errs
    maze_area[roi == 0] = 1 #errs.shape[1]
    maze *= errs.shape[0] ** 3
    maze += 1
    maze *= errs.shape[0] ** 3
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[1:-1, -1] = roi[:, -1] * maze[1:-1, -1] + (1 - roi[:, -1])  # bottom holes escape path
    #maze[1:-1,:][roi==0] = 1 # don't try this, it won't work
    maze[:, :non_overlap_radius] = np.inf  # TODO define or make configurable the overlap outer radius


    start = (0, maze.shape[1] - 1)
    end = (maze.shape[0] - 1, maze.shape[1] - 1)

    path = pyastar2d.astar_path(maze, start, end, allow_diagonal=False)
    mask = np.ones((errs.shape[0], errs.shape[1] + 1), dtype=roi.dtype)

    shape_m2 = maze.shape[0] - 2

    print(f"mask.shape={mask.shape}")

    start_index = 0  # find start index to avoid checking 0 < i every iteration
    for idx, (i, j) in enumerate(path):
        if 0 < i:
            start_index = idx
            break

    for i, j in path[start_index:]:  # draw path
        mask[i - 1, j] = 0
        if i >= shape_m2:
            break

    showInMovedWindow("cut path", mask * 255,  100 + (8+roi.shape[1])*3, 200)

    print(f"seed point={(mask.shape[0] - 1, mask.shape[1] - 1)}")
    cv2.floodFill(mask, None, (mask.shape[1] - 1, mask.shape[0] - 1), (0,))
    #cv2.floodFill(mask, None, (0, 0), (0,))
    if empty_row is not None:
        mask = np.roll(mask, empty_row, axis=0)
    showInMovedWindow("cut mask", mask * 255, 100 + (8 + roi.shape[1])*4, 200)
    cv2.waitKey(0)

    return mask[:, :-2]


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
    text_idx = "10"
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
