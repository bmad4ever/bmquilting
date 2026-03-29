import numpy as np
import cv2

from bmquilting._internal.hexagonal_lattice import HexagonalLatticeIterator


if __name__ == "__main__":
    height, width = 384, 384
    radius = 16

    debug_img = np.zeros((height+2*radius, width+2*radius, 3), dtype=np.uint8)
    print("Press any key to advance the iteration.")

    lattice = HexagonalLatticeIterator(
        min_x=radius, max_x=width+radius,
        min_y=radius, max_y=height+radius,
        spacing=radius*2
    )

    batch_colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 255, 255),
     ]
    color_idx:int = 0
    batch_idx:int = 0
    for batch in lattice._debug_iterate_spiral():
        color = batch_colors[color_idx]
        if batch_idx == 0 or batch_idx == 1 or batch_idx == 7:
            color_idx += 1
        print(f"Processing batch {batch_idx}")
        for x, y in batch:
            cv2.circle(debug_img, (x, y), radius, color, -1)
            cv2.imshow("Spiral Hexagonal Lattice Iterator Visualizer", debug_img)
            cv2.waitKey(0)

        batch_idx += 1

    cv2.destroyAllWindows()