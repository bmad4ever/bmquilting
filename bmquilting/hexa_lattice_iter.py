from __future__ import annotations

from collections.abc import Callable, Generator
from multiprocessing import shared_memory
from joblib import Parallel, delayed
import numpy as np
import math
import cv2

# type Vec2_int = tuple[int, int]  BECOMES NOT PICKABLE, can't be used with parallel...
Vec2_int = tuple[int, int]
"""(x, y)"""

def _wrap_generator_func(user_func, generator_func, generator_args, shared_data):
    batch = generator_func(*generator_args)
    user_func(batch, shared_data)

class _DebugSet(set):
    def add(self, item):
        assert item not in self
        super().add(item)

    def update(self, *iterables):
        for iterable in iterables:
            for item in iterable:
                self.add(item)

    def __ior__(self, other):
        """Handles the 'set_a |= set_b' syntax"""
        self.update(other)
        return self


class HexagonalLatticeIterator:
    def __getstate__(self):
        """clear external funcs """
        return {k: v for k, v in self.__dict__.items() if k not in ["center_func", "process_func", "shared_data"]}

    def __init__(self, min_x, max_x, min_y, max_y, x_spacing, y_spacing,
                 center_func: Callable = None, process_func: Callable = None,
                 shared_data: dict = None):
        """
        :param process_func: should receive the following args: points_batch, shared_data
        """
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.center_func = center_func  # Function specifically for center point
        self.process_func = process_func  # Strategy pattern: user-defined processing function
        self.shared_data = shared_data  # Dictionary containing shared memory info


    def _is_valid_point(self, point: Vec2_int) -> bool:
        """Check if a point is within the lattice bounds"""
        x, y = point
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)

    def _snap_to_grid(self, x: float, y: float) -> Vec2_int:
        """Snap coordinates to the nearest hexagonal grid point"""
        y_grid = round(y / self.y_spacing) * self.y_spacing
        row = int(y_grid // self.y_spacing)
        x_offset = (self.x_spacing // 2) if (row % 2 != 0) else 0
        x_grid = round((x - x_offset) / self.x_spacing) * self.x_spacing + x_offset
        return int(x_grid), int(y_grid)

    def _is_on_grid(self, point: Vec2_int) -> bool:
        """Check if a point is exactly on the hexagonal grid"""
        x, y = point

        if y % self.y_spacing != 0:
            return False  # not on grid y-wise

        row = y // self.y_spacing
        x_offset = (self.x_spacing // 2) if (row % 2 != 0) else 0

        # Check if x is on grid (considering row offset)
        return (x - x_offset) % self.x_spacing == 0

    def _find_center_point(self) -> Vec2_int:
        """Find the lattice point nearest to the center"""
        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2

        # Snap to nearest grid point
        return self._snap_to_grid(center_x, center_y)

    def _get_hex_neighbors(self, point: Vec2_int) -> list[Vec2_int]:
        """Get the 6 adjacent hexagonal neighbors of a point"""
        x, y = point

        # Hexagonal neighbors depend on whether we're in an even or odd row
        neighbors = [
            (x + self.x_spacing, y),  # right
            (x + self.x_spacing // 2, y + self.y_spacing),  # bottom-right
            (x - self.x_spacing // 2, y + self.y_spacing),  # bottom-left
            (x - self.x_spacing, y),  # left
            (x - self.x_spacing // 2, y - self.y_spacing),  # top-left
            (x + self.x_spacing // 2, y - self.y_spacing),  # top-right
        ]

        # Filter to only include points that are valid and on grid
        return [n for n in neighbors if self._is_valid_point(n) and self._is_on_grid(n)]

    @staticmethod
    def _get_direction_vector(from_point: Vec2_int, to_point: Vec2_int) -> Vec2_int:
        """Calculate direction vector between two points"""
        return to_point[0] - from_point[0], to_point[1] - from_point[1]

    def _get_points_in_direction(self, start: Vec2_int, direction: Vec2_int) -> Generator[Vec2_int]:
        """Get all points in a given direction from start point"""
        current = (start[0] + direction[0], start[1] + direction[1])
        while (self._is_valid_point(current) and
               self._is_on_grid(current)  ):
            yield current
            current = (current[0] + direction[0], current[1] + direction[1])

    @staticmethod
    def _get_angle(center: Vec2_int, point: Vec2_int) -> float:
        """Calculate angle from center to point in radians"""
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return math.atan2(dy, dx)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [0, 2π)"""
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle

    def _is_angle_between(self, test_angle: float, start_angle: float, end_angle: float) -> bool:
        """Check if test_angle is between start_angle and end_angle (going counterclockwise)"""
        # Normalize all angles
        test = self._normalize_angle(test_angle)
        start = self._normalize_angle(start_angle)
        end = self._normalize_angle(end_angle)

        if start <= end:
            return start <= test <= end
        else:  # Wraps around 0
            return test >= start or test <= end

    def _get_sector_between(self, center: Vec2_int,
                            dir1_neighbor: Vec2_int, dir2_neighbor: Vec2_int) -> Generator[Vec2_int]:
        """
        Get points in the sector between angle1 and angle2,
        using scanline iteration along the sector boundaries.
        """
        # Calculate direction vectors
        dir1 = self._get_direction_vector(center, dir1_neighbor)
        dir2 = self._get_direction_vector(center, dir2_neighbor)
        angle1 = self._get_angle(center, dir1_neighbor)
        angle2 = self._get_angle(center, dir2_neighbor)

        # Outer loop: move along angle1 direction
        current_start = (center[0] + dir1[0], center[1] + dir1[1])

        while True:
            # Check if we've gone out of bounds along angle1
            if not self._is_valid_point(current_start):
                break

            # Inner loop: scan parallel to angle2 direction from current_start
            scan_point = current_start
            while True:
                # Move to next point along angle2 direction
                scan_point = (scan_point[0] + dir2[0], scan_point[1] + dir2[1])
                scan_point = self._snap_to_grid(*scan_point)

                # Check bounds
                if not self._is_valid_point(scan_point):
                    break

                point_angle = self._get_angle(center, scan_point)
                if self._is_angle_between(point_angle, angle1, angle2):
                    yield scan_point

            # If we didn't find any valid points in this scanline, we might be done
            # But continue to be safe (the sector might have gaps)

            # Move to next scanline along angle1
            current_start = (current_start[0] + dir1[0], current_start[1] + dir1[1])

    def _partition_remaining_points(self, center: Vec2_int, neighbors: list[Vec2_int], ) -> list[Generator[Vec2_int]]:
        """
        Partition points into 6 triangular sectors.
        Each sector is named by its angle and filled in scan-line fashion from center outward.
        """
        sector_points = [
            self._get_sector_between(center, neighbor, neighbors[(i + 1) % len(neighbors)])
            for i, neighbor in enumerate(neighbors)
        ]
        return sector_points

    def _debug_iterate_spiral(self):
        visited = _DebugSet()
        for batch in self.iterate_spiral():
            batch = list(batch)
            for x, y in batch:
                assert self._is_on_grid((x, y))
            visited.update(batch)
            yield batch

    def iterate_spiral(self):
        """
        Iterate through the lattice in spiral pattern from center

        Steps:
        1. Find center point
        2. Process 6 immediate neighbors
        3. Process points along 6 directions (in parallel if requested)
        4. Process regions between directions (in parallel if requested)

        NOTE: This is a generator for manual control. Use process_spiral() for automatic processing.
        """
        # Step 1: Find and process center point
        center = self._find_center_point()
        print(f"Center point: {center}")
        yield [center]

        # Step 2: Process 6 immediate neighbors
        neighbors = self._get_hex_neighbors(center)
        print(f"Immediate neighbors ({len(neighbors)} points)")
        yield neighbors

        # Step 3: Process points along 6 directions
        if len(neighbors) >= 2:
            directions = [self._get_direction_vector(center, n) for n in neighbors]

            directional_points = []
            for i, direction in enumerate(directions):
                points = self._get_points_in_direction(neighbors[i], direction)
                directional_points.append(points)

            print(f"Processing {len(directional_points)} directions sequentially")
            for points in directional_points:
                if points:
                    yield points

            # Step 4: Process regions between directions
            print("Processing regions between directions")
            sectors = self._partition_remaining_points(center, neighbors)

            print(f"Processing {len(sectors)} sectors sequentially")
            for sector_points in sectors:
                yield sector_points

    def process_spiral(self, n_processes=6):
        """
        Process the lattice using the strategy pattern with proper parallel execution:
        - First batch (center) processed with center_func (or process_func if center_func not set)
        - Second batch (neighbors) processed sequentially with process_func
        - 6 directional batches processed in parallel using joblib
        - 6 sector batches processed in parallel using joblib

        Requires process_func to be set during initialization.
        center_func is optional - if not set, process_func is used for center too.
        shared_data is optional - if provided, it will be passed to worker processes.
        """
        if self.process_func is None:
            raise ValueError("process_func must be set to use process_spiral()")

        # Use center_func if provided, otherwise use process_func
        center_processing_func = self.center_func if self.center_func is not None else self.process_func

        center = self._find_center_point()

        # Step 1: Process center with center_func (or process_func)
        print(f"Processing center point: {center}")
        center_batch = [center]
        center_processing_func(center_batch, self.shared_data)

        # Step 2: Process neighbors sequentially
        neighbors = self._get_hex_neighbors(center)
        print(f"Processing {len(neighbors)} immediate neighbors")
        self.process_func(neighbors, self.shared_data)

        # Step 3: Process 6 directions in parallel with joblib
        directions = [self._get_direction_vector(center, n) for n in neighbors]
        directional_points = []

        print(f"Processing {len(directional_points)} directions in parallel")
        parallel = Parallel(n_jobs=n_processes, backend="loky", timeout=None, verbose=0)
        parallel(
            delayed(_wrap_generator_func)(
                self.process_func,
                self._get_points_in_direction,
                (neighbors[i], direction),
                self.shared_data)
            for i, direction in enumerate(directions)
        )

        # Step 4: Process 6 sectors in parallel with joblib
        print("Processing sectors between directions")
        parallel(
            delayed(_wrap_generator_func)(
                self.process_func,
                self._get_sector_between,
                (center, neighbor, neighbors[(i + 1) % len(neighbors)]),
                self.shared_data)
            for i, neighbor in enumerate(neighbors)
        )


# Worker function to handle shared data
def _worker_with_shared_data(batch, func, shared_data):
    """Wrapper that reconstructs shared memory and calls the user function"""
    return func(batch, shared_data)


# Example usage
def my_center_func(batch, shared_data=None):
    """Special function for processing the center point"""
    print("  [CENTER] Processing center")

    if shared_data:
        shm = shared_memory.SharedMemory(name=shared_data['name'])
        debug_img = np.ndarray(shared_data['shape'], dtype=shared_data['dtype'], buffer=shm.buf)

        for x, y in batch:
            cv2.circle(debug_img, (int(x + 10), int(y + 10)), 5, (0, 0, 255), -1)


def my_func(batch, shared_data=None):
    """User-defined processing function for all other points"""
    if shared_data:
        shm = shared_memory.SharedMemory(name=shared_data['name'])
        debug_img = np.ndarray(shared_data['shape'], dtype=shared_data['dtype'], buffer=shm.buf)

        for x, y in batch:
            cv2.circle(debug_img, (int(x + 10), int(y + 10)), 5, (255, 255, 255), -1)
            cv2.imshow("ITER", debug_img)
            cv2.waitKey(0)



if __name__ == "__main__":
    # Create shared memory for the image
    debug_img = np.zeros((220, 120, 3), dtype=np.uint8)

    # Create shared memory block
    shm = shared_memory.SharedMemory(create=True, size=debug_img.nbytes)

    # Create a numpy array backed by shared memory
    shared_img = np.ndarray(debug_img.shape, dtype=debug_img.dtype, buffer=shm.buf)
    shared_img[:] = debug_img[:]  # Copy initial data

    # Package shared memory info
    shared_data = {
        'name': shm.name,
        'shape': debug_img.shape,
        'dtype': debug_img.dtype
    }

    print("=== Processing with shared memory ===\n")

    try:
        lattice = HexagonalLatticeIterator(
            min_x=0, max_x=100,
            min_y=0, max_y=200,
            x_spacing=10, y_spacing=10,
            center_func=my_center_func,
            process_func=my_func,
            shared_data=shared_data  # Pass shared memory info
        )

        #print(f"Total points in lattice: {len(lattice.points)}\n")

        # Process with automatic parallelization
        #lattice.process_spiral(n_processes=1)
        #print(f"\nFinished processing {len(processed_points)} points")
        for batch in lattice._debug_iterate_spiral():
            my_func(batch, shared_data)

        # Display final result
        cv2.imshow("Final Result", shared_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()