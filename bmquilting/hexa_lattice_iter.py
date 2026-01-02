import numpy as np
from multiprocessing import Pool, shared_memory
from functools import partial
import cv2
from typing import List, Tuple, Set, Callable
from joblib import Parallel, delayed


class HexagonalLatticeIterator:
    def __init__(self, min_x, max_x, min_y, max_y, x_spacing, y_spacing,
                 center_func: Callable = None, process_func: Callable = None,
                 shared_data: dict = None):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.center_func = center_func  # Function specifically for center point
        self.process_func = process_func  # Strategy pattern: user-defined processing function
        self.shared_data = shared_data  # Dictionary containing shared memory info

        # Generate all lattice points
        self.points = self._generate_all_points()

    def _generate_all_points(self) -> Set[Tuple[int, int]]:
        """Generate all points in the hexagonal lattice"""
        points = set()
        for y in range(int(self.min_y), int(self.max_y + self.y_spacing), self.y_spacing):
            for x in range(int(self.min_x), int(self.max_x + self.x_spacing), self.x_spacing):
                actual_x = x
                if (y // self.y_spacing) % 2 != 0:
                    actual_x = x + self.x_spacing // 2
                points.add((actual_x, y))
        return points

    def _find_center_point(self) -> Tuple[int, int]:
        """Find the lattice point nearest to the center"""
        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2

        min_dist = float('inf')
        center_point = None

        for point in self.points:
            dist = (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2
            if dist < min_dist:
                min_dist = dist
                center_point = point

        return center_point

    def _get_hex_neighbors(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get the 6 adjacent hexagonal neighbors of a point"""
        x, y = point
        row = y // self.y_spacing

        # Hexagonal neighbors depend on whether we're in an even or odd row
        if row % 2 == 0:  # even row
            neighbors = [
                (x + self.x_spacing, y),  # right
                (x + self.x_spacing // 2, y + self.y_spacing),  # bottom-right
                (x - self.x_spacing // 2, y + self.y_spacing),  # bottom-left
                (x - self.x_spacing, y),  # left
                (x - self.x_spacing // 2, y - self.y_spacing),  # top-left
                (x + self.x_spacing // 2, y - self.y_spacing),  # top-right
            ]
        else:  # odd row
            neighbors = [
                (x + self.x_spacing, y),  # right
                (x + self.x_spacing // 2, y + self.y_spacing),  # bottom-right
                (x - self.x_spacing // 2, y + self.y_spacing),  # bottom-left
                (x - self.x_spacing, y),  # left
                (x - self.x_spacing // 2, y - self.y_spacing),  # top-left
                (x + self.x_spacing // 2, y - self.y_spacing),  # top-right
            ]

        # Filter to only include points that exist in the lattice
        return [n for n in neighbors if n in self.points]

    def _get_direction_vector(self, from_point: Tuple[int, int], to_point: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate direction vector between two points"""
        return (to_point[0] - from_point[0], to_point[1] - from_point[1])

    def _get_points_in_direction(self, start: Tuple[int, int], direction: Tuple[int, int],
                                 visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get all points in a given direction from start point"""
        points = []
        current = (start[0] + direction[0], start[1] + direction[1])

        while current in self.points and current not in visited:
            points.append(current)
            current = (current[0] + direction[0], current[1] + direction[1])

        return points

    def _get_angle(self, center: Tuple[int, int], point: Tuple[int, int]) -> float:
        """Calculate angle from center to point in radians"""
        import math
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return math.atan2(dy, dx)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [0, 2π)"""
        import math
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle

    def _is_angle_between(self, test_angle: float, start_angle: float, end_angle: float) -> bool:
        """Check if test_angle is between start_angle and end_angle (going counterclockwise)"""
        import math

        # Normalize all angles
        test = self._normalize_angle(test_angle)
        start = self._normalize_angle(start_angle)
        end = self._normalize_angle(end_angle)

        if start <= end:
            return start <= test <= end
        else:  # Wraps around 0
            return test >= start or test <= end

    def _get_sector_at_angle(self, center: Tuple[int, int], angle1: float, angle2: float,
                             visited: Set[Tuple[int, int]], sector_name: str) -> List[Tuple[int, int]]:
        """
        Get all unvisited points in the sector between angle1 and angle2,
        sorted by distance from center (nearest first for scan-line filling)
        """
        import math

        sector_points = []

        for point in self.points:
            if point in visited:
                continue

            point_angle = self._get_angle(center, point)

            # Check if point is in this sector
            if self._is_angle_between(point_angle, angle1, angle2):
                dist = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
                sector_points.append((dist, point))

        # Sort by distance (scan from center outward)
        sector_points.sort(key=lambda x: x[0])

        return [point for dist, point in sector_points]

    def _partition_remaining_points(self, center: Tuple[int, int],
                                    neighbors: List[Tuple[int, int]],
                                    directional_points: List[List[Tuple[int, int]]],
                                    visited: Set[Tuple[int, int]]) -> List[Tuple[str, List[Tuple[int, int]]]]:
        """
        Partition all remaining unvisited points into 6 triangular sectors.
        Each sector is named by its angle and filled in scan-line fashion from center outward.
        """
        import math

        # Calculate angles for each of the 6 directions
        sector_info = []
        for i, neighbor in enumerate(neighbors):
            angle = self._get_angle(center, neighbor)
            sector_info.append({
                'index': i,
                'angle': angle,
                'neighbor': neighbor
            })

        # Sort by angle to get proper ordering
        sector_info.sort(key=lambda x: x['angle'])

        # Create 6 sectors, each between consecutive angles
        sectors = []
        for i in range(len(sector_info)):
            angle1 = sector_info[i]['angle']
            angle2 = sector_info[(i + 1) % len(sector_info)]['angle']

            # Create descriptive name based on angle
            angle1_deg = math.degrees(angle1) % 360
            angle2_deg = math.degrees(angle2) % 360
            sector_name = f"sector_{int(angle1_deg)}_{int(angle2_deg)}"

            # Get points in this sector, sorted by distance from center
            sector_points = self._get_sector_at_angle(center, angle1, angle2, visited, sector_name)

            if sector_points:
                sectors.append((sector_name, sector_points))

        return sectors

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
        visited = set()

        # Step 1: Find and process center point
        center = self._find_center_point()
        print(f"Center point: {center}")
        yield [center]
        visited.add(center)

        # Step 2: Process 6 immediate neighbors
        neighbors = self._get_hex_neighbors(center)
        print(f"Immediate neighbors ({len(neighbors)} points)")
        yield neighbors
        visited.update(neighbors)

        # Step 3: Process points along 6 directions
        if len(neighbors) >= 2:
            directions = [self._get_direction_vector(center, n) for n in neighbors]

            directional_points = []
            for i, direction in enumerate(directions):
                points = self._get_points_in_direction(neighbors[i], direction, visited)
                directional_points.append(points)

            print(f"Processing {len(directional_points)} directions sequentially")
            for points in directional_points:
                if points:
                    yield points
                    visited.update(points)

            # Step 4: Process regions between directions
            print("Processing regions between directions")
            sectors = self._partition_remaining_points(center, neighbors, directional_points, visited)

            print(f"Processing {len(sectors)} sectors sequentially")
            for sector_name, sector_points in sectors:
                print(f"  {sector_name}: {len(sector_points)} points")
                yield sector_points
                visited.update(sector_points)

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

        visited = set()  # Simple set, no need for Manager since we update after batches complete

        center = self._find_center_point()

        # Step 1: Process center with center_func (or process_func)
        print(f"Processing center point: {center}")
        center_batch = [center]
        center_processing_func(center_batch, self.shared_data)
        visited.add(center)

        # Step 2: Process neighbors sequentially
        neighbors = self._get_hex_neighbors(center)
        print(f"Processing {len(neighbors)} immediate neighbors")
        self.process_func(neighbors, self.shared_data)
        visited.update(neighbors)

        if len(neighbors) < 2:
            print(f"Total points processed: {len(visited)}")
            return

        # Step 3: Process 6 directions in parallel with joblib
        directions = [self._get_direction_vector(center, n) for n in neighbors]
        directional_points = []

        for i, direction in enumerate(directions):
            points = self._get_points_in_direction(neighbors[i], direction, visited)
            directional_points.append(points)

        print(f"Processing {len(directional_points)} directions in parallel")
        parallel = Parallel(n_jobs=n_processes, backend="loky", timeout=None, verbose=0)
        parallel(
            delayed(self.process_func)(batch, self.shared_data)
            for batch in directional_points
        )

        # Update visited after all parallel processes complete
        for points in directional_points:
            visited.update(points)

        # Step 4: Process 6 sectors in parallel with joblib
        print("Processing sectors between directions")
        sectors = self._partition_remaining_points(center, neighbors, directional_points, visited)

        sector_points_list = [sector_points for _, sector_points in sectors]

        print(f"Processing {len(sector_points_list)} sectors in parallel")
        parallel(
            delayed(self.process_func)(batch, self.shared_data)
            for batch in sector_points_list
        )

        # Update visited after all parallel processes complete
        for sector_points in sector_points_list:
            visited.update(sector_points)

        print(f"Total points processed: {len(visited)}")

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

        for y, x in batch:
            cv2.circle(debug_img, (int(x + 10), int(y + 5)), 5, (0, 0, 255), -1)


def my_func(batch, shared_data=None):
    """User-defined processing function for all other points"""
    print(f"  Processing batch of {len(batch)} points")

    if shared_data:
        shm = shared_memory.SharedMemory(name=shared_data['name'])
        debug_img = np.ndarray(shared_data['shape'], dtype=shared_data['dtype'], buffer=shm.buf)

        for y, x in batch:
            cv2.circle(debug_img, (int(x + 10), int(y + 5)), 5, (255, 255, 255), -1)
            cv2.imshow("ITER", debug_img)
            cv2.waitKey(0)



if __name__ == "__main__":
    # Create shared memory for the image
    debug_img = np.zeros((120, 120, 3), dtype=np.uint8)

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
            min_y=0, max_y=100,
            x_spacing=10, y_spacing=10,
            center_func=my_center_func,
            process_func=my_func,
            shared_data=shared_data  # Pass shared memory info
        )

        print(f"Total points in lattice: {len(lattice.points)}\n")

        # Process with automatic parallelization
        lattice.process_spiral(n_processes=1)
        #print(f"\nFinished processing {len(processed_points)} points")

        # Display final result
        cv2.imshow("Final Result", shared_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()