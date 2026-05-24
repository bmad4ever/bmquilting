from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from joblib import Parallel, delayed
import numpy as np
import math

from numpy.random.bit_generator import SeedSequence

# type Vec2_int = tuple[int, int]  BECOMES NOT PICKABLE, can't be used with parallel...
Vec2_int = tuple[int, int]
"""(x, y)"""

def _wrap_generator_func(user_func, generator_func, generator_args, shared_data=None, job_id:int=0, seed:SeedSequence=0):
    batch = generator_func(*generator_args)
    user_func(batch, shared_data, job_id, seed)

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

    def __init__(self, min_x, max_x, min_y, max_y, spacing,
                 first_patch_func: Callable[[Vec2_int, dict, int, SeedSequence], None] = None,
                 process_func: Callable[[Iterable[Vec2_int], dict, int, SeedSequence], None] = None,
                 shared_data: dict = None):
        """
        :param process_func: should receive the following args: points_batch, shared_data, job_id, generator's seed
        :param first_patch_func: should receive the following args: point, shared_data, job_id, generator's seed
        """
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.spacing = spacing
        self.first_point_func = first_patch_func
        self.process_func = process_func
        self.shared_data = shared_data


    def _is_valid_point(self, point: Vec2_int) -> bool:
        """Check if a point is within the lattice bounds"""
        x, y = point
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)

    def _snap_to_grid(self, x: float, y: float) -> Vec2_int:
        """Snap coordinates to the nearest hexagonal grid point"""
        y_grid = round(y / self.spacing) * self.spacing
        row = int(y_grid // self.spacing)
        x_offset = (self.spacing // 2) if (row % 2 != 0) else 0
        x_grid = round((x - x_offset) / self.spacing) * self.spacing + x_offset
        return int(x_grid), int(y_grid)

    def _is_on_grid(self, point: Vec2_int) -> bool:
        """Check if a point is exactly on the hexagonal grid"""
        x, y = point

        if y % self.spacing != 0:
            return False  # not on grid y-wise

        row = y // self.spacing
        x_offset = (self.spacing // 2) if (row % 2 != 0) else 0

        # Check if x is on grid (considering row offset)
        return (x - x_offset) % self.spacing == 0

    def _find_center_point(self) -> Vec2_int:
        """Find the lattice point nearest to the center"""
        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2

        # Snap to nearest grid point
        return self._snap_to_grid(center_x, center_y)

    def _get_hex_neighbors(self, point: Vec2_int) -> list[Vec2_int]:
        """Get the 6 adjacent hexagonal neighbors of a point"""
        x, y = point
        neighbors = [
            (x + self.spacing, y),  # right
            (x + self.spacing // 2, y + self.spacing),  # bottom-right
            (x - self.spacing // 2, y + self.spacing),  # bottom-left
            (x - self.spacing, y),  # left
            (x - self.spacing // 2, y - self.spacing),  # top-left
            (x + self.spacing // 2, y - self.spacing),  # top-right
        ]
        return [self._snap_to_grid(*n) for n in neighbors]

    @staticmethod
    def _get_direction_vector(from_point: Vec2_int, to_point: Vec2_int) -> Vec2_int:
        """Calculate direction vector between two points"""
        return to_point[0] - from_point[0], to_point[1] - from_point[1]

    def _get_points_in_direction(self, start: Vec2_int, direction: Vec2_int) -> Generator[Vec2_int]:
        """Get all points in a given direction from start point"""
        current = (start[0] + direction[0], start[1] + direction[1])
        while self._is_valid_point(current):
            current = self._snap_to_grid(*current)
            yield current
            current = (current[0] + direction[0], current[1] + direction[1])

    def _get_angle(self, center: Vec2_int, point: Vec2_int) -> float:
        """Calculate angle from center to point in radians using ideal hexagonal coordinates"""
        # In our hexagonal grid, odd rows are offset by spacing // 2.
        # The "ideal" hexagonal offset is spacing / 2.0.
        # We calculate the angle as if the points were on an ideal hexagonal grid.
        offset_diff = (self.spacing / 2.0) - (self.spacing // 2)

        c_x, c_y = center
        p_x, p_y = point

        c_row = c_y // self.spacing
        p_row = p_y // self.spacing

        c_ideal_x = c_x + (c_row % 2) * offset_diff
        p_ideal_x = p_x + (p_row % 2) * offset_diff

        dx = p_ideal_x - c_ideal_x
        dy = p_y - c_y
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
            current_start = current_start[0] + dir1[0], current_start[1] + dir1[1]
            current_start = self._snap_to_grid(*current_start)

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

    def _debug_iterate_spiral(self) -> Iterable[Iterable[Vec2_int]]:
        visited = _DebugSet()
        for batch in self.iterate_spiral():
            batch = list(batch)
            for x, y in batch:
                assert self._is_on_grid((x, y))
            visited.update(batch)
            yield batch

    def iterate_spiral(self) -> Iterable[Iterable[Vec2_int]]:
        """
        Iterate through the lattice in spiral pattern from center.
        :return: yields point batches.

        Steps:
        1. Find center point
        2. Process 6 immediate neighbors
        3. Process points along 6 directions (in parallel if requested)
        4. Process regions between directions (in parallel if requested)

        NOTE: This is a generator for manual control. Use process_spiral() for automatic processing.
        """
        # Step 1: Find and process center point
        center = self._find_center_point()
        yield [center]

        # Step 2: Process 6 immediate neighbors
        neighbors = self._get_hex_neighbors(center)
        yield neighbors

        # Step 3: Process points along 6 directions
        if len(neighbors) >= 2:
            directions = [self._get_direction_vector(center, n) for n in neighbors]

            directional_points = []
            for i, direction in enumerate(directions):
                points = self._get_points_in_direction(neighbors[i], direction)
                directional_points.append(points)

            for points in directional_points:
                if points:
                    yield points

            # Step 4: Process regions between directions
            sectors = self._partition_remaining_points(center, neighbors)

            for sector_points in sectors:
                yield sector_points

    def process_spiral(self, n_processes=4, seed_sequence:SeedSequence=None) -> None:
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

        if n_processes <= 0:
            raise ValueError("n_processes must be positive")

        if n_processes > 6:
            raise ValueError("n_processes must be <= 6")

        if seed_sequence is None:
            seed_sequence = SeedSequence(0)

        seed_iterator = iter(seed_sequence.spawn(14))

        center = self._find_center_point()

        # Step 1: Process center with center_func (or process_func)
        if self.first_point_func is None:
            self.process_func([center], self.shared_data, 0, next(seed_iterator))
        else:
            self.first_point_func(center, self.shared_data, 0, next(seed_iterator))

        # Step 2: Process neighbors sequentially
        neighbors = self._get_hex_neighbors(center)
        self.process_func(neighbors, self.shared_data, 0, next(seed_iterator))

        # Step 3: Process 6 directions in parallel with joblib
        directions = [self._get_direction_vector(center, n) for n in neighbors]

        parallel = Parallel(n_jobs=n_processes, backend="loky", timeout=None, verbose=0)
        parallel(
            delayed(_wrap_generator_func)(
                self.process_func,
                self._get_points_in_direction,
                (neighbors[i], direction),
                self.shared_data,
                i,  # job_id
                next(seed_iterator)
            )
            for i, direction in enumerate(directions)
        )

        # Step 4: Process 6 sectors in parallel with joblib
        parallel(
            delayed(_wrap_generator_func)(
                self.process_func,
                self._get_sector_between,
                (center, neighbor, neighbors[(i + 1) % len(neighbors)]),
                self.shared_data,
                i,  # job_id
                next(seed_iterator)
            )
            for i, neighbor in enumerate(neighbors)
        )

    def iterate_row_major(self, mask: np.ndarray | None = None) -> Iterable[Vec2_int]:
        """
        :param mask: Mask to filter points of interest.
            Only points whose value is **zero** on the mask **are** processed.
        :return: yields individual points (x, y)
        """
        has_mask = mask is not None
        if has_mask:
            if mask.shape[0] <= self.max_y - self.min_y:
                raise ValueError("mask must not be smaller than max_y - min_y")
            if mask.shape[1] <= self.max_x - self.min_x:
                raise ValueError("mask must not be smaller than max_x - min_x")

        _, _sy = self._snap_to_grid(0, self.min_y)
        _sx, _ = self._snap_to_grid(self.min_x, 0)

        for y in range(int(_sy), int(self.max_y + 1), self.spacing):
            for x in range(int(_sx), int(self.max_x + 1), self.spacing):
                if (y // self.spacing) % 2 != 0:  # offset every odd row to create a hexagonal pattern
                    x += self.spacing // 2

                x, y = self._snap_to_grid(x, y)

                if not self._is_valid_point((x, y)):
                    continue

                if has_mask and mask[y, x] > 0:
                    continue    # skip if non zero

                yield x, y
