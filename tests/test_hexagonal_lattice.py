
import os
import sys
import unittest
import math
from typing import Set, Tuple

# Ensure we can import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Gracefully handle potentially missing dependencies in restricted environments
try:
    import numpy as np
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['numpy'] = MagicMock()
    sys.modules['numpy.random.bit_generator'] = MagicMock()

try:
    import joblib
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['joblib'] = MagicMock()

from _internal.hexagonal_lattice import HexagonalLatticeIterator

class TestHexagonalLatticeIterator(unittest.TestCase):
    def test_lattice_coverage_completeness(self):
        """
        Ensures that every grid point within the specified bounds is visited 
        exactly once by the spiral iteration strategy. This verifies that 
        the sector boundaries and angle calculations are precise even at 
        large distances from the center.
        """
        # Spacing of 11 is used because it's an odd prime that highlights 
        # rounding errors when calculating offsets (spacing // 2 vs spacing / 2.0).
        spacing = 11
        min_x, max_x = 0, 2048
        min_y, max_y = 0, 2048
        
        # 1. Determine all expected grid points manually based on the grid definition
        expected_points: Set[Tuple[int, int]] = set()
        for y in range(min_y, max_y + 1, spacing):
            row = y // spacing
            x_offset = (spacing // 2) if (row % 2 != 0) else 0
            for x in range(min_x, max_x + 1, spacing):
                point_x = x + x_offset
                if min_x <= point_x <= max_x:
                    expected_points.add((point_x, y))
        
        iterator = HexagonalLatticeIterator(min_x, max_x, min_y, max_y, spacing)
        
        # 2. Collect all points visited by the iterator
        visited_points: Set[Tuple[int, int]] = set()
        
        # Replicating the spiral logic flow:
        
        # Step 1: Center
        center = iterator._find_center_point()
        self.assertNotIn(center, visited_points, "Center point revisited")
        visited_points.add(center)
        
        # Step 2: Immediate Neighbors
        neighbors = iterator._get_hex_neighbors(center)
        for n in neighbors:
            if iterator._is_valid_point(n):
                self.assertNotIn(n, visited_points, f"Neighbor {n} revisited")
                visited_points.add(n)
                
        # Step 3: Directional Stripes
        directions = [iterator._get_direction_vector(center, n) for n in neighbors]
        for i, direction in enumerate(directions):
            for p in iterator._get_points_in_direction(neighbors[i], direction):
                self.assertNotIn(p, visited_points, f"Point {p} in direction {i} revisited")
                visited_points.add(p)
                
        # Step 4: Sectors (The area where the angle fix was applied)
        for i in range(len(neighbors)):
            n1 = neighbors[i]
            n2 = neighbors[(i + 1) % len(neighbors)]
            for p in iterator._get_sector_between(center, n1, n2):
                self.assertNotIn(p, visited_points, f"Point {p} in sector {i} revisited")
                visited_points.add(p)
                
        # 3. Final Validation
        missing = expected_points - visited_points
        self.assertEqual(len(missing), 0, 
                         f"Missing {len(missing)} points! Lattice coverage is incomplete. Example missing: {list(missing)[:5]}")
        
        extra = visited_points - expected_points
        self.assertEqual(len(extra), 0, 
                         f"Found {len(extra)} extra points! Lattice contains out-of-bounds or duplicate points. Example: {list(extra)[:5]}")

    def test_ideal_angle_precision(self):
        """Tests that the _get_angle method calculates angles based on an ideal hexagonal grid."""
        spacing = 11
        iterator = HexagonalLatticeIterator(0, 100, 0, 100, spacing)
        
        # Reference point at row 0
        center = (0, 0) 
        
        # Point at row 1 (physical offset is spacing // 2 = 5)
        # Ideal offset should be spacing / 2.0 = 5.5
        point = (5, 11) 
        
        # Physical dx = 5, dy = 11
        # Ideal dx = (5 + 1 * 0.5) - (0 + 0 * 0.5) = 5.5
        # Ideal angle = atan2(11, 5.5)
        expected_angle = math.atan2(11, 5.5)
        self.assertAlmostEqual(iterator._get_angle(center, point), expected_angle)

if __name__ == "__main__":
    unittest.main()
