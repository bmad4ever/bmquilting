"""
PyAStar2D Behavior Visual Testing Script.
Check if pyastar2d behaves as expected.
"""

import numpy as np
import pyastar2d
import matplotlib.pyplot as plt
from cv2 import blur


def visualize_path(weights, path, start, end, title="", ax=None):
    """Visualize the weight grid and the path found by pyastar2d"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Show weights as heatmap
    im = ax.imshow(weights, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Cost/Weight')

    # Draw grid
    ax.set_xticks(np.arange(-0.5, weights.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, weights.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Draw path
    if path is not None and len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2, label='Path')
        ax.plot(path_array[:, 1], path_array[:, 0], 'bo', markersize=4)

    # Mark start and end
    ax.plot(start[1], start[0], 'g^', markersize=12, label='Start', markeredgecolor='black')
    ax.plot(end[1], end[0], 'rs', markersize=12, label='End', markeredgecolor='black')

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('Column (j)')
    ax.set_ylabel('Row (i)')

    return ax


def test_basic_path(test_name: str, heur=pyastar2d.Heuristic.DEFAULT):
    """Test 1: Basic path finding"""
    print("\n=== TEST 1: Basic Path Finding ===")
    weights = np.ones((10, 10), dtype=np.float32)
    start = (0, 5)
    end = (9, 5)

    path = pyastar2d.astar_path(weights, start, end, allow_diagonal=True, heuristic_override=heur)
    print(f"Start: {start}, End: {end}")
    print(f"Path length: {len(path)}")
    print(f"Path: {path}")
    print(f"Path cost: {len(path)}")  # uniform weights

    visualize_path(weights, path, start, end, test_name)
    return path


def test_basic_path_default_heur():
    return test_basic_path("Test 1.1: Basic Uniform Grid using default heuristic")


def test_basic_path_ort_y():
    return test_basic_path("Test 1.2: Basic Uniform Grid using orthogonal Y heur", pyastar2d.Heuristic.ORTHOGONAL_Y)


def test_basic_path_diagonal():
    """Test 1: Basic path finding"""
    print("\n=== TEST 1: Basic Path Finding ===")
    weights = np.ones((10, 10), dtype=np.float32)
    start = (0, 0)
    end = (9, 9)

    path = pyastar2d.astar_path(weights, start, end, allow_diagonal=True)
    print(f"Start: {start}, End: {end}")
    print(f"Path length: {len(path)}")
    print(f"Path: {path}")
    print(f"Path cost: {len(path)}")  # uniform weights

    visualize_path(weights, path, start, end, "Test 1.3: Basic Uniform Grid Diagonal Path")
    return path


def test_diagonal_vs_non_diagonal():
    """Test 2: Diagonal movement behavior"""
    print("\n=== TEST 2: Diagonal vs Non-Diagonal ===")
    weights = np.ones((10, 10), dtype=np.float32)
    start = (0, 0)
    end = (9, 9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    path_diag = pyastar2d.astar_path(weights, start, end, allow_diagonal=True)
    visualize_path(weights, path_diag, start, end,
                   f"With Diagonal (length={len(path_diag)})", ax=ax1)

    path_no_diag = pyastar2d.astar_path(weights, start, end, allow_diagonal=False)
    visualize_path(weights, path_no_diag, start, end,
                   f"Without Diagonal (length={len(path_no_diag)})", ax=ax2)

    print(f"Diagonal path length: {len(path_diag)}")
    print(f"Non-diagonal path length: {len(path_no_diag)}")
    print(f"Diagonal path: {path_diag}")


def test_high_cost_barrier():
    """Test 3: How it handles high cost barriers"""
    print("\n=== TEST 3: High Cost Barrier ===")
    weights = np.ones((10, 10), dtype=np.float32)
    # Create a high-cost barrier
    weights[5, 3:7] = 100.0

    start = (0, 5)
    end = (9, 5)

    path = pyastar2d.astar_path(weights, start, end, allow_diagonal=True)
    print(f"Path avoids barrier: {path}")
    print(f"Path length: {len(path)}")

    visualize_path(weights, path, start, end, "Test 3: High Cost Barrier")


def test_narrow_passage():
    """Test 4: Narrow passage behavior"""
    print("\n=== TEST 4: Narrow Passage ===")
    weights = np.ones((10, 10), dtype=np.float32) * 100
    # Create a narrow passage
    weights[5, :] = 1.0
    weights[:, 5] = 1.0

    start = (0, 0)
    end = (9, 9)

    path = pyastar2d.astar_path(weights, start, end, allow_diagonal=True)
    print(f"Path through narrow passage length: {len(path)}")

    visualize_path(weights, path, start, end, "Test 4: Narrow Passage")


def test_texture_seam_scenario(block_size=60, overlap=18, seed=128):
    """Test 5: Simulate texture seam scenario"""
    print("\n=== TEST 5: Texture Seam Simulation ===")

    if seed is not None:
        np.random.seed(seed)

    # Simulate error like in your function
    # Create some pattern that represents texture mismatch
    err = np.random.rand(block_size, overlap).astype(np.float32)
    # Add gradient to simulate texture difference
    err += np.linspace(0, 1, overlap)[np.newaxis, :] / 10

    for _ in range(3):
        blur(err, [3, 3], dst=err)

    # High freq noise
    err += np.random.rand(block_size, overlap).astype(np.float32) / 10

    # Normalize
    err /= np.max(err)

    # Error scaling
    err = err ** 2
    err *= 255
    err += 1
    err *= block_size ** 2

    # Your padding
    err_padded = np.pad(err, ((1, 1), (0, 0)), 'constant', constant_values=(1, 1))

    start = (0, err_padded.shape[1] // 2)
    end = (err_padded.shape[0] - 1, err_padded.shape[1] // 2)

    print(f"Error grid shape: {err_padded.shape}")
    print(f"Error range: [{err_padded.min()}, {err_padded.max()}]")
    print(f"Start: {start}, End: {end}")

    path = pyastar2d.astar_path(err_padded, start, end, allow_diagonal=True)
    print(f"Path length: {len(path)}")
    print(f"Path: {path}")

    visualize_path(err_padded, path, start, end,
                   "Test 5: Texture Seam Scenario")


def test_edge_cases():
    """Test 6: Edge cases and boundary behavior"""
    print("\n=== TEST 6: Edge Cases ===")

    # Test 6a: Start at edge
    print("\n6a: Start/End at edges")
    weights = np.ones((10, 10), dtype=np.float32)
    start = (0, 5)
    end = (9, 5)
    path = pyastar2d.astar_path(weights, start, end, allow_diagonal=True)
    print(f"Vertical path length: {len(path)}")

    # Test 6b: Very small grid
    print("\n6b: Small grid (3x3)")
    weights_small = np.ones((3, 3), dtype=np.float32)
    path_small = pyastar2d.astar_path(weights_small, (0, 0), (2, 2), allow_diagonal=True)
    print(f"Small grid path: {path_small}")

    # Test 6c: Single column (like your overlap scenario)
    print("\n6c: Single column width")
    weights_col = np.ones((10, 1), dtype=np.float32)
    try:
        path_col = pyastar2d.astar_path(weights_col, (0, 0), (9, 0), allow_diagonal=True)
        print(f"Single column path length: {len(path_col)}")
        print(f"Path: {path_col}")
    except Exception as e:
        print(f"Error with single column: {e}")


def test_cost_calculation():
    """Test 7: How costs are actually calculated"""
    print("\n=== TEST 7: Cost Calculation ===")

    # Create a grid where we can calculate expected costs
    weights = np.ones((5, 5), dtype=np.float32)
    weights[1:4, 1:4] = 10.0  # High cost center

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Opposite corners case
    start1, end1 = (0, 0), (4, 4)
    path1 = pyastar2d.astar_path(weights, start1, end1, allow_diagonal=True)
    total_cost1 = sum(weights[i, j] for i, j in path1)
    visualize_path(weights, path1, start1, end1,
                   f"Test 7 Case 1 (cost={total_cost1:.1f})", ax=axes[0])

    # Forward, on the opposite side case
    start2, end2 = (0, 2), (4, 2)
    path2 = pyastar2d.astar_path(weights, start2, end2, allow_diagonal=True)
    total_cost2 = sum(weights[i, j] for i, j in path2)
    visualize_path(weights, path2, start2, end2,
                   f"Case 2 (cost={total_cost2:.1f})", ax=axes[1])

    print(f"Path 1 cost: {total_cost1}, length: {len(path1)}")
    print(f"Path 2 cost: {total_cost2}, length: {len(path2)}")


def test_float_precision():
    """Test 8: Float precision and very small differences"""
    print("\n=== TEST 8: Float Precision ===")

    weights = np.ones((10, 10), dtype=np.float32)
    # Create very small cost differences
    weights[5, :9] = 1.00001

    start = (0, 5)
    end = (9, 5)

    # mind that diagonals cost the same as a straight path, so using default heuristic it should contour the obstacle
    path = pyastar2d.astar_path(weights, start, end, allow_diagonal=True)
    print(f"Path with tiny cost differences: {path}")
    print("Does it prefer the slightly cheaper path?")

    visualize_path(weights, path, start, end, "Test 8: Small Cost Differences")


if __name__ == "__main__":
    """Run all tests"""
    print("="*60)
    print("PyAStar2D Behavior Testing")
    print("="*60)

    test_basic_path_default_heur()
    test_basic_path_ort_y()
    test_basic_path_diagonal()
    test_diagonal_vs_non_diagonal()
    test_high_cost_barrier()
    test_narrow_passage()
    test_edge_cases()
    test_cost_calculation()
    test_float_precision()
    test_texture_seam_scenario()

    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
