import numpy as np
import time

from bmquilting.misc.functions import (
    _nts_no_deadzone_numpy,
    _nts_with_deadzone_numpy,
    _nts_no_deadzone_numba_flat,
    _nts_with_deadzone_numba_flat,
    TwoNTS,
    NUMBA_AVAILABLE
)


# ============================================================
# Helper functions
# ============================================================
NUMBER_OF_RUNS = 20   # run multiple time to avoid outliers and simulate what would be like to actually use in the algo

def _run_multiple_times(func, *args) -> np.ndarray:
    times = []
    for _ in range(NUMBER_OF_RUNS):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # Convert to ms
    return np.array(times)


# --- Main Benchmarking Functions ---

def benchmark_numpy(func, x: np.ndarray, *args: any, ):
    def runner():
        """Wrapper to handle the necessary input copy for the timing loop."""
        x_local = x.copy()
        func(x_local, *args)

    times = _run_multiple_times(runner)

    return np.median(times)


def benchmark_numba(func, x: np.ndarray, *args: any):
    if not NUMBA_AVAILABLE:
        return None

    # --- 1. Warm-up (Crucial for JIT-compiled code!) ---
    x_warmup = x.copy()
    # Assuming Numba functions are designed to work on the flattened array
    func(x_warmup.ravel(), *args)

    # --- 2. Hot Timing ---
    def runner():
        """Wrapper to handle the necessary input copy and raveling for the timing loop."""
        x_local = x.copy()
        # Ensure the input is flattened for the Numba function
        func(x_local.ravel(), *args)

    times = _run_multiple_times(runner)

    return np.median(times)


# ============================================================
# Config
# ============================================================

sizes = [16**2, 32**2, 64**2, 128**2, 256**2, 512**2, 1024**2]
k = -0.5
deadzone = 0.25
beta = 0.5

two_nts = TwoNTS()

print("\n=====================================")
print("   NTS Benchmark (NumPy vs Numba)")
print("=====================================\n")


# ============================================================
# Run benchmarks
# ============================================================

for n in sizes:
    print(f"Array size = {n:,}")

    # Prepare sample input in [-1,1]
    x = np.linspace(-1, 1, n).astype(np.float64)

    # -----------------------------------------------
    # Target: no-deadzone path
    # -----------------------------------------------
    t_np_no = benchmark_numpy(_nts_no_deadzone_numpy, x, k)
    t_nb_no = benchmark_numba(_nts_no_deadzone_numba_flat, x, k)

    print(f"  No-deadzone:")
    print(f"    NumPy : {t_np_no:7.3f} ms")
    if NUMBA_AVAILABLE:
        print(f"    Numba : {t_nb_no:7.3f} ms")
    else:
        print(f"    Numba : unavailable")

    # -----------------------------------------------
    # Target: soft-deadzone path
    # -----------------------------------------------
    t_np_dz = benchmark_numpy(_nts_with_deadzone_numpy, x, k, deadzone, beta)
    t_nb_dz = benchmark_numba(_nts_with_deadzone_numba_flat, x, k, deadzone, beta)

    print(f"  Soft-deadzone:")
    print(f"    NumPy : {t_np_dz:7.3f} ms")
    if NUMBA_AVAILABLE:
        print(f"    Numba : {t_nb_dz:7.3f} ms")
    else:
        print(f"    Numba : unavailable")

    # -----------------------------------------------
    # Target: TwoNTS
    # -----------------------------------------------

    print(f"  TwoNTS:")
    if NUMBA_AVAILABLE:
        t_nts_used = benchmark_numba(two_nts.inplace_func, x)
        print(f"    Numba (caveat: not a single numba func!) : {t_nts_used:7.3f} ms")
    else:
        t_nts_used = benchmark_numpy(two_nts.inplace_func, x)
        print(f"    NumPy : {t_nts_used:7.3f} ms")


    print()
