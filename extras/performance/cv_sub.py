import numpy as np
import cv2
import timeit

# 4MP Float32 buffer
shape = (2048, 2048)
n_iter = 1000

# Setup data
mask_np = np.random.rand(*shape).astype(np.float32)
mask_cv = mask_np.copy()

def test_np_sub():
    global mask_np
    np.subtract(1.0, mask_np, out=mask_np)

def test_cv_sub():
    global mask_cv
    cv2.subtract(mask_cv, 1.0, dst=mask_cv)


# Warm up the CPU
test_np_sub()
test_cv_sub()

# Benchmark
t_np = timeit.timeit(test_np_sub, number=n_iter) / n_iter * 1000
t_cv = timeit.timeit(test_cv_sub, number=n_iter) / n_iter * 1000

print(f"Results for {shape} float32 array:")
print(f"  np.subtract(1.0, x): {t_np:.6f} ms")
print(f"  cv2.subtract(1.0, x): {t_cv:.6f} ms")
print(f"  Speedup:              {t_np / t_cv:.2f}x")