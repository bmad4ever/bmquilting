import numpy as np
import cv2
import timeit

# Buffer Size: 2048 x 2048 (4MP)
#shape = (2048, 2048)
#shape = (256, 256)
shape = (128, 128)
#shape = (64, 64)
n_iter = 2000

# Data Setup
mask_a = np.random.randint(0, 2, size=shape, dtype=np.uint8)
mask_b = np.random.randint(0, 2, size=shape, dtype=np.uint8)
float_a = np.random.rand(*shape).astype(np.float32)
float_b = np.random.rand(*shape).astype(np.float32)

def benchmark(stmt, setup=""):
    return timeit.timeit(stmt, setup=setup, globals=globals(), number=n_iter) / n_iter * 1000

results = []
# --- 1. Inversion (Logical NOT) ---
t_np_inv = benchmark("np.bitwise_xor(1, mask_a, out=mask_a)")
t_cv_inv = benchmark("cv2.bitwise_xor(mask_a, 1, dst=mask_a)")
results.append(("Inversion (0/1)", "NumPy XOR", t_np_inv, "Base"))
results.append(("", "OpenCV XOR", t_cv_inv, f"{t_np_inv/t_cv_inv:.2f}x"))

# --- 2. Mask Intersection (AND) ---
t_np_and = benchmark("np.bitwise_and(mask_a, mask_b, out=mask_a)")
t_cv_and = benchmark("cv2.bitwise_and(mask_a, mask_b, dst=mask_a)")
results.append(("Intersection", "NumPy AND", t_np_and, "Base"))
results.append(("", "OpenCV AND", t_cv_and, f"{t_np_and/t_cv_and:.2f}x"))

# --- 3. Linear Blending (The Heavy Lifter) ---
# NumPy: requires 3 passes over memory (2 mult, 1 add)
# OpenCV: single pass fusion
t_np_blend = benchmark("res = float_a * 0.7 + float_b * 0.3")
t_cv_blend = benchmark("cv2.addWeighted(float_a, 0.7, float_b, 0.3, 0, dst=float_a)")
results.append(("Blending (Float)", "NumPy Manual", t_np_blend, "Base"))
results.append(("", "OpenCV addWeighted", t_cv_blend, f"{t_np_blend/t_cv_blend:.2f}x"))


# --- 4. Clipping ---
def test_cv_clip(data):
    cv2.threshold(data, 1.0, 1.0, cv2.THRESH_TRUNC, dst=data)
    cv2.threshold(data, 0.0, 0.0, cv2.THRESH_TOZERO, dst=data)

t_np_clip = benchmark("np.clip(float_a, 0, 1, out=float_a)")
t_cv_clip = benchmark("test_cv_clip(float_a)")
results.append(("Clipping", "NumPy Clip", t_np_clip, "Base"))
results.append(("", "OpenCV 2 Thresholds", t_cv_clip, f"{t_np_clip/t_cv_clip:.2f}x"))

# --- 5. Max/Union ---
t_np_max = benchmark("np.maximum(float_a, float_b, out=float_a)")
t_cv_max = benchmark("cv2.max(float_a, float_b, dst=float_a)")
results.append(("Maximum", "NumPy Max", t_np_max, "Base"))
results.append(("", "OpenCV Max", t_cv_max, f"{t_np_max/t_cv_max:.2f}x"))

# Final Printout
print(f"{'Operation':<20} | {'Method':<20} | {'Time (ms)':<10} | {'Speedup'}")
print("-" * 75)
for op, method, time, speed in results:
    print(f"{op:<20} | {method:<20} | {time:<10.4f} | {speed}")