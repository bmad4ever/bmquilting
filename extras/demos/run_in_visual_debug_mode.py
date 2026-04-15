# Run as: python -m extras.demos.run_in_visual_debug_mode
"""
Example: run bmquilting with visual debugging enabled.

This script demonstrates how to use `visual_debug.enable_visual_debug()` to
step through intermediate arrays in seams_blur.py, circular_subroutines.py,
and square_subroutines.py.

Set SAVE_FRAMES = True to also write every frame as a PNG into ./debug_viz/.

Uses a single-process circular-patch generator so that the trace is easy to follow step-by-step.
"""

import numpy as np
import cv2

from bmquilting.circular import (
    generate_cphl6p, seamless_both,
    CircularPatchingConfig,
)
from bmquilting._internal.visual_debug import (
    enable_visual_debug,
    disable_visual_debug,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SAVE_FRAMES = False          # Set True to also save frames to ./debug_viz/
SEED = 239487
DIAMETER = 49                # Circular patch diameter (should be odd)
OVERLAP_RATIO = 0.45
TOLERANCE = 0.05
SPACING_FACTOR = 1.05
OUT_H, OUT_W = 256, 256      # Output Size

# ---------------------------------------------------------------------------
# Load a sample image (replace with any texture you like)
# ---------------------------------------------------------------------------
try:
    # Try loading a local image; fallback to synthetic if not found
    SAMPLE_IMG_PATH = "sample_texture.png"
    img = cv2.imread(SAMPLE_IMG_PATH)
    if img is None:
        raise FileNotFoundError
except (FileNotFoundError, cv2.error):
    print(f"[demo] '{SAMPLE_IMG_PATH}' not found – generating a synthetic texture.")
    tile = 32
    rows, cols = 8, 8
    img = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = [255, 180, 140]
            else:
                img[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = [80, 200-c%3*60, r%4*63]
img_float = np.float32(img) / 255.0
cv2.imshow("source", img_float)

# ---------------------------------------------------------------------------
# Setup generation
# ---------------------------------------------------------------------------

patching_config = CircularPatchingConfig.with_seams(
    diameter=DIAMETER,
    overlap_ratio=OVERLAP_RATIO,
    tolerance=TOLERANCE,
    spacing_factor=SPACING_FACTOR,
    blend=True,
)


target = np.zeros((OUT_H, OUT_W, 3), dtype=np.float32)

# ---------------------------------------------------------------------------
# Run in visual-debug mode (single process, sequential)
# ---------------------------------------------------------------------------
print("[demo] Enabling visual debugger ...")
enable_visual_debug(save_frames=SAVE_FRAMES)

try:
    result, seams = generate_cphl6p(
        source_textures=[img_float],
        patching_config=patching_config,
        out_h=OUT_H, out_w=OUT_W,
        seed=SEED,
    )
    print("[demo] Generation complete.")

    if result is not None:
        # Show final result
        result_u8 = (result*255).astype(np.uint8)
        cv2.imshow("Final Result", result_u8)
        cv2.waitKey(0)

finally:
    disable_visual_debug()
    print("[demo] Done.")
