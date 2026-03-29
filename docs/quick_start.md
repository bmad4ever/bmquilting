# Quick Start Guide

This guide will help you get started with `bmquilting` for texture synthesis using both square and circular patching methods.

## 1. Square Patching

Square patching is the classic approach where patches are arranged in a regular Cartesian grid.

### Basic Generation with Seams (A*)

By default, the algorithm uses A* to find optimal seams between patches, minimizing visible discontinuities.

```python
import cv2
import numpy as np
from bmquilting.square import generate_texture, SquarePatchingConfig

# Load your source texture
src = cv2.imread("source.png")

# Configure: 64px blocks, 16px overlap, 0.1 tolerance
config = SquarePatchingConfig.with_seams(block_size=64, overlap=16, tolerance=0.1)
rng = np.random.default_rng(42)

# Generate a 512x512 texture
out_tex, out_seams = generate_texture(
    src_textures=[src],
    patching_config=config,
    out_h=512,
    out_w=512,
    rng=rng,
    uicd=None
)

cv2.imwrite("output_square_seams.png", out_tex)
```

| Input | Output (Texture) | Output (Seams) |
| :---: | :---: | :---: |
| ![Input](https://via.placeholder.com/150?text=Source+Texture) | ![Output](https://via.placeholder.com/300?text=Generated+Texture) | ![Seams](https://via.placeholder.com/300?text=Seams+Map) |

### Generation with Feathering

Feathering uses a smooth gradient blend instead of calculating seams. It is faster and works well for stochastic textures.

```python
from bmquilting.square import generate_texture, SquarePatchingConfig

# Initialize config with feathering
config = SquarePatchingConfig.with_feathering(block_size=64, overlap=16, tolerance=0.1)

# Generate as usual
out_tex, _ = generate_texture([src], config, 512, 512, rng, None)
```

---

## 2. Circular Patching

Circular patches are placed on a **Hexagonal Lattice**, providing a more organic, non-grid-like distribution.

### Basic Generation with Seams

Circular patches are placed on a **Hexagonal Lattice** (CPHL), providing a more organic, non-grid-like distribution.

```python
import cv2
import numpy as np
from bmquilting.circular import generate_cphl6p, CircularPatchingConfig, CircularPatchParams

src = cv2.imread("source.png")

# 1. Define patch parameters: 65px diameter, 25% overlap ratio
params = CircularPatchParams(diameter=65, overlap_ratio=0.25)

# 2. Configure: 0.1 tolerance and 1.12 spacing factor
config = CircularPatchingConfig.with_seams(params, tolerance=0.1, spacing_factor=1.12)
seed = 42

# 3. Generate a 512x512 texture
out_tex, out_seams = generate_cphl6p(
    source_textures=[src],
    out_h=512,
    out_w=512,
    patching_config=config,
    seed=seed
)

cv2.imwrite("output_circular_seams.png", out_tex)
```

| Input | Output (Texture) | Output (Seams) |
| :---: | :---: | :---: |
| ![Input](https://via.placeholder.com/150?text=Source+Texture) | ![Output](https://via.placeholder.com/300?text=Generated+Texture) | ![Seams](https://via.placeholder.com/300?text=Seams+Map) |

### Generation with Feathering

```python
from bmquilting.circular import generate_cphl6p, CircularPatchingConfig, CircularPatchParams

params = CircularPatchParams(diameter=65, overlap_ratio=0.25)
config = CircularPatchingConfig.with_feathering(params, tolerance=0.1, spacing_factor=1.12)
out_tex, _ = generate_cphl6p([src], 512, 512, config, seed)
```

---

## 3. Creating Seamless Textures

If you have an existing image and want to make it tileable (seamless), you can patch its boundaries.

### Seamless Square
```python
from bmquilting.square import seamless_both_multi

# Makes the 'src' image tileable in both X and Y directions
# Uses multiple source textures (if provided in list) for patching
seamless_img, seams_map = seamless_both_multi(src, config, rng)
```

### Seamless Circular
```python
from bmquilting.circular import seamless_both

# Similar for circular patching
# Takes a target image and a list of lookup textures
seamless_img, seams_map = seamless_both(src, [src], config, seed)
```

---

## 4. Proxy Synthesis (Guided Generation)

Proxy synthesis allows matching patches based on a "proxy" (e.g., a blurred or downscaled version) while reconstructing the final output with high-resolution source textures.

```python
from bmquilting.circular import generate_cphl6p_guided

# Create a blurred proxy to ignore fine noise during matching
proxy = cv2.medianBlur(src, 5)

out_tex, out_seams, out_proxy = generate_cphl6p_guided(
    proxy_textures=[proxy],
    source_textures=[src],
    patching_config=config,
    out_h=512,
    out_w=512,
    seed=seed
)
```

For more details on scaling and multi-resolution guidance, see the [Proxy Synthesis Guide](proxy_synth.md).

---

## 5. Progress Tracking and Step Prediction

All main generation functions are decorated with a `step_predictor`, which adds a `.predict_steps()` method to the function. This is useful for initializing progress bars in UI applications.

```python
from bmquilting.circular import generate_cphl6p, CircularPatchParams, CircularPatchingConfig

params = CircularPatchParams(diameter=65, overlap_ratio=0.25)
config = CircularPatchingConfig.with_seams(params, tolerance=0.1, spacing_factor=1.12)

# Predict how many patches will be generated
total_patches = generate_cphl6p.predict_steps(
    patching_config=config, 
    out_h=512, 
    out_w=512
)
print(f"Expected patches: {total_patches}")
```

For more details on integrating with a GUI, see the [UICD Demo](../extras/demos/uicd.py).

---

## Next Steps

- **Parameter Details:** See [Arguments Explained](args_explained.md).
- **Examples:** Check the `extras/demos/` to tinker with the methods via a GUI interface.
