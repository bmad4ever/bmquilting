# bmquilting

`bmquilting` is a Python library for texture synthesis based on the image quilting algorithms. It allows to patch or extend textures from small source samples by stitching patches together.


## Features

*   **Dual Grid Support:** Choose between Square (Cartesian) and Circular (Hexagonal) patching strategies.
*   **Adaptive Seam Blending:** Dynamically adjusts blur intensity based on local gradient differences to hide transitions.
*   **Proxy Synthesis (Guided):** Match patches using simplified "proxies" (blurred, downscaled, etc.) while reconstructing with full-resolution detail.
*   **Seamless Tiling:** Tools to make a texture tileable.
*   **Parallel Generation:** Multi-process implementation for faster synthesis.
*   **Block Size Heuristics:** (Experimental) Automatically estimate an adequate patch size using FFT or SIFT descriptors distribution.


### Synthesis Example

| Input | Output (Texture) | Output (Seams) |
| :---: | :---: | :---: |
| ![Input](docs/imgs/in_circ_tex_184x124.png) | ![Output](docs/imgs/out_circ_tex.png) | ![Seams](docs/imgs/out_circ_seams.png) |

## Installation

You can install `bmquilting` directly from the source:

```bash
# Basic installation
pip install .

# Recommended: With Numba acceleration, slightly improves performance
pip install .[fast]
```


## Documentation and Examples

For detailed guides and API references, please see:

*   [**Main Documentation Index**](docs/main.md)
*   [**Quick Start Guide**](docs/quick_start.md) — More examples including circular patching and inpainting.
*   [**Arguments Explained**](docs/args_explained.md) — Deep dive into block size, overlap, and tolerance.
*   [**Advanced Configuration**](docs/advanced.md) — Fine-tuning blending kernels and seam algorithms.

For example scripts regarding synthesis and other utilities in the library see: [extras/demos](extras/demos).
