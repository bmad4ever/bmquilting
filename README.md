# bmquilting

`bmquilting` is a Python library for texture synthesis based on image quilting algorithms. It synthesises new textures from small source samples by intelligently stitching patches together.

## Features

- **Dual Grid and Patch Shape Support:** Square patches on a cartesian grid, or circular patches over a hexagonal lattice.
- **Multiple Stitching Modes:** Seams, blurred seams, feathering, or hybrid stitching solutions.
- **Adaptive Seam Blending:** Dynamically adjusts blur intensity based on local gradient differences to conceal transitions.
- **Proxy Synthesis:** Match patches using simplified proxies (blurred, downscaled, etc.) whilst reconstructing with full-resolution detail.
- **Seamless Tiling:** Built-in functions to make any texture tileable.
- **Hole Filling:** Inpaint textures with missing or masked-out regions.
- **Texture Transfer:** Transfer the texture of one object onto another, or create a textured stencil of a subject.
- **Incomplete References:** Mark invalid sections of a source image; the library will still make partial use of them during synthesis.
- **Parallel Generation:** Multi-process variants of synthesis functions for faster throughput.
- **Block Size Heuristics** *(Experimental)*: Automatically estimate a suitable patch size using FFT or SIFT descriptor distributions.

## Synthesis Examples

<details>
<summary><strong>📸 Click to expand example section</strong></summary>

### Texture Transfer

| Input Target | Input Texture | Output |
| :---: | :---: | :---: |
| ![Target](docs/imgs/samples/gallery_trans_target_0.jpg) | ![Texture](docs/imgs/samples/gallery_trans_tex_0.jpg) | ![Output](docs/imgs/samples/gallery_trans_out_0.jpg) |
| ![Target](docs/imgs/samples/gallery_trans_target_1.jpg) | ![Texture](docs/imgs/samples/gallery_tile_in_1.jpg) | ![Output](docs/imgs/samples/gallery_trans_out_1.jpg) |
| ![Target](docs/imgs/samples/gallery_trans_target_2.jpg) | ![Texture](docs/imgs/samples/gallery_trans_tex_2.jpg) | ![Output](docs/imgs/samples/gallery_trans_out_2.jpg) |

### Hole Filling

| Input | Seams | Output |
| :---: | :---: | :---: |
| ![Input](docs/imgs/samples/gallery_fill_in_0.jpg) | ![Seams](docs/imgs/samples/gallery_fill_seams_0.jpg) | ![Output](docs/imgs/samples/gallery_fill_out_0.jpg) |
| ![Input](docs/imgs/samples/gallery_fill_in_1.jpg) | ![Seams](docs/imgs/samples/gallery_fill_seams_1.jpg) | ![Output](docs/imgs/samples/gallery_fill_out_1.jpg) |
| ![Input](docs/imgs/samples/gallery_fill_in_2.jpg) | ![Seams](docs/imgs/samples/gallery_fill_seams_2.jpg) | ![Output](docs/imgs/samples/gallery_fill_out_2.jpg) |

### Proxy Synthesis

| Input (Noisy) | Proxy (Blurred) | Output |
| :---: | :---: | :---: |
| ![Input](docs/imgs/samples/gallery_proxy_in_0.jpg) | ![Proxy](docs/imgs/samples/gallery_proxy_p_0.jpg) | ![Output](docs/imgs/samples/gallery_proxy_out_0.jpg) |
| ![Input](docs/imgs/samples/gallery_proxy_in_1.jpg) | ![Proxy](docs/imgs/samples/gallery_proxy_p_1.jpg) | ![Output](docs/imgs/samples/gallery_proxy_out_1.jpg) |
| ![Input](docs/imgs/samples/gallery_proxy_in_2.jpg) | ![Proxy](docs/imgs/samples/gallery_proxy_p_2.jpg) | ![Output](docs/imgs/samples/gallery_proxy_out_2.jpg) |

### Texture Generation

| Input | Seams | Output |
| :---: | :---: | :---: |
| ![Input](docs/imgs/samples/gallery_gen_in_0.jpg) | ![Seams](docs/imgs/samples/gallery_gen_seams_0.jpg) | ![Output](docs/imgs/samples/gallery_gen_out_0.jpg) |
| ![Input](docs/imgs/samples/gallery_gen_in_1.jpg) | ![Seams](docs/imgs/samples/gallery_gen_seams_1.jpg) | ![Output](docs/imgs/samples/gallery_gen_out_1.jpg) |
| ![Input](docs/imgs/samples/gallery_gen_in_2.jpg) | ![Seams](docs/imgs/samples/gallery_gen_seams_2.jpg) | ![Output](docs/imgs/samples/gallery_gen_out_2.jpg) |

### Make Texture Tile-able

| Input | Output | Tiled 2x2 |
| :---: | :---: | :---: |
| ![Input](docs/imgs/samples/gallery_tile_in_0.jpg) | ![Output](docs/imgs/samples/gallery_tile_out_0.jpg) | ![Tiled](docs/imgs/samples/gallery_tile_2x2_0.jpg) |
| ![Input](docs/imgs/samples/gallery_tile_in_1.jpg) | ![Output](docs/imgs/samples/gallery_tile_out_1.jpg) | ![Tiled](docs/imgs/samples/gallery_tile_2x2_1.jpg) |
| ![Input](docs/imgs/samples/gallery_tile_in_2.jpg) | ![Output](docs/imgs/samples/gallery_tile_out_2.jpg) | ![Tiled](docs/imgs/samples/gallery_tile_2x2_2.jpg) |

</details>



## Requirements

- Python 3.12 or higher
- See `pyproject.toml` for full dependency list

## Installation

Install directly from source:

```bash
# Basic installation
pip install .

# With Numba acceleration (improves certain internals; overall speedup is typically 1–6%)
pip install .[fast]
```

## Documentation

| Resource | Description |
| :--- | :--- |
| [Main Documentation Index](docs/main.md) | Full API reference and overview |
| [Quick Start Guide](docs/quick_start.md) | Examples including circular patching and inpainting |
| [Arguments Explained](docs/args_explained.md) | Deep dive into block size, overlap, and tolerance |
| [Advanced Configuration](docs/advanced_config.md) | Fine-tuning blending kernels and seam algorithms |
| [Demo Scripts](extras/demos) | Runnable examples for synthesis and utility functions |

---


## Licence

This project is distributed under [MIT](LICENSE) licence.


