# Proxy Synthesis (Guided Generation)

Proxy synthesis methods, or **Guided Generation**, utilise a set of "proxy" textures for the patch-matching process, while the final output is reconstructed using the original "source" textures.

---

## 1. Why Use Proxy Synthesis?

Guided generation is particularly useful in scenarios where raw source textures contain noise, artefacts, or fine details that might confuse the patch-matching algorithm.

**Potential use cases:**
- **Faster Synthesis**: Proxy textures are faster to process if they are scaled down or have fewer channels.
- **Noise Reduction**: Matching patches on a de-noised or blurred proxy to avoid stitching based on transient noise patterns.
- **Structural Guidance**: Using a simplified version of a texture to ensure global structure is maintained while keeping original details in the final result.
- **Custom Error Weighting**: Proxy textures can be mathematically transformed to prioritise certain features during matching. For example, if images are processed in the **CIELAB** colour space, the L (Lightness) component can be re-scaled in the proxy to have more or less impact on patch selection than the chrominance channels.
- **Latent Space Guidance**: It is possible to utilise the image represented in latent space as a proxy in hybrid texture synthesis workflows.

> [!NOTE]
> In the current implementation, a proxy must be the same size as or smaller than the target texture. Since latent representations are typically smaller than the full‑resolution image, they can be used as proxies—but the reverse (using a full image to guide a latent) is not supported.

---

## 2. Available Methods

The project provides the following guided variants:

- `bmquilting.square.generate_guided`
- `bmquilting.circular.generate_cphl6p_guided`
- `bmquilting.circular.fill_cphl_guided`
- `bmquilting.circular.seamless_vertical_guided`
- `bmquilting.circular.seamless_horizontal_guided`
- `bmquilting.circular.seamless_both_guided`
- `bmquilting.circular.texture_transfer_guided_advanced`

---

## 3. How It Works

The process involves two main steps:

1. **Proxy Synthesis**: The algorithm identifies the best-matching patches by comparing the **overlap areas of the proxy textures**. It generates a "synthesis map" (a record of which patch from which texture is placed where).
2. **Reconstruction**: The algorithm utilises the synthesis map from Step 1 but extracts the corresponding patches from the **source textures** to build the final high-quality output.

---

## 4. Code Example (Square)

This example demonstrates how to use `generate_guided` to synthesise a noisy texture by using a median-blurred version as a proxy.

```python
import cv2
import numpy as np
from bmquilting.square import generate_guided, SquarePatchingConfig

# 1. Prepare textures
source_img = cv2.imread("noisy_texture.png")
# Create a proxy by blurring the source (to ignore noise during matching)
proxy_img = cv2.medianBlur(source_img, 5)

# 2. Setup configuration
config = SquarePatchingConfig.with_seams(block_size=64, overlap=16, tolerance=0.1)
rng = np.random.default_rng(42)

# 3. Generate with guidance
# Note: The order of textures in proxy_textures MUST match source_textures
out_tex, out_seams, out_proxy = generate_guided(
    proxy_textures=[proxy_img],
    source_textures=[source_img],
    patching_config=config,
    out_h=512,
    out_w=512,
    rng=rng,
    uicd=None
)

# out_tex: The final high-quality texture (source patches, proxy logic)
# out_proxy: The result of synthesising just the proxies
# out_seams: The seams map
cv2.imwrite("result.png", out_tex)
```

---

## 5. Scaling and Multi-resolution Guidance (Circular Only)

For **Circular Patching** methods (`generate_cphl6p_guided`, `fill_cphl_guided`, and seamless variants), proxy textures that are smaller than the source textures can be used. This is useful for processing very large textures or integrating with latent-space representations.

### How it works
The algorithm detects the scale factor ($S = \text{Source Size} / \text{Proxy Size}$) and performs the matching process at the lower resolution. The results are then scaled up by $S$ to reconstruct the final output using the full-resolution source textures.

> [!IMPORTANT]
> The scale factor must be an integer.
> 
> A 512x512 texture works with a 256x256 proxy ($S=2$), but not with a 300x300 proxy ($S≈1.7$). 

### Auto-adjustment and Alignment
To ensure that the centres of patches and the hexagonal grid align precisely between scales without "drifting", the algorithm may automatically adjust parameters:
1.  **Radius Adjustment**: The source `radius` is rounded to the nearest integer multiple of the scale factor $S$.
2.  **Spacing Adjustment**: The `spacing_factor` is slightly adjusted so that the distance between patches in the source grid is exactly $S$ times the distance in the proxy grid.

> [!Note]
> These adjustments are logged at the `INFO` level.
>
> To avoid them, ensure the source `radius` is divisible by the scale factor and output dimensions are multiples of the scale.

---

## 6. Important Considerations

- **List Matching**: The `proxy_textures` and `source_textures` lists must have the same length, and the order of textures must correspond exactly.
- **Dimensions**: 
    - **Square Methods**: Proxies and sources **must** have identical dimensions.
    - **Circular Methods**: Proxies can be smaller, but the scale factor must be a **uniform integer** (e.g. exactly 2x or 4x smaller in both width and height).
- **Performance**: Guided generation involves an additional reconstruction pass. However, using a smaller proxy in circular patching can significantly accelerate the matching phase for high-resolution outputs.
