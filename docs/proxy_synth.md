# Proxy Synthesis (Guided Generation)

Proxy synthesis methods, or **Guided Generation**, use a set of "proxy" textures for the patch-matching process, while the final output is reconstructed using the original "source" textures.

---

## 1. Why use Proxy Synthesis?

Guided generation can be particularly useful in scenarios where the raw source textures contain noise, artifacts, or fine details that might confuse the patch-matching algorithm; its usage is not, however, restricted to such scenarios.

**Potential use cases:**
- **Faster Synthesis**: Proxy textures are faster to process if they are scaled down or have less channels. 
- **Noise Reduction:** Matching patches on a denoised or blurred proxy to avoid stitching based on transient noise patterns.
- **Structural Guidance:** Using a simplified or "blocked" version of a texture to ensure the global structure is maintained while keeping the original details in the final result.
- **Custom Error Weighting:** Proxy textures can be mathematically transformed to prioritize certain features during matching. For example, if images are processed in the **CIELAB** color space, the L (Lightness) component can be re-scaled in the proxy to have more or less impact on patch selection than the chrominance channels.
- **Latent Space Guidance:** It is possible to use the image represented in latent space as a proxy in hybrid texture synthesis workflows.

> [!NOTE]
> Due to the current implementation, a proxy must be the same size or smaller than the target texture. Since latent representations are typically smaller than the full‑resolution image, they can be used as proxies—but the reverse (using a full image to guide a latent) is not supported.

---

## 2. Available Methods

The project provides the following guided variants:

- `bmquilting.square.generate_guided`
- `bmquilting.circular.generate_cphl6p_guided`
- `bmquilting.circular.fill_cphl_guided`
- `bmquilting.circular.seamless_vertical_guided`
- `bmquilting.circular.seamless_horizontal_guided`
- `bmquilting.circular.seamless_both_guided`

---

## 3. How it Works

The process involves two main steps:

1. **Proxy Synthesis:** The algorithm finds the best-matching patches by comparing the **overlap areas of the proxy textures**. It generates a "synthesis map" (a record of which patch from which texture goes where).
2. **Reconstruction:** The algorithm uses the synthesis map from Step 1 but extracts the corresponding patches from the **source textures** to build the final high-quality output.

---

## 4. Code Example (Square)

This example demonstrates how to use `generate_guided` to synthesize a noisy texture by using a median-blurred version as a proxy.

```python
import cv2
import numpy as np
from bmquilting.square import generate_guided, SquarePatchingConfig

# 1. Prepare your textures
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
# out_proxy: The result of synthesizing just the proxies
# out_seams: The seams map
cv2.imwrite("result.png", out_tex)
```

---

## 5. Scaling and Multi-resolution Guidance (Circular Only)

For **Circular Patching** methods (`generate_cphl6p_guided`, `fill_cphl_guided`, and seamless variants), you can use proxy textures that are smaller than the source textures. This is useful for processing very large textures or integrating with latent-space representations.

### How it works
The algorithm detects the scale factor ($S = \text{Source Size} / \text{Proxy Size}$) and performs the matching process at the lower resolution. The results are then scaled up by $S$ to reconstruct the final output using the full-resolution source textures.

> [!IMPORTANT]
> The scale factor must be an integer.
> 
> A 512x512 sized texture works with a 256x256 sized proxy ($S=2$), but not with a 300x300 sized ($S≈1.7$). 

### Auto-Adjustment & Alignment
To ensure that the centers of patches and the hexagonal grid align perfectly between scales without "drifting," the algorithm may automatically adjust your parameters:
1.  **Radius Adjustment:** The source `radius` is rounded to the nearest integer multiple of the scale factor $S$.
2.  **Spacing Adjustment:** The `spacing_factor` is slightly adjusted so that the distance between patches in the source grid is exactly $S$ times the distance in the proxy grid.

> [!Note]
> These adjustments are logged at the `INFO` level.
>
> If you want to avoid them, ensure your source `radius` is divisible by your scale factor and your output dimensions are multiples of the scale.

---

## 6. Important Considerations

- **List Matching:** The `proxy_textures` and `source_textures` lists must have the same length, and the order of textures must correspond exactly.
- **Dimensions:** 
    - **Square Methods:** Proxies and sources **must** have identical dimensions.
    - **Circular Methods:** Proxies can be smaller, but the scale factor must be a **uniform integer** (e.g., exactly 2x or 4x smaller in both width and height).
- **Performance:** Guided generation involves an additional reconstruction pass. However, using a smaller proxy in circular patching can significantly speed up the matching phase for high-resolution outputs.
