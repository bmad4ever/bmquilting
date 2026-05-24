# Texture Transfer

Texture transfer synthesises a new image by stitching patches taken from one or more **source textures** so that the result visually resembles a given **target image**. A classic use-case is turning a portrait into a stencil made entirely of rice, wood-grain, or any other tileable material.

Three public entry-points are provided. 

| Function | When to use |
|---|---|
| `texture_transfer` | Handles curation and most of the configuration details internally. |
| `texture_transfer_advanced` | When you want to curate the texture yourself or have complete control over the synthesis schedule. |
| `texture_transfer_guided_advanced` | When you want patch *selection* guided by a proxy texture that differs from the source. See [`proxy_synth.md`](proxy_synth.md) for the full rationale and use-cases. |

All three functions are interruptible from the UI and return `None` values if the user cancels mid-run.

---

## Texture Curation

Curation is the preprocessing step that converts a raw image into a representation that can be used for **patch similarity scoring**. 

`curate_for_tex_transfer` is called automatically by `texture_transfer` on both the source textures and the target. When using the `_advanced` variants, you can call it yourself or replace it with your own custom curation procedure.

Crucially, **the curation you apply defines what "similar patches" means** — it is the matching criterion. By changing how the image is converted to a single channel, you can make the algorithm match patches based on luminance, a specific colour channel, or any other scalar feature you can derive from the image.

**The curated textures and curated target must always be produced with the same method and the same pre-processing**, otherwise the similarity scores will be meaningless. When calling `texture_transfer_advanced` or `texture_transfer_guided_advanced`, it is your responsibility to ensure this.

### Processing pipeline

`curate_for_tex_transfer` applies the following operations:

1. **Channel reduction**: converts the input to a single-channel 2D image using `cvt_code` (see [Choosing a matching criterion](#choosing-a-matching-criterion) below). If the input is already single-channel, this step is skipped.
2. **Gaussian blur**: 5×5 kernel, σ = 1. Suppresses high-frequency noise so that patch comparison is driven by structure rather than potential noise.
3. **UINT8 Normalisation**: normalise the values to the interval `[0, 255]`; this is done with respect to the minimum and maximum values found in the data, not the format bounds. 
4. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation): clip limit 2.0, tile grid size `(H/10, W/10)`. Boosts local contrast while preventing over-amplification, making patch differences more discriminable across varying illumination conditions.
5. **Global histogram equalisation**: spreads the intensity distribution evenly, normalising global brightness differences between texture and target so that their similarity scores are directly comparable.
7. **Normalisation**: maps the values to the interval `[0, 1]`.
8. **Reshape**: output is always `H×W×1` (`ndim = 3`) regardless of the input shape.

The mask, when provided, is applied after steps 2, 4, and 6 to zero out regions of no interest at each stage.

> [!IMPORTANT]
> **Mask values must be `0` and `1`, not `0` and `255`.** Using 255s will incorrectly scale pixel values at each masking step and produce wrong results.

### Choosing a matching criterion

The `cvt_code` argument is the main lever for controlling the matching criterion:

- **Default (`cv2.COLOR_BGR2GRAY`)** – Converts BGR to perceived luminance using the standard Rec. 601 formula. Best for matching the light/dark structure of the target.
- **Other grayscale conversion codes** – For example, `cv2.COLOR_RGB2GRAY` if your input is RGB. Any code that produces a single channel is acceptable.
- **`None`** – Computes the per‑pixel mean across channels. Similar to grayscale but without perceptual weighting.

> [!IMPORTANT] 
> **Avoid circular colour spaces (e.g. HSV hue):** patch similarity is computed with `matchTemplate`, which measures linear distance. The hue channel in HSV is circular (0° and 360° are the same colour but numerically far apart), so `matchTemplate` will produce incorrect distances near the wrap-around point. Stick to colour spaces where all channels vary linearly, such as CIELAB a\* and b\*.

> [!TIP] 
> **Matching by chrominance (CIELAB a\*, b\*):** properly representing chrominance requires keeping both the a\* and b\* channels together — collapsing them to a single channel loses colour direction information. `curate_for_tex_transfer` always reduces to one channel and is therefore not suitable for this case. You will need to implement a custom curation function that normalises and enhances the a\* and b\* channels while preserving both, and pass the result (shape `H×W×2`) directly to `texture_transfer_advanced` or `texture_transfer_guided_advanced`. Those functions do accept multi-channel curated inputs.

---

## Iteration Scheduling

All three transfer functions ultimately run the quilting algorithm as a sequence of passes, each defined by a patch geometry (`CircularPatchingConfig`) and a blending weight (`alpha`). Together these form the **iteration schedule**.

### The alpha parameter

Alpha controls the trade-off between two patch selection criteria at each pass:

- **α = 0** : select patches purely by how well they blend with already-placed neighbours (overlap coherence). The target image is ignored.
- **α = 1** : select patches purely by how closely they resemble the target at that location. Existing neighbours are ignored.
- **0 < α < 1** : a weighted combination of both.

### Recommended schedule shape

A typical schedule starts with large patches and high alpha (strong target guidance) and progressively decreases both as the transfer is refined, producing a result that matches the target globally while remaining locally coherent. Notice that doing the reverse, i.e., patching an area using bigger patches than a prior iteration, is probably counter-productive, as the bigger patches get rid of the prior more granular synthesis; nevertheless, you are free to experiment with the scheduling: it is not constrained to monotonic non-increasing values of alpha or patch diameters.

### How each function handles scheduling

The three functions differ in how much of the schedule is computed automatically.

#### `texture_transfer` — automatic

The schedule is derived from `patching_config`, `alphas`, and `last_diameter` via an internal helper:

- **`alphas`** defaults to `[0.75, 0.5, 0.25]`. The number of elements determines the number of passes.
- **`last_diameter`** defaults to `round(diameter / 3.6) | 1` (roughly 28 % of the starting diameter, rounded up to the nearest odd integer).
- Patch diameters are **linearly interpolated** from `patching_config.diameter` (the starting diameter) down to `last_diameter` across the passes.
- `overlap_ratio` is **overridden to `0.5`** at every pass regardless of the value in `patching_config`.
- All other config fields (tolerance, spacing, feathering, etc.) are preserved across passes.

#### `texture_transfer_advanced` — manual

You supply the full `config_alpha_pairs` list directly: a list of `(CircularPatchingConfig, float)` tuples, one per pass. There is no automatic interpolation.
This gives complete control over the schedule.

#### `texture_transfer_guided_advanced` — manual, proxy-scaled

Same as `texture_transfer_advanced`, but with one important distinction: **configs must be specified in source-resolution units**, not proxy-resolution units. The function scales them down to the proxy resolution internally. Expressing configs at source resolution keeps the schedule intuitive and consistent regardless of the proxy scale factor.

---
