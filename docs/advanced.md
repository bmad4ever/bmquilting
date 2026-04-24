# Advanced Configuration

For users needing more control over the synthesis process, `bmquilting` provides `advanced` class methods for both `SquarePatchingConfig` and `CircularPatchingConfig`; these allow to more granularly configure the patch searching, seam computations, and blending behaviors. 


## 1. Shared Advanced Options

This section details advanced options common in both (square and circular) procedures. 

### Adjust Seam Computation  

`custom_error_func`: Replace the default `avg_squared_diff`, used to compute the overlapping area errors, with a custom function of your own. This function will only affect seam computation.

> [!NOTE]
> If you are considering using a custom error function, the [guided algorithm variants](proxy_synth.md) might be of relevance.


### Blend Configuration (`BlendConfig`)

Both square and circular configs use `BlendConfig` to control how seams are softened. 

To better understand how these parameters interact, see the [Seam Blending Diagram](seam_blending_diagram.md). More specialized settings are documented within the source code docstrings at [seams_blur.py](../bmquilting/_internal/seams_blur.py).

#### **Key Parameters**
- `min_blur_diameter` & `max_blur_diameter`: Bounds for the adaptive blur applied to the seam.
- `sobel_kernel_size`: Size of the kernel used for gradient computation. 
- `use_vignette`: Enables a smooth fade towards the patch edges. It can be used both with and without seams.
- `use_blur_radii_limiter`: Limits the blur radius with respect to its proximity to the edges of the overlapping area to prevent artifacts.

#### **Blending Sensitivity and Shape**

You can customize the transition between images by modifying two core functions: `blur_size_func`, which determines the **intensity** of the blur based on gradient differences, and `blur_shape_func`, which defines the **curve** of the spatial transition.

By default, these utilize `LogScalingFunc` and `TwoNTS` (Two-Normalized Tunable Sigmoid), respectively.

<br>

**`blur_size_func`**

This function maps gradient differences to a blur diameter, acting as a sensivity scale.

* **Input:** Values in the range $[0, 1]$, representing the normalized difference between two gradients for the given kernel size (where $1$ is the maximum possible difference).
* **Output:** Values in the range $[0, 1]$.
* **Mapping:** The output is interpolated between two user-defined parameters:
    * **0** maps to `min_blur_diameter`.
    * **1** maps to `max_blur_diameter`.

<br>

**`blur_shape_func`**

Defines the blending weight across the seam, shaping the transition between the source and patched areas.

* **Input:** signed distance from the seam relative to the blur size, $u ∈ (-∞, +∞)$
    * $u < -1$ → outside a blur affected area and outside the patched region (inside source or empty area)
    * $u < 0$ → blur affected and outside the patched region (inside source or empty area)
    * $u > 0$ → blur affected and inside the patched region
    * $u > 1$ → outside a blur affected area and within the patched region
* **Output:** Blending coefficients, $α ∈ [0, 1]$.
    * $α = 0$ → 100% patched pixel (0% source)
    * $α = 1$ → 100% source pixel (0% patched)

---

## 2. Square Patching Only Advanced Options

The following are the options available solely for `SquarePatchingConfig.advanced`. 

### Options for Patch Searching

`match_template_method` The method used by match template.
- `cv2.TM_SQDIFF` (Default): Squared difference matching.
- `cv2.TM_CCOEFF_NORMED`: Correlation coefficient matching.

`vignette_on_match_template`: If `True`, applies an adjusted blending vignette as a weight mask during the patch search. 
The image below illustrates its relation to the blending vignette. 

<div style="text-align: center;"><img  alt="Outer Corners Weighted Match Template" src="imgs/VignetteOnMatchTemplate.svg"></div>
<br>

### Options for Seam Computation

`seam_algorithm` defines the algorithm used for seam computation.
- `SeamsAlgorithm.ASTAR` (Default): Uses A* to find the optimal seam. It can backtrack and can handle complex overlaps.
- `SeamsAlgorithm.MIN_CUT`: A strict one‑direction seam strategy that forbids sideways shifts or backward steps.
- `SeamsAlgorithm.NONE`: Bypasses seam computation entirely (useful when only feathering is desired).

### Additional Blending Config Option

When `use_blur_radii_guess_pathfind_limiter` is enabled (`True` by default), an experimental heuristic constrains the seam search region on a per-patch basis. Patches with higher error values are assigned a narrower search area. The goal is to reduce blur artifacts without relying on `use_blur_radii_limiter`.

Unlike the pathfinding limiter, `use_blur_radii_limiter` does not affect the seam search space. Instead, it limits the effective blur radii based on proximity to the patch edges. In practice, this can result in blur not being applied in regions where it would otherwise be appropriate, simply because those regions fall too close to the patch boundaries.

Be aware that restricting seam search based on error can be counterproductive if it forces the seam through portions of the overlap that may not be visually coherent.

---

## 3. Circular Patching Only Advanced Options

The following are the options available solely for `CircularPatchingConfig.advanced`.

### A* Variants
- `AstarVariant.DEFAULT`: Standard A*.
- `AstarVariant.ORTHO_Y`: A* using pyastar2d's experimental `ORTHOGONAL_Y` heuristic — "moves by y first, then half way by x".
- `AstarVariant.NONE`: No seams computed (useful when only feathering is desired).

### Specialized Matching

`outer_corners_weighted_template_matching`: If `True`, weights the template matching to prioritize the outer corners of the annulus.

To understand why this feature exists, we must first look at the specific mechanics of this circular quilting implementation.

**The current implementation uses the outer corners of the partial annulus as fixed endpoints in order to maximize the used non-overlapping area**.

> Using a "free corridor" in order to use arbitrary endpoints (similar to the strategy used with square patches) is a viable alternative, but it is not currently supported.

Because the endpoints are fixed at the corners, the seam is restricted in its movement. If there is a high degree of error (mismatch) near these endpoints, the algorithm might be forced to route the seam through the "worst possible place".

To minimize this worst-case scenario, the area near the endpoints can be weighted more heavily during template matching. This tries to ensure that the selected patch provides a "clean fit" where it matters most, allowing for a smoother transition at the critical junction points.  

The below diagram shows how the template matching mask is built and how it relates this potential worst-case scenario.

<img style="width:100%;" alt="Outer Corners Weighted Match Template" src="imgs/OuterCornersWeightedMatchTemplate.svg">

---


## Example: Advanced Configs

### Square Advanced
```python
from bmquilting.square import SquarePatchingConfig, SeamsAlgorithm, SquarePatchingBlendConfig
import cv2

# Create a custom blend config
blend_cfg = SquarePatchingBlendConfig(
    min_blur_diameter=3,
    max_blur_diameter=15,
    use_vignette=True
)

# Initialize with advanced options
config = SquarePatchingConfig.advanced(
    block_size=64,
    overlap=16,
    tolerance=0.05,
    blend_config=blend_cfg,
    seam_algorithm=SeamsAlgorithm.MIN_CUT,
    match_template_method=cv2.TM_SQDIFF
)

```

### Circular Advanced
```python
from bmquilting.circular import CircularPatchingConfig, AstarVariant
from bmquilting import BlendConfig

blend_cfg = BlendConfig.auto_blend_config_2(block_size=65, overlap=32, use_vignette=False)

config = CircularPatchingConfig.advanced(
    diameter=65,
    overlap_ratio=0.5,
    tolerance=0.1,
    spacing_factor=1.12,
    blend_config=blend_cfg,
    a_star_variant=AstarVariant.ORTHO_Y
)
```
