# Arguments Explained

A reference to the key `bmquilting` synthesis parameters and how they affect output quality.

---

## 1. Common Arguments

These core concepts apply to both square and circular patching implementations, though units and behaviour differ.

### Block Size / Diameter
The fundamental unit of synthesis. It defines the size of the patches extracted from the source textures.
- **In Square Patching:** Defined as `block_size` in pixels.
- **In Circular Patching:** Defined by the `diameter`, which must be an odd integer.
- **Latent Space Note:** For processing latent images, dimensions should be adjusted according to the latent space; e.g. dividing by 8 for Stable Diffusion.

**Impact:**
- **Larger blocks:** Faster generation (fewer patches needed) and better preservation of large-scale structures. However, they require larger source textures to provide sufficient "addressable area" for variety.
- **Smaller blocks:** Better for fine-grained textures but may struggle with large patterns, leading to "broken" continuity or repetitive artefacts.

### Overlap
The region where adjacent patches are blended together.
- **Square Patching:** Defined as `overlap` in pixels.
- **Circular Patching:** Defined by the `overlap_ratio` (a percentage of the radius, from 0.0 to 1.0). The resulting `overlap_radius` determines the width of the blending ring.

**Impact:**
A larger overlap provides more data for the matching algorithm to find a seamless fit and more space for blending. However, it effectively reduces the "new" area each patch adds, requiring more patches to cover the same output size.

### Tolerance
The strictness of patch matching.
- Patch candidates are those with error <= `(1.0 + tolerance) * global_min_error`.
- **Low Tolerance (<0.1):** Selects only top matches. Usually more seamless but may cause excessive repetition, repeat source texture as-is, or produce discontinuities if "guided" towards source boundaries.
- **High Tolerance (>0.3):** Accepts a broader range of matches. Reduces repetition but can make seams visible if matches are poor.

> [!Note] 
> Because this is relative to `global_min_error`, tolerance behaviour depends heavily on the source texture content (contrast, pattern complexity, noise). Example ranges are a guideline, not absolute.

---

## 2. Patch Layout

### Square Patching
In the square implementation, the grid is Cartesian. The `block_size` and `overlap` are straightforward pixel values. Each new block is placed at an offset of `block_size - overlap` from the previous one.

### Circular Patching
Circular patches are placed on a **Hexagonal Lattice**, which provides a more organic distribution.
- **Radius-Based:** The geometry is driven by the `radius`.
- **Spacing Factor:** A parameter unique to circular patching that defines the distance between patch centres relative to the radius.

> [!WARNING]
> A **spacing factor** of `1.12` or lower may result in overlaps that affect the determinism of **parallel** algorithms.

---

## 3. Running with Feathering (No Seams)

By default, the algorithms utilise complex seam-finding (such as A*) to navigate through the overlap and find the path of least resistance. However, a simpler "feathering" approach can be selected.

### What is Feathering?
Instead of finding a specific path (a "seam"), feathering applies a smooth gradient mask (vignette) to the overlap area, blending the patches. It is faster and performs well for highly stochastic textures (such as sand or noise) where sharp transitions are less noticeable.

For code examples on how to run synthesis with feathering, see the [Quick Start Guide](quick_start.md).
