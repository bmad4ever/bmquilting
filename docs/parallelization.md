# Parallelisation

The main strategy employed for parallelisation is a radially structured generation approach, starting from the centre, which aims to divide the generation into independent regions.

This optimisation was not implemented for all methods that could benefit from it. The methods that utilise it are mentioned in this section.

## When to use Parallelisation

Unfortunately, relying on multiple processes doesn't always guarantee a speed-up; it can, sometimes, even make the synthesis slower.

Prefer to rely on parallelisation when dealing with high resolution images or large generations for which the multi-process overhead is negligible.


## For texture generation using Square Patches

```python
from bmquilting.square import generate_texture_parallel 
```

![Parallel Generation Schema](imgs/square_patch_generation_schema.jpg)

The generation process, illustrated in the image above, is composed of the following four steps:

1. **Centre:** The centre of the texture is patched using the standard, non-parallel algorithm. This is performed to ensure there are no areas common to the processes running in subsequent steps.
2. **Stripes:** Two horizontal and two vertical stripes are generated in parallel (four processes), starting from the centre patched area.
3. **Quadrants:** Each of the four quadrants is filled independently, in a separate process.
4. **Sub-Parallelism:** If the `nps` (Number of Parallel Stripes) is greater than 1, multiple stripes are computed in parallel within each quadrant. Because a stripe requires the previous one to be at least one block ahead, coordination between processes causes overhead; therefore, it is recommended to utilise `nps` higher than 1 only for larger generations.

> [!NOTE]
> A minimum `nps` of 1 always starts four processes, one per quadrant.


## For Texture generation using Circular Patches

```python
from bmquilting.circular import generate_cphl6p, generate_cphl6p_guided
```

![Spiral Parallel Generation Schema](imgs/circ_patch_spiral_generation_schema.jpg)

The generation process, illustrated in the image above, is composed of the following five steps:

1. **Centre:** A random patch is selected for the centre of the texture.
2. **Immediate Neighbours:** The six immediate neighbours around the centre are patched sequentially.
3. **Directions:** Six directional batches (rays from the centre) are generated in parallel (up to six processes).
4. **Sectors:** Six sector batches (the areas between the rays) are filled in parallel (up to six processes).
5. **(NOT IMPLEMENTED)** Similarly to the square patches variant, the hexagonal "stripes" could be processed in parallel.

> [!NOTE]
> The `n_processes` parameter indicates the number of parallel processes to be run. The maximum value for `n_processes` is six, corresponding to the maximum number of directions and sectors that can be computed simultaneously.


## Shared Memory and Efficiency

To avoid the overhead of copying large texture data and the resulting synthesis maps between processes, the project utilises shared memory via `multiprocessing.shared_memory.SharedMemory` (for NumPy arrays) and a custom `SharedTextureList` (for the source textures).

- **`SharedTextureList`**: Stores the source textures in a temporary file and memory-maps them across all worker processes. This ensures that all processes can access the same texture data without additional memory overhead.
- **Shared NumPy Arrays**: The output texture and seams map are also stored in shared memory, allowing multiple processes to write to different regions of the same array simultaneously.
