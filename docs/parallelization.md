# Parallelization 

The main strategy employed for parallelization is the radially structured generation approach, starting from the center, that aims to divide the generation into independant regions.

This optimization was not implemented for all the methods that could benefit from it. The ones that use it, are the ones mentioned in this section.


### For texture generation using Square Patches

```python
from bmquilting.square import generate_texture_parallel 
```

![Parallel Generation Schema](./imgs/square_patch_generation_schema.jpg)

The generation process, broken down in the above image, is comprised of the following 4 steps:

1. **Center:** The center of the texture is patched using the normal, non-parallel, algorithm. This is done so that there are no common areas to the processes running in the following steps.
2. **Stripes:** 2 horizontal and 2 vertical stripes are generated in parallel (4 processes), starting from the center patched area.
3. **Quadrants:** Each of the 4 quadrants is filled independently, in a separate process.
4. **Sub-Parallelism:** If the `nps` (Number of Parallel Stripes) is greater than 1, multiple stripes are computed in parallel within each quadrant. 
Because a stripe requires the previous one to be at least one block ahead, the coordination between processes causes some overhead; thus, it is recommended to only use `nps` higher than 1 for larger generations. 


> **Note:** The minimum `nps` of 1 always starts 4 processes, one per quadrant.


### For Texture generation using Circular Patches

```python
from bmquilting.circular import generate_cphl6p, generate_cphl6p_guided
```

![Spiral Parallel Generation Schema](./imgs/circ_patch_spiral_generation_schema.jpg)

The generation process, broken down in the above image, is comprised of the following 5 steps:

1. **Center:** A random patch is selected for the center of the texture.
2. **Immediate Neighbors:** The 6 immediate neighbors around the center are patched sequentially.
3. **Directions:** 6 directional batches (rays from the center) are generated in parallel (up to 6 processes).
4. **Sectors:** 6 sector batches (the areas between the rays) are filled in parallel (up to 6 processes).
5. **(NOT IMPLEMENTED)** Similarly to the square patches variant, the hexagonal "stripes" could be processed in parallel.

> **Note:** The `n_processes` parameter indicates how many parallel processes should run. The ceiling for `n_processes` is, therefore, 6, corresponding to the maximum number of directions and sectors that can be computed simultaneously.

## Shared Memory and Efficiency

To avoid the overhead of copying large texture data and the resulting synthesis maps between processes, the project utilizes shared memory via `multiprocessing.shared_memory.SharedMemory` (for NumPy arrays) and a custom `SharedTextureList` (for the source textures).

- **`SharedTextureList`**: Stores the source textures in a temporary file and memory-maps them across all worker processes. This ensures that all processes can access the same texture data without additional memory overhead.
- **Shared NumPy Arrays**: The output texture and seams map are also stored in shared memory, allowing multiple processes to write to different regions of the same array simultaneously.
