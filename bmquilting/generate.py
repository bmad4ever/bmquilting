from .synthesis_subroutines import (
    get_find_patch_to_the_right_method, get_find_patch_below_method, get_find_patch_both_method,
    get_min_cut_patch_horizontal_method, get_min_cut_patch_vertical_method, get_min_cut_patch_both_method,
    update_seams_map_view)
from .types import UiCoordData, GenParams, num_pixels
from .misc.shmem_utils import SharedTextureList

from multiprocessing.shared_memory import SharedMemory
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
from math import ceil
import numpy as np



# -- NOTE -- ___________________________________________________________________________________________________________
#
#   This implementation improves vanilla (a.k.a. sequential) image quilting speed via parallelization.
#
#   The following is a reference to the source implementation contained in the jena2020 folder:
#       https://github.com/rohitrango/Image-Quilting-for-Texture-Synthesis
#
#   And the original algorithm paper:
#       https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf
#
#   Conceptually, parallelization is done by creating a horizontal and vertical stripes in the shape of a cross
#   dividing the image into 4 sections that can be generated independently.
#   Additionally, each section's rows can also be "generated in parallel" ( clarifications below in 2.2 ).
#
#   The operations are as follows:
#   1.
#       2 stripes for each direction are generated, all in parallel (if enough cores available).
#       1 vertical and 1 horizontal stripes are created with inverted images, in order to re-use the original
#       implementation methods without any additional modifications.
#
#   2.1
#       The 4 separate sections are generated at the same time, using the required inversions,
#       and then flipped at the end so that all can be stitched together
#
#   2.2
#   An OPTIONAL parallelization step is implemented, but improvement is not significant.
#   ( also, can be worse in small generations, with a low number of patches or small source image )
#   While generating each section, instead of generating each row sequentially, multiple rows "can run in parallel" in
#   alternating fashion. The modulo of the row number is run by the job with that number, e.g.:  if using two jobs,
#   one runs the odd rows and another the even rows.
#   Note that one row section CAN ONLY be computed ONCE THE ADJACENT SECTION IN THE PRIOR ROW HAS BEEN COMPUTED.
#   The algorithm is still sequential in nature, this is not some "modern" variant of the algorithm,
#   but it does not require a row to be completely filled in order to compute some of the next row's patches.
#
#   3.
#   Stitching is straightforward, and only needs to take into account the removal
#   of the shared components belonging to the initially generated stripes.
#

# region     methods adapted from jena2020 for re-usability & node compliance

def fill_column(lookup_textures: list[np.ndarray]|SharedTextureList,
                initial_block, gen_args: GenParams, rows: int, rng: np.random.Generator):
    find_patch_below = get_find_patch_below_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_vertical_method(gen_args.version)
    block_size = initial_block.shape[0]
    overlap = gen_args.overlap

    # TODO set this up later to avoid fetching the texture
    #if isinstance(lookup_textures, SharedTextureList):
    #    shape, dtype = lookup_textures.metadata.global_number_of_channels, lookup_textures.metadata.global_dtype
    #else:
    #    shape, dtype = lookup_textures[0].shape, lookup_textures[0].dtype

    image = lookup_textures[0]
    texture_map = np.empty(
        ((block_size + rows * (block_size - overlap)), block_size, image.shape[2]), dtype=image.dtype)
    texture_map[:block_size, :block_size, :] = initial_block
    del image

    seams_map = np.zeros((texture_map.shape[0], block_size), dtype=np.float32)

    for i, blk_idx in enumerate(range((block_size - overlap), texture_map.shape[0] - overlap, (block_size - overlap))):
        ref_block = texture_map[(blk_idx - block_size + overlap):(blk_idx + overlap), :block_size]
        patch_block = find_patch_below(ref_block, lookup_textures, gen_args, rng)
        min_cut_patch, patch_weights = get_min_cut_patch(ref_block, patch_block, gen_args)
        texture_map[blk_idx:(blk_idx + block_size), :block_size] = min_cut_patch

        seams_map_view = seams_map[blk_idx:(blk_idx + block_size), :block_size]
        update_seams_map_view(seams_map_view, gen_args, patch_weights)

    return texture_map, seams_map


def fill_row(lookup_textures: list[np.ndarray]|SharedTextureList,
             initial_block, gen_args: GenParams, columns: int, rng: np.random.Generator):
    find_patch_to_the_right = get_find_patch_to_the_right_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_horizontal_method(gen_args.version)
    block_size = initial_block.shape[0]
    overlap = gen_args.overlap

    image = lookup_textures[0]
    texture_map = np.empty(
        (block_size, (block_size + columns * (block_size - overlap)), image.shape[2]), dtype=image.dtype)
    texture_map[:block_size, :block_size, :] = initial_block
    del image

    seams_map = np.zeros((block_size, texture_map.shape[1])).astype(np.float32)

    for i, blk_idx in enumerate(range((block_size - overlap), texture_map.shape[1] - overlap, (block_size - overlap))):
        ref_block = texture_map[:block_size, (blk_idx - block_size + overlap):(blk_idx + overlap)]
        patch_block = find_patch_to_the_right(ref_block, lookup_textures, gen_args, rng)
        min_cut_patch, patch_weights = get_min_cut_patch(ref_block, patch_block, gen_args)
        texture_map[:block_size, blk_idx:(blk_idx + block_size)] = min_cut_patch

        seams_map_view = seams_map[:block_size, blk_idx:(blk_idx + block_size)]
        update_seams_map_view(seams_map_view, gen_args, patch_weights)
    return texture_map, seams_map


def fill_quad(rows: int, columns: int, gen_args: GenParams, texture_map, seams_map,
              lookup_textures: list[np.ndarray]|SharedTextureList,
              rng: np.random.Generator, uicd: UiCoordData | None):
    find_patch_both = get_find_patch_both_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_both_method(gen_args.version)
    block_size, overlap = gen_args.bo

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            blk_index_i = i * (block_size - overlap)
            blk_index_j = j * (block_size - overlap)
            ref_block_left = texture_map[
                             blk_index_i:(blk_index_i + block_size),
                             (blk_index_j - block_size + overlap):(blk_index_j + overlap)]
            ref_block_top = texture_map[
                            (blk_index_i - block_size + overlap):(blk_index_i + overlap),
                            blk_index_j:(blk_index_j + block_size)]

            patch_block = find_patch_both(ref_block_left, ref_block_top, lookup_textures, gen_args, rng)
            min_cut_patch, patch_weights = get_min_cut_patch(ref_block_left, ref_block_top, patch_block, gen_args)

            texture_map[blk_index_i:(blk_index_i + block_size), blk_index_j:(blk_index_j + block_size)] = min_cut_patch

            seams_map_view = seams_map[blk_index_i:(blk_index_i + block_size), blk_index_j:(blk_index_j + block_size)]
            update_seams_map_view(seams_map_view, gen_args, patch_weights)

        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(columns):
            break
    return texture_map, seams_map


# endregion

# region    parallel solution


def generate_texture_parallel(src_textures: list[np.ndarray],
                              gen_args: GenParams, out_h: num_pixels, out_w: num_pixels, nps: int,
                              rng: np.random.Generator, uicd: UiCoordData | None):
    """
    @param out_h: output's height in pixels
    @param out_w: output's width in pixels
    @param uicd: contains:
                 * the shared memory name to a one dimensional array that stores the number of "blocks"
                 (with the overlapping area removed) processed by each job;
                 * the job id which should be equal to the batch index, without accounting for the sub processes.
    @param nps: number of parallel stripes; tells how many jobs to use for each of the 4 sections.
    """
    block_size, overlap = gen_args.block_size, gen_args.overlap

    if uicd is not None:
        uicd = UiCoordData(
            uicd.jobs_shm_name,
            uicd.job_id * 4 * nps  # offset job id according to the number of sub jobs
        )

    # select random image for the starting block
    image = src_textures[rng.integers(len(src_textures))]
    num_channels = image.shape[2]
    src_dtype = image.dtype

    # Starting block
    source_height, source_width = image.shape[:2]
    corner_y = rng.integers(source_height - block_size)
    corner_x = rng.integers(source_width - block_size)
    start_block = image[corner_y:corner_y + block_size, corner_x:corner_x + block_size]

    # get the vertical and horizontal flipped variants of the starting block
    vi_start_block = np.flipud(start_block)
    hi_start_block = np.fliplr(start_block)

    del image  # access textures list instead after this point

    # Get flipped textures to build the quadrants
    # Note: this step might be redundant in some use cases,
    #       but without passing some additional metadata there is no way to know
    hi_ltxts: list[np.ndarray] = []
    vi_ltxts: list[np.ndarray] = []
    for ltxt in src_textures:
        hi_ltxts.append(np.fliplr(ltxt))
        vi_ltxts.append(np.flipud(ltxt))
    # note: inverted both ways is computed later when filling the top-left quadrant/section

    # init stuff that will have to be released eventually
    parallel = Parallel(n_jobs=4, backend="loky", timeout=None, verbose=0)
    shm_ltxts = SharedTextureList.from_list(src_textures)
    shm_hi_ltxts = SharedTextureList.from_list(hi_ltxts)
    shm_vi_ltxts = SharedTextureList.from_list(vi_ltxts)

    try:
        del src_textures, hi_ltxts, vi_ltxts

        # auxiliary variables
        quad_row_width = ceil(out_w / 2 + block_size / 2)
        quad_column_height = ceil(out_h / 2 + block_size / 2)
        cols_per_quad = int(ceil((quad_row_width - block_size) / (block_size - overlap)))  # minding the overlap
        rows_per_quad = int(ceil((quad_column_height - block_size) / (block_size - overlap)))
        # might get 1 more row or column than needed here

        # generate 2 vertical strips and 2 horizontal strips that will split the generated canvas in half
        # the center, where the stripes connect, shares the same tile
        # image, initial_block, gen_args: GenParams, columns: int, rng: np.random.Generator
        args = [
            (shm_ltxts, start_block, gen_args, cols_per_quad, rng),
            (shm_hi_ltxts, hi_start_block, gen_args, cols_per_quad, rng),
            (shm_ltxts, start_block, gen_args, rows_per_quad, rng),
            (shm_vi_ltxts, vi_start_block, gen_args, rows_per_quad, rng)
        ]
        funcs = [fill_row, fill_row, fill_column, fill_column]

        stripes = parallel(delayed(funcs[i])(*args[i]) for i in range(4))
        text_stripes, smap_stripes = zip(*stripes)
        hs, his, vs, vis = text_stripes
        sm_hs, sm_his, sm_vs, sm_vis = smap_stripes
        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(rows_per_quad * 2 + cols_per_quad * 2):
            return None

        # generate the 4 sections (quadrants)
        args = [
            (vis, his, shm_hi_ltxts, rows_per_quad, cols_per_quad, gen_args, nps, rng, sm_vis, sm_his, uicd),
            (vis, hs, shm_vi_ltxts, rows_per_quad, cols_per_quad, gen_args, nps, rng, sm_vis, sm_hs, uicd),
            (vs, hs, shm_ltxts, rows_per_quad, cols_per_quad, gen_args, nps, rng, sm_vs, sm_hs, uicd),
            (vs, his, shm_hi_ltxts, rows_per_quad, cols_per_quad, gen_args, nps, rng, sm_vs, sm_his, uicd)
        ]
        funcs = [quad1, quad2, quad3, quad4]
        quads = parallel(delayed(funcs[i])(*args[i]) for i in range(4))
        text_quads, smap_quads = zip(*quads)
        tq1, tq2, tq3, tq4 = text_quads
        sq1, sq2, sq3, sq4 = smap_quads

        texture = np.empty((tq1.shape[0] * 2 - block_size, tq1.shape[1] * 2 - block_size, num_channels), dtype=src_dtype)
        seams_map = np.empty((texture.shape[0], texture.shape[1]), dtype=vis.dtype)

        bmo = block_size - overlap
        texture[:tq1.shape[0] - bmo, :tq1.shape[1] - bmo] = tq1[:tq1.shape[0] - bmo, :tq1.shape[1] - bmo]
        texture[:tq1.shape[0] - bmo, tq1.shape[1] - bmo:] = tq2[:tq1.shape[0] - bmo, overlap:]
        texture[tq1.shape[0] - bmo:, :tq1.shape[1] - bmo] = tq4[overlap:, :tq1.shape[1] - bmo]
        texture[tq1.shape[0] - bmo:, tq1.shape[1] - bmo:] = tq3[overlap:, overlap:]

        seams_map[:tq1.shape[0] - bmo, :tq1.shape[1] - bmo] = sq1[:tq1.shape[0] - bmo, :tq1.shape[1] - bmo]
        seams_map[:tq1.shape[0] - bmo, tq1.shape[1] - bmo:] = sq2[:tq1.shape[0] - bmo, overlap:]
        seams_map[tq1.shape[0] - bmo:, :tq1.shape[1] - bmo] = sq4[overlap:, :tq1.shape[1] - bmo]
        seams_map[tq1.shape[0] - bmo:, tq1.shape[1] - bmo:] = sq3[overlap:, overlap:]

        # fix overvalues due to seams overlap
        np.clip(seams_map, 0, 1, out=seams_map)

        return texture[:out_h, :out_w], seams_map[:out_h, :out_w]
    finally:
        # this is needed to guarantee the process have been shutdown
        # otherwise deleting the temporary may fail
        get_reusable_executor().shutdown(wait=True)

        for shm_txt_list in [shm_ltxts, shm_hi_ltxts, shm_vi_ltxts]:
            shm_txt_list.release()

# region    sub-routines for quadX functions

def _allocate_arrays(h: int, w: int, channels: int, dtype: np.dtype, use_shm: bool):
    """
    Return: texture, seams_map, shm_text, shm_smap
    - If use_shm==True the returned texture/seams_map are views into shared memory
      and shm_text/shm_smap are SharedMemory objects.
    - If use_shm==False the returned arrays are normal numpy arrays and shm_* are None.
    """
    if use_shm:
        num_pixels = h * w
        shm_text = SharedMemory(create=True, size=num_pixels * channels * dtype.itemsize)
        shm_smap = SharedMemory(create=True, size=num_pixels * np.float32().itemsize)
        texture = np.ndarray((h, w, channels), dtype=dtype, buffer=shm_text.buf)
        seams_map = np.ndarray((h, w), dtype=np.float32, buffer=shm_smap.buf)
        seams_map[:] = 0
        return texture, seams_map, shm_text, shm_smap
    else:
        texture = np.empty((h, w, channels), dtype=dtype)
        seams_map = np.zeros((h, w), dtype=np.float32)
        return texture, seams_map, None, None


@contextmanager
def shm_pair(h: int, w: int, channels: int, dtype: np.dtype, use_shm: bool):
    """
    Context manager that yields (texture, seams_map, shm_text, shm_smap).
    On exit it always closes & unlinks SHM objects if they were created.
    IMPORTANT: arrays backed by SHM must be copied before leaving the context manager
    (e.g. via np.ascontiguousarray(...)).
    """
    texture = seams_map = shm_text = shm_smap = None
    try:
        texture, seams_map, shm_text, shm_smap = _allocate_arrays(h, w, channels, dtype, use_shm)
        yield texture, seams_map, shm_text, shm_smap
    finally:
        # close & unlink shared memory only if it exists (safe even if exception thrown)
        for shm in (shm_text, shm_smap):
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    # already unlinked or never created on some platforms
                    pass


def _run_fill_quad(rows, columns, gen_args, texture, seams_map,
                   ltxts: list[np.ndarray] | SharedTextureList,
                   p_strips, rng, shm_text: Optional[SharedMemory], shm_smap: Optional[SharedMemory],
                   uicd: UiCoordData | None):
    """
    Call fill_quad_ps (when using SHM) or fill_quad (when not).
    Returns the (texture, seams_map) pair in both cases.
    """
    if p_strips > 1:
        if isinstance(ltxts, SharedTextureList):
            fill_quad_ps(rows, columns, gen_args, shm_text.name, shm_smap.name,
                         ltxts, p_strips, rng, uicd)
        else:
            with SharedTextureList.from_list(ltxts) as shm_ltxts:
                fill_quad_ps(rows, columns, gen_args, shm_text.name, shm_smap.name,
                             shm_ltxts, p_strips, rng, uicd)
                get_reusable_executor().shutdown(wait=True)
        return texture, seams_map
    else:   # if p_strips == 1
        # fill_quad returns (texture, seams_map)
        return fill_quad(rows, columns, gen_args, texture, seams_map, ltxts, rng, uicd)

# endregion     sub-routines for quadX functions


def quad1(vis, his,
          hi_ltxts: list[np.ndarray] | SharedTextureList,
          rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          sm_vis: np.ndarray, sm_his: np.ndarray,
          uicd: UiCoordData | None):
    """
    @param vis: vertical inverted stripe
    @param his: horizontal inverted stripe
    @param sm_vis: vertical inverted seams map of the computed stripe
    @param sm_his: horizontal inverted seams map of the computed stripe
    @param hi_ltxts: horizontal inverted lookup textures
    @param rows: number of blocks to compute per row
    @param columns: number of blocks to compute per columns
    @param p_strips: number of processes for building the quadrant (not accounting for the other quadrants)
    """
    vi_hi_s = np.ascontiguousarray(np.flipud(his))  # vertical inversion of the horizontal inverted stripe
    hi_vi_s = np.ascontiguousarray(np.fliplr(vis))

    vhi_ltxts = [np.ascontiguousarray(np.flipud(txt)) for txt in hi_ltxts]

    sm_vi_hi_s = np.ascontiguousarray(np.flipud(sm_his))
    sm_hi_vi_s = np.ascontiguousarray(np.flipud(sm_vis))

    H, W = vis.shape[0], his.shape[1]
    use_shm = (p_strips > 1)

    with shm_pair(H, W, vis.shape[2], vis.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        # place stripes & seams
        texture[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = vi_hi_s[:, :]
        texture[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = hi_vi_s[vi_hi_s.shape[0]:, :]
        seams_map[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = sm_vi_hi_s[:, :]
        seams_map[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = sm_hi_vi_s[vi_hi_s.shape[0]:, :]

        # call fill
        texture, seams_map = _run_fill_quad(
            rows, columns, gen_args,
            texture, seams_map, vhi_ltxts,
            p_strips, rng, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 0)
        )

        # make copies and apply final flips before SHM is unlinked when leaving the context
        texture_out = np.ascontiguousarray(np.flip(texture, axis=(0, 1)))
        seams_map_out = np.ascontiguousarray(np.flip(seams_map, axis=(0, 1)))

    return texture_out, seams_map_out


def quad2(vis, hs,
          vi_ltxts: list[np.ndarray] | SharedTextureList,
          rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          sm_vis: np.ndarray, sm_hs: np.ndarray,
          uicd: UiCoordData | None):

    vi_hs = np.ascontiguousarray(np.flipud(hs))  # flip the stripe
    sm_vi_hs = np.ascontiguousarray(np.flipud(sm_hs))
    sm_vis_f = np.ascontiguousarray(np.flipud(sm_vis))

    H, W = vis.shape[0], hs.shape[1]
    use_shm = (p_strips > 1)

    with shm_pair(H, W, vis.shape[2], vis.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:hs.shape[0], :hs.shape[1]] = vi_hs
        texture[hs.shape[0]:vis.shape[0], :vis.shape[1]] = vis[hs.shape[0]:]
        seams_map[:hs.shape[0], :hs.shape[1]] = sm_vi_hs
        seams_map[hs.shape[0]:vis.shape[0], :vis.shape[1]] = sm_vis_f[hs.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, gen_args,
            texture, seams_map, vi_ltxts,
            p_strips, rng, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 1)
        )

        # copy & final flip (flipud)
        texture_out = np.ascontiguousarray(np.flipud(texture))
        seams_map_out = np.ascontiguousarray(np.flipud(seams_map))

    return texture_out, seams_map_out


def quad4(vs, his,
          hi_ltxts: list[np.ndarray] | SharedTextureList,
          rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          sm_vs: np.ndarray, sm_his: np.ndarray,
          uicd: UiCoordData | None):

    hi_vs = np.ascontiguousarray(np.fliplr(vs))

    sm_hi_vs = np.ascontiguousarray(np.flipud(sm_vs))
    sm_his_f = np.ascontiguousarray(np.flipud(sm_his))

    H, W = vs.shape[0], his.shape[1]
    use_shm = (p_strips > 1)

    with shm_pair(H, W, hi_vs.shape[2], hi_vs.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:his.shape[0], :his.shape[1]] = his
        texture[his.shape[0]:vs.shape[0], :vs.shape[1]] = hi_vs[his.shape[0]:]
        seams_map[:his.shape[0], :his.shape[1]] = sm_his_f
        seams_map[his.shape[0]:vs.shape[0], :vs.shape[1]] = sm_hi_vs[his.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, gen_args,
            texture, seams_map, hi_ltxts,
            p_strips, rng, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 2)
        )

        # copy & final fliplr
        texture_out = np.ascontiguousarray(np.fliplr(texture))
        seams_map_out = np.ascontiguousarray(np.fliplr(seams_map))

    return texture_out, seams_map_out


def quad3(vs, hs,
          ltxts: list[np.ndarray] | SharedTextureList,
          rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          sm_vs: np.ndarray, sm_hs: np.ndarray,
          uicd: UiCoordData | None):

    H, W = vs.shape[0], hs.shape[1]
    use_shm = (p_strips > 1)

    with shm_pair(H, W, vs.shape[2], vs.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:hs.shape[0], :hs.shape[1]] = hs
        texture[hs.shape[0]:vs.shape[0], :vs.shape[1]] = vs[hs.shape[0]:]
        seams_map[:hs.shape[0], :hs.shape[1]] = sm_hs
        seams_map[hs.shape[0]:vs.shape[0], :vs.shape[1]] = sm_vs[hs.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, gen_args,
            texture, seams_map, ltxts,
            p_strips, rng, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 3)
        )

        # quad3: make copies (no flips)
        texture_out = texture.copy()
        seams_map_out = seams_map.copy()

    return texture_out, seams_map_out


# region        parallel cascading stripes


def fill_quad_ps(rows, columns, gen_args: GenParams,
                 texture_shared_mem_name: str,
                 seams_map_shared_mem_name: str,
                 shm_ltxts: SharedTextureList,
                 total_procs, rng, uicd: UiCoordData | None):
    from multiprocessing import Manager

    # note : ShareableList seems to have problems even though there are no concurrent writes to the same position...
    # setup shared array to store & check each job row & column
    size = 2 * total_procs * np.int32(1).itemsize
    shm_coord = SharedMemory(create=True, size=size)
    try:
        np_coord = np.ndarray((2 * total_procs,), dtype=np.int32, buffer=shm_coord.buf)
        for ip in range(total_procs):
            np_coord[2 * ip] = 1 + ip
            np_coord[2 * ip + 1] = 1

        job_data = ParaRowsJobInfo(
            total_procs=total_procs,
            coord_shared_list_name=shm_coord.name,
            texture_shm_name=texture_shared_mem_name,
            seams_map_shm_name=seams_map_shared_mem_name,
            shm_lookup_textures=shm_ltxts,
            gen_args=gen_args, rng=rng,
            columns=columns, rows=rows
        )

        with Manager() as manager:
            events = {pid: manager.Event() for pid in range(total_procs)}  # the stripe sub-job index (not the os pid)
            Parallel(n_jobs=total_procs, backend="loky", timeout=None)(
                delayed(fill_rows_ps)(i, job_data, events,
                                      None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + i))
                for i in range(total_procs))
    finally:
        shm_coord.close()
        shm_coord.unlink()


@dataclass
class ParaRowsJobInfo:
    total_procs: int
    coord_shared_list_name: str
    texture_shm_name: str
    seams_map_shm_name: str
    shm_lookup_textures: SharedTextureList
    gen_args: GenParams
    rng: np.random.Generator
    columns: int
    rows: int


def fill_rows_ps(pid: int, job: ParaRowsJobInfo, jobs_events: list, uicd: UiCoordData | None):
    gen_args = job.gen_args
    find_patch_both = get_find_patch_both_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_both_method(gen_args.version)

    # unwrap data
    block_size, overlap = gen_args.bo
    rng = job.rng
    total_procs, ltxts, rows, columns = job.total_procs, job.shm_lookup_textures, job.rows, job.columns
    bmo = block_size - overlap
    b_o = ceil(block_size / bmo)

    # get data in shared memory
    shm_coord_ref = SharedMemory(name=job.coord_shared_list_name)
    coord_list = np.ndarray((2 * job.total_procs,), dtype=np.int32, buffer=shm_coord_ref.buf)
    prior_pid = (pid + job.total_procs - 1) % job.total_procs
    shm_texture = SharedMemory(name=job.texture_shm_name)
    texture = np.ndarray(
        shape=(block_size + rows * bmo, block_size + columns * bmo, ltxts.metadata.global_number_of_channels),
        dtype=ltxts.metadata.global_dtype,
        buffer=shm_texture.buf)
    shm_seams_map = SharedMemory(name=job.seams_map_shm_name)
    seams_map = np.ndarray((block_size + rows * bmo, block_size + columns * bmo), dtype=np.float32,
                         buffer=shm_seams_map.buf)

    for i in range(1 + pid, rows + 1, total_procs):
        coord_list[pid * 2 + 1] = -b_o  # indicates no column yet processed on this row
        coord_list[pid * 2 + 0] = i
        for j in range(1, columns + 1):
            # if previous row hasn't processed the adjacent section yet wait for it to advance.
            # -1 is used to shortcircuit this check when the job on the prior row has completed all rows.
            while -1 < coord_list[prior_pid * 2 + 0] < i and \
                    coord_list[prior_pid * 2 + 1] - b_o <= j:
                jobs_events[prior_pid].wait()
                jobs_events[prior_pid].clear()

            # The same as source implementation ( similar to fill_quad )
            blk_index_i = i * bmo
            blk_index_j = j * bmo
            ref_block_left = texture[
                             blk_index_i:(blk_index_i + block_size),
                             (blk_index_j - block_size + overlap):(blk_index_j + overlap)]
            ref_block_top = texture[
                            (blk_index_i - block_size + overlap):(blk_index_i + overlap),
                            blk_index_j:(blk_index_j + block_size)]

            patch_block = find_patch_both(ref_block_left, ref_block_top, ltxts, gen_args, rng)
            min_cut_patch, patch_weights = get_min_cut_patch(ref_block_left, ref_block_top, patch_block, gen_args)

            texture[blk_index_i:(blk_index_i + block_size), blk_index_j:(blk_index_j + block_size)] = min_cut_patch

            seams_map_view = seams_map[blk_index_i:(blk_index_i + block_size), blk_index_j:(blk_index_j + block_size)]
            update_seams_map_view(seams_map_view, gen_args, patch_weights)

            coord_list[pid * 2 + 1] = j
            jobs_events[pid].set()

        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(columns):
            break  # is required to set job as complete and avoid deadlock in jobs waiting for prior row completion

    coord_list[pid * 2 + 0] = -1  # set job as completed
    jobs_events[pid].set()


# endregion     parallel cascading stripes

# endregion     parallel solution

# region    non-parallel solution adapted from jena2020 for node compliance & error function optionality


def generate_texture(src_textures: list[np.ndarray],
                     gen_args: GenParams,
                     out_h: num_pixels, out_w: num_pixels,
                     rng: np.random.Generator,
                     uicd: UiCoordData | None):
    """
    @param out_h: output's height in pixels
    @param out_w: output's width in pixels
    """
    block_size, overlap = gen_args.block_size, gen_args.overlap

    n_h = int(ceil((out_h - block_size) / (block_size - overlap)))
    n_w = int(ceil((out_w - block_size) / (block_size - overlap)))

    # select random image for the starting block
    image = src_textures[rng.integers(len(src_textures))]

    texture_map = np.zeros(
        ((block_size + n_h * (block_size - overlap)),
         (block_size + n_w * (block_size - overlap)),
         image.shape[2])).astype(image.dtype)

    seams_map = np.zeros((texture_map.shape[0], texture_map.shape[1]), dtype=np.float32)

    # Starting index and block
    h, w = image.shape[:2]
    rand_h = rng.integers(h - block_size)
    rand_w = rng.integers(w - block_size)

    start_block = image[rand_h:rand_h + block_size, rand_w:rand_w + block_size]
    texture_map[:block_size, :block_size, :] = start_block

    del image  # access texture list instead after this point

    # fill 1st row
    texture_map[:block_size, :], seams_map[:block_size, :] = fill_row(src_textures, start_block, gen_args, n_w, rng)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
        return None

    # fill 1st column
    texture_map[:, :block_size], seams_map[:, :block_size] = fill_column(src_textures, start_block, gen_args, n_h, rng)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_h):
        return None

    # fill the rest
    texture_map, seams_map = fill_quad(n_h, n_w, gen_args, texture_map, seams_map, src_textures, rng, uicd)

    # fix overvalues due to seams overlap
    np.clip(seams_map, 0, 1, out=seams_map)

    # crop to final size
    return texture_map[:out_h, :out_w], seams_map[:out_h, :out_w]

# endregion
