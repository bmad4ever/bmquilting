import random

import cv2
from numpy.random.bit_generator import SeedSequence

from .synthesis_subroutines import (
    get_find_patch_to_the_right_method, get_find_patch_below_method, get_find_patch_both_method,
    get_min_cut_patch_horizontal, get_min_cut_patch_vertical, get_min_cut_patch_both,
    update_seams_map_view)
from .types import GenParams, NumPixels, _2D_Slice
from .misc.shmem_utils import SharedTextureList
from .misc.texture_utils import quick_checksum
from .misc.custom_decorators import clear_cache_post_exec, step_predictor
from .misc.ui_coord import handle_ui_interrupts, UiCoordData, check_ui

from multiprocessing.shared_memory import SharedMemory
from joblib.externals.loky import get_reusable_executor
from joblib import Parallel, delayed

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Callable
from math import ceil
import numpy as np

type RetOnInterrupt = tuple[None, None]
ret_val_on_interrupt = (None, None)

type FindPatchFunc = Callable[[np.ndarray, list[np.ndarray] | SharedTextureList, GenParams, np.random.Generator], np.ndarray]
type MinCutFunc = Callable[[np.ndarray, np.ndarray, GenParams], np.ndarray]


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
#   Additionally, each section's rows can also be "generated in parallel".
#
#   The operations are as follows:
#   1.  At the center, create a patched region of dims equal to block_size + 2 (block_size - overlap).
#       This slightly larger than a block section is created so that the stripes can be generated without
#       sharing a common center tile whose corners would be shared by multiple stripes.
#       ( A block size + 2 overlaps sized region would suffice, but would mess up block alignment in this solution )
#
#   2.
#       2 stripes for each direction are generated, all in parallel (if enough cores available).
#       1 vertical and 1 horizontal stripes are created with inverted images, in order to re-use the original
#       implementation methods without any additional modifications.
#
#   3
#       The 4 separate sections are generated at the same time, using the required inversions,
#       and then flipped at the end so that all can be stitched together
#
#   4.
#   An OPTIONAL parallelization step is implemented, but improvement is not significant.
#   ( also, can be worse in small generations, with a low number of patches or small source image )
#   While generating each section, instead of generating each row sequentially, multiple rows "can run in parallel" in
#   alternating fashion. The modulo of the row number is run by the job with that number, e.g.:  if using two jobs,
#   one runs the odd rows and another the even rows.
#   Note that one row section CAN ONLY be computed ONCE THE ADJACENT SECTION IN THE PRIOR ROW HAS BEEN COMPUTED.
#   The algorithm is still sequential in nature, this is not some "modern" variant of the algorithm,
#   but it does not require a row to be completely filled in order to compute some of the next row's patches.
#
#   5.
#   Stitching is straightforward, and only needs to take into account the removal
#   of the shared components belonging to the initially generated stripes.
#

# region     methods adapted from jena2020 for re-usability & node compliance

def process_block(text_view: np.ndarray, seams_view: np.ndarray,
                  lookup_textures: list[np.ndarray] | SharedTextureList,
                  block_idx: _2D_Slice,
                  find_patch_func: FindPatchFunc, get_min_cut_func: MinCutFunc,
                  gen_params: GenParams, rng: np.random.Generator,
                  uicd: UiCoordData | None = None) -> None:
    ref_block = text_view[block_idx]
    patch_block = find_patch_func(ref_block, lookup_textures, gen_params, rng)
    min_cut_patch, patch_weights = get_min_cut_func(ref_block, patch_block, gen_params)
    ref_block[:] = min_cut_patch
    seams_map_sub_view = seams_view[block_idx]
    update_seams_map_view(seams_map_sub_view, patch_weights, gen_params.blend_into_patch)
    check_ui(uicd, 1)


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_column(lookup_textures: list[np.ndarray] | SharedTextureList,
                  initial_block, gen_params: GenParams, rows: int, seed: SeedSequence,
                  initial_block_seams: np.ndarray = None, uicd: UiCoordData | None = None
                  ) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    block_size, overlap = gen_params.block_size, gen_params.overlap

    if isinstance(lookup_textures, SharedTextureList):
        num_channels, dtype = lookup_textures.metadata.global_number_of_channels, lookup_textures.metadata.global_dtype
    else:
        num_channels, dtype = lookup_textures[0].shape, lookup_textures[0].dtype

    texture_map = np.zeros(((initial_block.shape[0] + rows * (block_size - overlap)), block_size, num_channels),
                           dtype=dtype)
    texture_map[:initial_block.shape[0], :, :] = initial_block
    texture_map_view = texture_map[initial_block.shape[0] - block_size:, :]

    seams_map = np.zeros((texture_map.shape[0], block_size), dtype=np.float32)
    if initial_block_seams is not None:
        seams_map[:initial_block_seams.shape[0], :block_size] = initial_block_seams
    seams_map_view = seams_map[initial_block.shape[0] - block_size:, :]

    rng = np.random.default_rng(seed)
    _fill_column_inplace(texture_map_view, seams_map_view, lookup_textures, gen_params, rng, uicd)
    return texture_map, seams_map


def _fill_column_inplace(texture_map_view: np.ndarray, seams_map_view: np.ndarray,
                         lookup_textures: list[np.ndarray] | SharedTextureList,
                         gen_params: GenParams, rng: np.random.Generator, uicd: UiCoordData | None = None):
    find_patch_below = get_find_patch_below_method()
    b, o = gen_params.block_size, gen_params.overlap
    bmo = b - o
    for blk_y in range(bmo, texture_map_view.shape[0] - b + 1, bmo):
        block_idx = np.s_[blk_y:blk_y + b, :b]
        process_block(texture_map_view, seams_map_view, lookup_textures, block_idx,
                      find_patch_below, get_min_cut_patch_vertical, gen_params, rng, uicd)


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_row(lookup_textures: list[np.ndarray] | SharedTextureList,
               initial_block, gen_params: GenParams, columns: int, seed: SeedSequence,
               initial_block_seams: np.ndarray = None, uicd: UiCoordData | None = None
               ) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    block_size, overlap = gen_params.block_size, gen_params.overlap

    if isinstance(lookup_textures, SharedTextureList):
        num_channels, dtype = lookup_textures.metadata.global_number_of_channels, lookup_textures.metadata.global_dtype
    else:
        num_channels, dtype = lookup_textures[0].shape, lookup_textures[0].dtype

    texture_map = np.zeros((block_size, (initial_block.shape[1] + columns * (block_size - overlap)), num_channels),
                           dtype=dtype)
    texture_map[:, :initial_block.shape[1], :] = initial_block
    texture_map_view = texture_map[:, initial_block.shape[1] - block_size:]

    seams_map = np.zeros((block_size, texture_map.shape[1])).astype(np.float32)
    if initial_block_seams is not None:
        seams_map[:block_size, :initial_block_seams.shape[1]] = initial_block_seams
    seams_map_view = seams_map[:, initial_block.shape[1] - block_size:]

    rng = np.random.default_rng(seed)
    _fill_row_inplace(texture_map_view, seams_map_view, lookup_textures, gen_params, rng, uicd)
    return texture_map, seams_map


def _fill_row_inplace(texture_map_view: np.ndarray, seams_map_view: np.ndarray,
                      lookup_textures: list[np.ndarray] | SharedTextureList,
                      gen_params: GenParams, rng: np.random.Generator, uicd: UiCoordData | None = None):
    find_patch_to_the_right = get_find_patch_to_the_right_method()
    b, o = gen_params.block_size, gen_params.overlap
    bmo = b - o
    for blk_x in range(bmo, texture_map_view.shape[1] - b + 1, bmo):
        block_idx = np.s_[:b, blk_x:blk_x + b]
        process_block(texture_map_view, seams_map_view, lookup_textures, block_idx,
                      find_patch_to_the_right, get_min_cut_patch_horizontal, gen_params, rng, uicd)


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_quad(gen_params: GenParams, texture_map, seams_map,
                lookup_textures: list[np.ndarray] | SharedTextureList,
                rng: np.random.PCG64,
                uicd: UiCoordData | None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    notes:
     * requires the 1st row and column of blocks to be already filled
     * PCG64 is set as arg type to explicitly enforce consistency with stripe variant implementation
    """
    find_patch_both = get_find_patch_both_method()
    b, o = gen_params.bo
    bmo = b - o
    _rng = rng
    rng = np.random.Generator(rng)
    dy, dx = 1, 1 # TODO DELETE FOR DEBUG ONLY
    for blk_y in range(bmo, texture_map.shape[0] - b + 1, bmo):
        dx = 1
        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            print(f"{dy, dx = }; {_rng.state}")
            block_idx = np.s_[blk_y:(blk_y + b), blk_x:(blk_x + b)]
            process_block(texture_map, seams_map, lookup_textures, block_idx,
                          find_patch_both, get_min_cut_patch_both, gen_params, rng, uicd)
            dx += 1
        dy += 1
    return texture_map, seams_map


def _fill_quad_purist(gen_params: GenParams, texture_map, seams_map,
                      lookup_textures: list[np.ndarray] | SharedTextureList,
                      rng: np.random.Generator, uicd: UiCoordData | None) -> tuple[np.ndarray, np.ndarray]:
    """
        Only requires the first Row already filled.
        A "purist" solution, where there is no apriori column generation.
    """
    find_patch_below = get_find_patch_below_method()
    find_patch_both = get_find_patch_both_method()
    b, o = gen_params.bo
    bmo = b - o
    for blk_y in range(bmo, texture_map.shape[0] - b + 1, bmo):
        block_idx = np.s_[blk_y:blk_y + b, :b]
        process_block(texture_map, seams_map, lookup_textures, block_idx,
                      find_patch_below, get_min_cut_patch_vertical, gen_params, rng, uicd)

        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            block_idx = np.s_[blk_y:blk_y + b, blk_x:blk_x + b]
            process_block(texture_map, seams_map, lookup_textures, block_idx,
                          find_patch_both, get_min_cut_patch_both, gen_params, rng, uicd)
    return texture_map, seams_map


# endregion

# region    parallel solution

def _parallel_generate_texture_step_predictor(gen_params: GenParams, out_h: NumPixels, out_w: NumPixels):
    quad_row_width = out_w / 2
    quad_column_height = out_h / 2
    block_size = gen_params.block_size
    overlap = gen_params.overlap
    cols_per_quad = int(ceil((quad_row_width - block_size) / (block_size - overlap)))  # w/o the last full block
    rows_per_quad = int(ceil((quad_column_height - block_size) / (block_size - overlap)))
    return 9 + cols_per_quad * rows_per_quad * 4 + (cols_per_quad-1)*2+(rows_per_quad-1)*2


@step_predictor(_parallel_generate_texture_step_predictor)
@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def generate_texture_parallel(
        src_textures: list[np.ndarray], gen_params: GenParams, out_h: NumPixels, out_w: NumPixels, nps: int,
        #rng: np.random.Generator,
        seed: int,
        uicd: UiCoordData | None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param out_h: output's height in pixels
    :param out_w: output's width in pixels
    :param nps: number of parallel stripes; tells how many jobs to use for each of the 4 sections.
    :param uicd: utility to track the generation progress and to process UI interrupts
    """
    assert out_w > gen_params.block_size + 2 * (gen_params.block_size - gen_params.overlap)
    assert out_h > gen_params.block_size + 2 * (gen_params.block_size - gen_params.overlap)
    assert gen_params.overlap <= gen_params.block_size // 2

    block_size, overlap = gen_params.block_size, gen_params.overlap
    seeds = iter(SeedSequence(seed).spawn(1 + 4*2)) # 1 + 1 per stripe & quad
    _rng = np.random.default_rng(seed=next(seeds))

    # TODO this is a remnant from allowing to run multiple texture gens simultaneously in comfy when using a list
    #   it makes no sense for a library in isolation UNLESS PROPERLY DOCUMENTED!
    #   However such "utility" workaround would render the handle_ui_interrupts useless in this function, THUS:
    #   it should be the process triggering the multiple generations setting up uicd prior to running the func !!!
    #   I will leave the code commented out for now in case I need to check something later
    #   IT MUST (eventually) BE REMOVED
    #if uicd is not None:
    #    uicd = UiCoordData(
    #        uicd.jobs_shm_name,
    #        uicd.job_id * 4 * nps  # offset job id according to the number of sub jobs
    #    )

    # select random image for the starting block
    image = src_textures[_rng.integers(len(src_textures))]
    num_channels = image.shape[2]
    src_dtype = image.dtype

    del image  # access textures list instead after this point

    # Get flipped textures to build the quadrants
    hi_ltxts: list[np.ndarray] = []
    vi_ltxts: list[np.ndarray] = []
    for ltxt in src_textures:
        hi_ltxts.append(np.fliplr(ltxt))
        vi_ltxts.append(np.flipud(ltxt))
    # note: inverted both ways is computed later when filling the top-left quadrant/section

    # check if lists match; if so there is no need to use a separate SharedTextureList
    src_txt_ids: set[int] = {quick_checksum(txt) for txt in src_textures}
    hi_txt_ids: set[int] = {quick_checksum(txt) for txt in hi_ltxts}
    vi_txt_ids: set[int] = {quick_checksum(txt) for txt in vi_ltxts}
    hi_eq_src = src_txt_ids == hi_txt_ids
    vi_eq_src = src_txt_ids == vi_txt_ids
    del src_txt_ids, hi_txt_ids, vi_txt_ids

    # init stuff that will have to be released eventually
    parallel = Parallel(n_jobs=4, backend="loky", timeout=None, verbose=0)
    shm_ltxts = SharedTextureList.from_list(src_textures)
    shm_hi_ltxts = shm_ltxts if hi_eq_src else SharedTextureList.from_list(hi_ltxts)
    shm_vi_ltxts = shm_ltxts if vi_eq_src else SharedTextureList.from_list(vi_ltxts)

    try:
        # auxiliary variables
        quad_row_width = out_w / 2
        quad_column_height = out_h / 2
        cols_per_quad = int(ceil((quad_row_width - block_size) / (block_size - overlap)))  # w/o the last full block
        rows_per_quad = int(ceil((quad_column_height - block_size) / (block_size - overlap)))
        # might get 1 more row or column than needed here

        # Patch the central area containing the central block shared by in all the stripes
        gen_res = generate_texture.__wrapped__.__wrapped__ (  # access wrapped to avoid clearing cache
            src_textures,
            gen_params,
            block_size + 2 * (block_size - overlap),  # 2*overlap would suffice, but seams would have an offset
            block_size + 2 * (block_size - overlap),
            _rng,
            uicd    # should add 9
        )
        if gen_res is None:
            return ret_val_on_interrupt
        else:
            central_patch, cp_seams = gen_res

        bmo = block_size - overlap
        hor_start_block = central_patch[bmo:-bmo, -block_size - bmo:]
        hor_inv_start_block = np.fliplr(central_patch[bmo:-bmo, :block_size + bmo])
        ver_start_block = central_patch[-block_size - bmo:, bmo:-bmo]
        ver_inv_start_block = np.flipud(central_patch[:block_size + bmo, bmo:-bmo])

        hor_start_block_seams = cp_seams[bmo:-bmo, -block_size - bmo:]
        hor_inv_start_block_seams = np.fliplr(cp_seams[bmo:-bmo, :block_size + bmo])
        ver_start_block_seams = cp_seams[-block_size - bmo:, bmo:-bmo]
        ver_inv_start_block_seams = np.flipud(cp_seams[:block_size + bmo, bmo:-bmo])

        del src_textures, hi_ltxts, vi_ltxts

        # generate 2 vertical strips and 2 horizontal strips that will split the generated canvas in half
        # the center, where the stripes connect, shares the same tile
        # image, initial_block, gen_params: GenParams, columns: int, rng: np.random.Generator
        args = [
            (shm_ltxts, hor_start_block, gen_params, cols_per_quad-1, next(seeds), hor_start_block_seams, uicd),
            (shm_hi_ltxts, hor_inv_start_block, gen_params, cols_per_quad-1, next(seeds), hor_inv_start_block_seams, uicd),
            (shm_ltxts, ver_start_block, gen_params, rows_per_quad-1, next(seeds), ver_start_block_seams, uicd),
            (shm_vi_ltxts, ver_inv_start_block, gen_params, rows_per_quad-1, next(seeds), ver_inv_start_block_seams, uicd)
        ]
        funcs = [_pfill_row, _pfill_row, _pfill_column, _pfill_column]

        stripes = parallel(delayed(funcs[i])(*args[i]) for i in range(4))

        text_stripes, smap_stripes = zip(*stripes)
        hs, his, vs, vis = text_stripes
        sm_hs, sm_his, sm_vs, sm_vis = smap_stripes

        del gen_res, central_patch, cp_seams

        # generate the 4 sections (quadrants)
        args = [
            (vis, his, shm_hi_ltxts, rows_per_quad, cols_per_quad, gen_params, nps, next(seeds), sm_vis, sm_his, uicd),
            (vis, hs, shm_vi_ltxts, rows_per_quad, cols_per_quad, gen_params, nps, next(seeds), sm_vis, sm_hs, uicd),
            (vs, hs, shm_ltxts, rows_per_quad, cols_per_quad, gen_params, nps, next(seeds), sm_vs, sm_hs, uicd),
            (vs, his, shm_hi_ltxts, rows_per_quad, cols_per_quad, gen_params, nps, next(seeds), sm_vs, sm_his, uicd)
        ]
        funcs = [_quad1, _quad2, _quad3, _quad4]
        quads = parallel(delayed(funcs[i])(*args[i]) for i in range(4))

        text_quads, smap_quads = zip(*quads)
        tq1, tq2, tq3, tq4 = text_quads
        sq1, sq2, sq3, sq4 = smap_quads

        texture = np.empty((tq1.shape[0] * 2 - block_size, tq1.shape[1] * 2 - block_size, num_channels),
                           dtype=src_dtype)  # empty here is fine, it won't be passed to any func that computes stuff
        seams_map = np.zeros((texture.shape[0], texture.shape[1]), dtype=vis.dtype)

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

        shm_ltxts.release()
        if not hi_eq_src:
            shm_hi_ltxts.release()
        if not vi_eq_src:
            shm_vi_ltxts.release()


# region    sub-routines for quadX functions


def _allocate_arrays(h: int, w: int, channels: int, dtype: np.dtype, use_shm: bool):
    """
    Returns: texture, seams_map, shm_text, shm_smap

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
        texture[:] = 0
        seams_map[:] = 0
        return texture, seams_map, shm_text, shm_smap
    else:
        texture = np.zeros((h, w, channels), dtype=dtype)
        seams_map = np.zeros((h, w), dtype=np.float32)
        return texture, seams_map, None, None


@contextmanager
def _shm_pair(h: int, w: int, channels: int, dtype: np.dtype, use_shm: bool):
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


def _run_fill_quad(rows, columns, gen_params, texture, seams_map,
                   ltxts: list[np.ndarray] | SharedTextureList,
                   p_strips, seed: SeedSequence, shm_text: SharedMemory | None, shm_smap: SharedMemory | None,
                   uicd: UiCoordData | None):
    """
    Call fill_quad_ps (when using SHM for p_strips > 1) or fill_quad (when not).
    :return: the (texture, seams_map) pair.
    """
    if p_strips > 1:
        if isinstance(ltxts, SharedTextureList):
            _pfill_quad_ps(rows, columns, gen_params, shm_text.name, shm_smap.name,
                           ltxts, p_strips, seed, uicd)
        else:
            with SharedTextureList.from_list(ltxts) as shm_ltxts:
                _pfill_quad_ps(rows, columns, gen_params, shm_text.name, shm_smap.name,
                               shm_ltxts, p_strips, seed, uicd)
                get_reusable_executor().shutdown(wait=True)
        return texture, seams_map
    else:  # if p_strips == 1
        # fill_quad returns (texture, seams_map)
        rng = np.random.PCG64(seed) # needs to be the same generator as the one used in _pfill_quad_ps
        return _pfill_quad(gen_params, texture, seams_map, ltxts, rng, uicd)


# endregion     sub-routines for quadX functions


def _quad1(vis, his,
           hi_ltxts: list[np.ndarray] | SharedTextureList,
           rows: int, columns: int, gen_params: GenParams, p_strips, seed: SeedSequence,
           sm_vis: np.ndarray, sm_his: np.ndarray,
           uicd: UiCoordData | None):
    """
    :param vis: vertical inverted stripe
    :param his: horizontal inverted stripe
    :param sm_vis: vertical inverted seams map of the computed stripe
    :param sm_his: horizontal inverted seams map of the computed stripe
    :param hi_ltxts: horizontal inverted lookup textures
    :param rows: number of blocks to compute per row
    :param columns: number of blocks to compute per columns
    :param p_strips: number of processes for building the quadrant (not accounting for the other quadrants)
    """
    vi_hi_s = np.ascontiguousarray(np.flipud(his))  # vertical inversion of the horizontal inverted stripe
    hi_vi_s = np.ascontiguousarray(np.fliplr(vis))

    vhi_ltxts = [np.ascontiguousarray(np.flipud(txt)) for txt in hi_ltxts]

    sm_vi_hi_s = np.ascontiguousarray(np.flipud(sm_his))
    sm_hi_vi_s = np.ascontiguousarray(np.fliplr(sm_vis))

    H, W = vis.shape[0], his.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(H, W, vis.shape[2], vis.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        # place stripes & seams
        texture[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = vi_hi_s[:, :]
        texture[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = hi_vi_s[vi_hi_s.shape[0]:, :]
        seams_map[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = sm_vi_hi_s[:, :]
        seams_map[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = sm_hi_vi_s[vi_hi_s.shape[0]:, :]

        # call fill
        texture, seams_map = _run_fill_quad(
            rows, columns, gen_params,
            texture, seams_map, vhi_ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 0)
        )

        # make copies and apply final flips before SHM is unlinked when leaving the context
        texture_out = np.ascontiguousarray(np.flip(texture, axis=(0, 1)))
        seams_map_out = np.ascontiguousarray(np.flip(seams_map, axis=(0, 1)))

    return texture_out, seams_map_out


def _quad2(vis, hs,
           vi_ltxts: list[np.ndarray] | SharedTextureList,
           rows: int, columns: int, gen_params: GenParams, p_strips, seed: SeedSequence,
           sm_vis: np.ndarray, sm_hs: np.ndarray,
           uicd: UiCoordData | None):
    vi_hs = np.ascontiguousarray(np.flipud(hs))  # flip the stripe
    sm_vi_hs = np.ascontiguousarray(np.flipud(sm_hs))

    H, W = vis.shape[0], hs.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(H, W, vis.shape[2], vis.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:hs.shape[0], :hs.shape[1]] = vi_hs
        texture[hs.shape[0]:vis.shape[0], :vis.shape[1]] = vis[hs.shape[0]:]
        seams_map[:hs.shape[0], :hs.shape[1]] = sm_vi_hs
        seams_map[hs.shape[0]:vis.shape[0], :vis.shape[1]] = sm_vis[hs.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, gen_params,
            texture, seams_map, vi_ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 1)
        )

        # copy & final flip (flipud)
        texture_out = np.ascontiguousarray(np.flipud(texture))
        seams_map_out = np.ascontiguousarray(np.flipud(seams_map))

    return texture_out, seams_map_out


def _quad4(vs, his,
           hi_ltxts: list[np.ndarray] | SharedTextureList,
           rows: int, columns: int, gen_params: GenParams, p_strips, seed: SeedSequence,
           sm_vs: np.ndarray, sm_his: np.ndarray,
           uicd: UiCoordData | None):
    hi_vs = np.ascontiguousarray(np.fliplr(vs))
    sm_hi_vs = np.ascontiguousarray(np.fliplr(sm_vs))

    H, W = vs.shape[0], his.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(H, W, hi_vs.shape[2], hi_vs.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:his.shape[0], :his.shape[1]] = his
        texture[his.shape[0]:vs.shape[0], :vs.shape[1]] = hi_vs[his.shape[0]:]

        seams_map[:his.shape[0], :his.shape[1]] = sm_his
        seams_map[his.shape[0]:vs.shape[0], :vs.shape[1]] = sm_hi_vs[his.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, gen_params,
            texture, seams_map, hi_ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 2)
        )

        # copy & final fliplr
        texture_out = np.ascontiguousarray(np.fliplr(texture))
        seams_map_out = np.ascontiguousarray(np.fliplr(seams_map))

    return texture_out, seams_map_out


def _quad3(vs, hs,
           ltxts: list[np.ndarray] | SharedTextureList,
           rows: int, columns: int, gen_params: GenParams, p_strips, seed: SeedSequence,
           sm_vs: np.ndarray, sm_hs: np.ndarray,
           uicd: UiCoordData | None):
    H, W = vs.shape[0], hs.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(H, W, vs.shape[2], vs.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:hs.shape[0], :hs.shape[1]] = hs
        texture[hs.shape[0]:vs.shape[0], :vs.shape[1]] = vs[hs.shape[0]:]
        seams_map[:hs.shape[0], :hs.shape[1]] = sm_hs
        seams_map[hs.shape[0]:vs.shape[0], :vs.shape[1]] = sm_vs[hs.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, gen_params,
            texture, seams_map, ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 3)
        )

        # quad3: make copies (no flips)
        texture_out = texture.copy()
        seams_map_out = seams_map.copy()

    return texture_out, seams_map_out


# region        parallel cascading stripes


def _pfill_quad_ps(rows, columns, gen_params: GenParams,
                   texture_shared_mem_name: str,
                   seams_map_shared_mem_name: str,
                   shm_ltxts: SharedTextureList,
                   total_procs, seed: SeedSequence, uicd: UiCoordData | None):
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

        job_data = _ParaRowsJobInfo(
            total_procs=total_procs,
            coord_shared_list_name=shm_coord.name,
            texture_shm_name=texture_shared_mem_name,
            seams_map_shm_name=seams_map_shared_mem_name,
            shm_lookup_textures=shm_ltxts,
            gen_params=gen_params, seed=seed,
            columns=columns, rows=rows
        )

        with Manager() as manager:
            events = {pid: manager.Event() for pid in range(total_procs)}  # the stripe sub-job index (not the os pid)
            Parallel(n_jobs=total_procs, backend="loky", timeout=None)(
                delayed(_pfill_rows_ps)(i, job_data, events,
                                        None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + i))
                for i in range(total_procs))
    finally:
        shm_coord.close()
        shm_coord.unlink()


@dataclass
class _ParaRowsJobInfo:
    total_procs: int
    coord_shared_list_name: str
    texture_shm_name: str
    seams_map_shm_name: str
    shm_lookup_textures: SharedTextureList
    gen_params: GenParams
    seed: SeedSequence
    columns: int
    rows: int


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_rows_ps(pid: int, job: _ParaRowsJobInfo, jobs_events: list, uicd: UiCoordData | None):
    gen_params = job.gen_params
    find_patch_both = get_find_patch_both_method()

    # unwrap data
    block_size, overlap = gen_params.bo
    _rng: np.random.PCG64 = np.random.PCG64(job.seed)
    rng = np.random.Generator(_rng)
    total_procs, ltxts, rows, columns = job.total_procs, job.shm_lookup_textures, job.rows, job.columns
    bmo = block_size - overlap
    b_o = ceil(block_size / bmo)

    # get data in shared memory
    shm_coord_ref = SharedMemory(name=job.coord_shared_list_name)
    coord_list = np.ndarray((2 * job.total_procs,), dtype=np.int32, buffer=shm_coord_ref.buf)
    try:
        prior_pid = (pid + job.total_procs - 1) % job.total_procs
        shm_texture = SharedMemory(name=job.texture_shm_name)
        texture = np.ndarray(
            shape=(block_size + rows * bmo, block_size + columns * bmo, ltxts.metadata.global_number_of_channels),
            dtype=ltxts.metadata.global_dtype,
            buffer=shm_texture.buf)
        shm_seams_map = SharedMemory(name=job.seams_map_shm_name)
        seams_map = np.ndarray((block_size + rows * bmo, block_size + columns * bmo), dtype=np.float32,
                               buffer=shm_seams_map.buf)

        _DEBUG_ID = 0  # TODO DELETE ME!
        if uicd.job_id == _DEBUG_ID:
            print(f"{pid=}")

        # region ____RNG Tracking Setup____

        global_idx = pid * columns   # set starting index

        # offset rng with respect to starting index
        _rng.advance(global_idx // 2)
        if global_idx % 2 != 0:
            rng.integers(100, dtype=np.uint32)  # fake extraction for odd number of items

        number_of_patches_to_skip = (total_procs-1) * columns
        may_land_on_odd_idx = columns % 2 == 1 #and (global_idx % 2 != 0 or number_of_patches_to_skip % 2 != 0)
        number_of_patches_to_skip_intdiv2 = number_of_patches_to_skip // 2

        def advance_rng():
            """updates pcg64 with respect to not processed patches """
            nonlocal global_idx, _rng, rng

            for _ in range(number_of_patches_to_skip):
                rng.integers(100, dtype=np.uint32)
            return

            # TODO  doesn't work as expected, need to investigate PCG64
            _rng.advance(number_of_patches_to_skip_intdiv2)
            global_idx += columns + number_of_patches_to_skip # the target idx after the jump
            if may_land_on_odd_idx and global_idx % 2 != 0:
                rng.integers(100, dtype=np.uint32)

        # endregion ____RGN Tracking Setup____

        for i in range(1 + pid, rows + 1, total_procs):
            coord_list[pid * 2 + 1] = -b_o  # indicates no column yet processed on this row
            coord_list[pid * 2 + 0] = i
            for j in range(1, columns + 1):
                if uicd.job_id == _DEBUG_ID:
                    print(f"{i, j = } ;  {_rng.state =}")

                # if previous row hasn't processed the adjacent section yet wait for it to advance.
                # -1 is used to shortcircuit this check when the job on the prior row has completed all rows.
                while -1 < coord_list[prior_pid * 2 + 0] < i and \
                            coord_list[prior_pid * 2 + 1] - b_o - 2<= j:
                    jobs_events[prior_pid].wait()
                    jobs_events[prior_pid].clear()
                    check_ui(uicd)

                # The same as source implementation ( similar to fill_quad )
                blk_index_i = i * bmo
                blk_index_j = j * bmo

                block_idx = np.s_[blk_index_i:blk_index_i + block_size, blk_index_j:blk_index_j + block_size]
                process_block(texture, seams_map, ltxts, block_idx, find_patch_both, get_min_cut_patch_both,
                              gen_params, rng, uicd)

                coord_list[pid * 2 + 1] = j
                jobs_events[pid].set()

            advance_rng()
    finally:
        # note: this finally block is not "redundant" with respect to "safety";
        #       the .wait() will get stuck if .set() is never called due to an interrupt (or some other) exception.
        coord_list[pid * 2 + 0] = -1  # set job as completed
        jobs_events[pid].set()


# endregion     parallel cascading stripes

# endregion     parallel solution

# region    non-parallel solutions

def _generate_texture_step_predictor(gen_params: GenParams, out_h: NumPixels, out_w: NumPixels):
    block_size = gen_params.block_size
    overlap = gen_params.overlap
    n_h = int(ceil((out_h - block_size) / (block_size - overlap)))
    n_w = int(ceil((out_w - block_size) / (block_size - overlap)))
    return (n_h + 1) * (n_w + 1)


@step_predictor(_generate_texture_step_predictor)
@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def generate_texture(src_textures: list[np.ndarray],
                     gen_params: GenParams,
                     out_h: NumPixels, out_w: NumPixels,
                     rng: np.random.Generator,
                     uicd: UiCoordData | None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param out_h: output's height in pixels
    :param out_w: output's width in pixels
    """
    block_size, overlap = gen_params.block_size, gen_params.overlap

    n_h = int(ceil((out_h - block_size) / (block_size - overlap)))
    n_w = int(ceil((out_w - block_size) / (block_size - overlap)))

    # select random image for the starting block
    image = src_textures[rng.integers(len(src_textures))]

    texture_map = np.zeros(
        ((block_size + n_h * (block_size - overlap)),
         (block_size + n_w * (block_size - overlap)),
         image.shape[2]), dtype=image.dtype)

    seams_map = np.zeros(texture_map.shape[:2], dtype=np.float32)

    # Starting index and block
    h, w = image.shape[:2]
    rand_h = rng.integers(h - block_size)
    rand_w = rng.integers(w - block_size)

    start_block = image[rand_h:rand_h + block_size, rand_w:rand_w + block_size]
    texture_map[:block_size, :block_size, :] = start_block
    check_ui(uicd, to_add=1)

    del image  # access texture list instead after this point

    # --- "Purist" generation: fill only the 1st row and then the rest ---
    _fill_row_inplace(texture_map[:block_size, :], seams_map[:block_size, :], src_textures, gen_params, rng, uicd)
    _fill_quad_purist(gen_params, texture_map, seams_map, src_textures, rng, uicd)

    # fix overvalues due to seams overlap
    np.clip(seams_map, 0, 1, out=seams_map)

    # crop to final size
    return texture_map[:out_h, :out_w], seams_map[:out_h, :out_w]


@clear_cache_post_exec()
def generate_texture_diagonal(src_textures: list[np.ndarray],
                              gen_params: GenParams,
                              out_h: NumPixels, out_w: NumPixels,
                              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
        I've wondered if iterating diagonally with an overlap sized offset --- so that the corners shared by 4 patches are
        covered by the new patches --- could improve the generation quality, and perhaps help avoid "loose" seams.

        For the most part doesn't seem to make that big of a difference.
    """
    b, o = gen_params.block_size, gen_params.overlap
    assert o * 2 < b

    bm2o = b - 2 * o
    _n_h = n_h = ceil(out_h / bm2o)
    _n_w = n_w = ceil(out_w / bm2o)
    if out_h % bm2o <= o:
        _n_h -= 1
    if out_w % bm2o <= o:
        _n_w -= 1

    n_d = min(_n_h, _n_w)  # number of diagonal iterations

    # select random image for the starting block
    image = src_textures[rng.integers(len(src_textures))]

    # texture needs to be generated w/ an overlap sized offset due to the auxiliary functions receiving a block sized
    #  patch instead of just the overlap section
    # mind that we need to over-estimate the size of the texture to make room for the adjacent patches to be filled
    # in rows that may not be aligned
    texture_map = np.zeros((o + b * 2 + n_h * bm2o, o + b * 2 + n_w * bm2o, image.shape[2]), dtype=image.dtype)
    seams_map = np.zeros(texture_map.shape[:2], dtype=np.float32)

    # Starting index and block
    h, w = image.shape[:2]
    rand_h = rng.integers(h - b)
    rand_w = rng.integers(w - b)

    start_block = image[rand_h:rand_h + b, rand_w:rand_w + b]
    texture_map[o:o + b, o:o + b, :] = start_block

    del image

    # --- generation ----
    d_ixd = o  # texture offset ( both for x & y )

    # fill the 1st row & column
    _fill_row_inplace(texture_map[d_ixd:, d_ixd:], seams_map[d_ixd:, d_ixd:], src_textures, gen_params, rng)
    _fill_column_inplace(texture_map[d_ixd:, d_ixd:], seams_map[d_ixd:, d_ixd:], src_textures, gen_params, rng)

    # fill the rest iterating diagonally
    find_patch = get_find_patch_both_method()
    find_cut = get_min_cut_patch_both
    for d in range(1, n_d):
        d_ixd += bm2o

        block_idx = np.s_[d_ixd:d_ixd + b, d_ixd:d_ixd + b]
        process_block(texture_map, seams_map, src_textures, block_idx, find_patch, find_cut, gen_params, rng)

        # columns
        for blk_x in range(d_ixd + b - o, texture_map.shape[1] - b + 1, (b - o)):
            block_idx = np.s_[d_ixd:d_ixd + b, blk_x:blk_x + b]
            process_block(texture_map, seams_map, src_textures, block_idx, find_patch, find_cut, gen_params, rng)

        # rows
        for blk_y in range(d_ixd + b - o, texture_map.shape[0] - b + 1, (b - o)):
            block_idx = np.s_[blk_y:blk_y + b, d_ixd:d_ixd + b]
            process_block(texture_map, seams_map, src_textures, block_idx, find_patch, find_cut, gen_params, rng)

    return texture_map[o:o + out_h, o:o + out_w], seams_map[o:o + out_h, o:o + out_w]


# endregion
