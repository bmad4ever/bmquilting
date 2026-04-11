from __future__ import annotations

from ._internal.square_subroutines import (
    SquarePatchingConfig, SquarePatchingBlendConfig, SeamsAlgorithm,

    get_find_patch_to_the_right_method, get_find_patch_below_method, get_find_patch_both_method,
    get_seam_patched_horizontal, get_seam_patched_vertical, get_seam_patched_both,
    update_seams_map_view, find_patch_vx_idx, find_patch_vx, get_4way_seam_patched,

    # methods w/ cache
    _get_overlap_mask, _get_vignetted_overlap_mask, _patch_blending_vignette
)
from ._internal.seams_blur import (
    # methods w/ cache
    _circular_kernel, _get_max_possible_gradient_diff, _get_radii_limiter, _get_buffers_for_graddiffs_computation,
)

from ._internal.common import (
    TextureList, ValidatedTexturesIterator,
    NumPixels, _2D_Slice, PatchIdx,
    apply_mask, _get_random_valid_block
)
from ._internal.decorators import clear_cache_post_exec, step_predictor, auto_uint8_to_float32
from .utils.ui_coord import handle_ui_interrupts, UiCoordData, check_ui
from ._internal.shmem_utils import SharedTextureList
from .utils.texture import quick_checksum

from joblib.externals.loky import get_reusable_executor
from multiprocessing.shared_memory import SharedMemory
from numpy.random.bit_generator import SeedSequence
from joblib import Parallel, delayed

from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
import numpy as np
import dataclasses
import cv2 as cv

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


type RetOnInterrupt = tuple[None, None]
ret_val_on_interrupt = (None, None)

type _FindPatchFunc = Callable[[np.ndarray, ValidatedTexturesIterator, SquarePatchingConfig, np.random.Generator], np.ndarray]
type _MinCutFunc = Callable[[np.ndarray, np.ndarray, SquarePatchingConfig], np.ndarray]
"""Method variant (hor, vert, both) that computes the seam. Its arguments are: ref_block, patch_block, patching_config."""


_CACHED_FUNCS = [
    _get_buffers_for_graddiffs_computation,
    _get_max_possible_gradient_diff,
    _get_vignetted_overlap_mask,
    _patch_blending_vignette,
    _get_radii_limiter,
    _get_overlap_mask,
    _circular_kernel,
]



# region     _____ AUXILIARY_METHODS _____

def child_seed(root_seed: SeedSequence, spawn_key: int):
    return SeedSequence(
        entropy=root_seed.entropy,
        spawn_key=(spawn_key,)
    )


def process_block(text_view: np.ndarray, seams_view: np.ndarray,
                  lookup_textures: ValidatedTexturesIterator,
                  block_idx: _2D_Slice,
                  find_patch_func: _FindPatchFunc, get_min_cut_func: _MinCutFunc,
                  patching_config: SquarePatchingConfig, rng: np.random.Generator,
                  uicd: UiCoordData | None = None) -> None:
    ref_block = text_view[block_idx]
    patch_block = find_patch_func(ref_block, lookup_textures, patching_config, rng)
    min_cut_patch, patch_weights = get_min_cut_func(ref_block, patch_block, patching_config)
    ref_block[:] = min_cut_patch
    seams_map_sub_view = seams_view[block_idx]
    update_seams_map_view(seams_map_sub_view, patch_weights, patching_config.blend_into_patch)
    check_ui(uicd, 1)


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_column(lookup_textures: ValidatedTexturesIterator,
                  initial_block, patching_config: SquarePatchingConfig, rows: int, seed: SeedSequence,
                  initial_block_seams: np.ndarray = None, uicd: UiCoordData | None = None
                  ) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    block_size, overlap = patching_config.block_size, patching_config.overlap

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
    _fill_column_inplace(texture_map_view, seams_map_view, lookup_textures, patching_config, rng, uicd)
    return texture_map, seams_map


def _fill_column_inplace(texture_map_view: np.ndarray, seams_map_view: np.ndarray,
                         lookup_textures: TextureList | SharedTextureList,
                         patching_config: SquarePatchingConfig, rng: np.random.Generator, uicd: UiCoordData | None = None):
    find_patch_below = get_find_patch_below_method()
    b, o = patching_config.block_size, patching_config.overlap
    bmo = b - o
    for blk_y in range(bmo, texture_map_view.shape[0] - b + 1, bmo):
        block_idx = np.s_[blk_y:blk_y + b, :b]
        process_block(texture_map_view, seams_map_view, lookup_textures, block_idx,
                      find_patch_below, get_seam_patched_vertical, patching_config, rng, uicd)


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_row(lookup_textures: TextureList | SharedTextureList,
               initial_block, patching_config: SquarePatchingConfig, columns: int, seed: SeedSequence,
               initial_block_seams: np.ndarray = None, uicd: UiCoordData | None = None
               ) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    block_size, overlap = patching_config.block_size, patching_config.overlap

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
    _fill_row_inplace(texture_map_view, seams_map_view, lookup_textures, patching_config, rng, uicd)
    return texture_map, seams_map


def _fill_row_inplace(texture_map_view: np.ndarray, seams_map_view: np.ndarray,
                      lookup_textures: ValidatedTexturesIterator,
                      patching_config: SquarePatchingConfig, rng: np.random.Generator, uicd: UiCoordData | None = None):
    find_patch_to_the_right = get_find_patch_to_the_right_method()
    b, o = patching_config.block_size, patching_config.overlap
    bmo = b - o
    for blk_x in range(bmo, texture_map_view.shape[1] - b + 1, bmo):
        block_idx = np.s_[:b, blk_x:blk_x + b]
        process_block(texture_map_view, seams_map_view, lookup_textures, block_idx,
                      find_patch_to_the_right, get_seam_patched_horizontal, patching_config, rng, uicd)


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_quad_inplace(patching_config: SquarePatchingConfig, texture_map, seams_map,
                lookup_textures: ValidatedTexturesIterator,
                seed: SeedSequence, uicd: UiCoordData | None):
    """note: requires the 1st row and column of blocks to be already filled"""
    find_patch_both = get_find_patch_both_method()
    b, o = patching_config.bo
    bmo = b - o
    for blk_y in range(bmo, texture_map.shape[0] - b + 1, bmo):
        rng: np.random.Generator = np.random.default_rng(child_seed(seed, blk_y))
        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            block_idx = np.s_[blk_y:(blk_y + b), blk_x:(blk_x + b)]
            process_block(texture_map, seams_map, lookup_textures, block_idx,
                          find_patch_both, get_seam_patched_both, patching_config, rng, uicd)


def _fill_quad_purist(patching_config: SquarePatchingConfig, texture_map, seams_map,
                      lookup_textures: ValidatedTexturesIterator,
                      rng: np.random.Generator, uicd: UiCoordData | None) -> tuple[np.ndarray, np.ndarray]:
    """
        Only requires the first Row already filled.
        A "purist" solution, where there is no apriori column generation.
    """
    find_patch_below = get_find_patch_below_method()
    find_patch_both = get_find_patch_both_method()
    b, o = patching_config.bo
    bmo = b - o
    for blk_y in range(bmo, texture_map.shape[0] - b + 1, bmo):
        block_idx = np.s_[blk_y:blk_y + b, :b]
        process_block(texture_map, seams_map, lookup_textures, block_idx,
                      find_patch_below, get_seam_patched_vertical, patching_config, rng, uicd)

        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            block_idx = np.s_[blk_y:blk_y + b, blk_x:blk_x + b]
            process_block(texture_map, seams_map, lookup_textures, block_idx,
                          find_patch_both, get_seam_patched_both, patching_config, rng, uicd)
    return texture_map, seams_map


# endregion     _____ AUXILIARY_METHODS _____


# region    _____ PARALLEL IMPLEMENTATION _____

def _parallel_generate_texture_step_predictor(patching_config: SquarePatchingConfig, out_h: NumPixels, out_w: NumPixels):
    quad_row_width = out_w / 2
    quad_column_height = out_h / 2
    block_size = patching_config.block_size
    overlap = patching_config.overlap
    cols_per_quad = int(ceil((quad_row_width - block_size) / (block_size - overlap)))  # w/o the last full block
    rows_per_quad = int(ceil((quad_column_height - block_size) / (block_size - overlap)))
    return 9 + cols_per_quad * rows_per_quad * 4 + (cols_per_quad-1)*2+(rows_per_quad-1)*2


@step_predictor(_parallel_generate_texture_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def generate_texture_parallel(
        src_textures: list[np.ndarray], patching_config: SquarePatchingConfig,
        out_h: NumPixels, out_w: NumPixels,
        nps: int, seed: int,
        uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param out_h: output's height in pixels
    :param out_w: output's width in pixels
    :param nps: number of parallel stripes; tells how many jobs to use for each of the 4 sections.
    :param uicd: utility to track the generation progress and to process UI interrupts
    """
    assert out_w > patching_config.block_size + 2 * (patching_config.block_size - patching_config.overlap)
    assert out_h > patching_config.block_size + 2 * (patching_config.block_size - patching_config.overlap)
    assert patching_config.overlap <= patching_config.block_size // 2

    block_size, overlap = patching_config.block_size, patching_config.overlap
    seeds = iter(SeedSequence(seed).spawn(1 + 4*2)) # 1 + 1 per stripe & quad
    _rng = np.random.default_rng(seed=next(seeds))

    # Get flipped textures to build the quadrants
    hi_ltxts: list[np.ndarray] = []
    vi_ltxts: list[np.ndarray] = []
    vhi_ltxts: list[np.ndarray] = []
    for ltxt in src_textures:
        hi = np.fliplr(ltxt)
        hi_ltxts.append(hi)
        vi_ltxts.append(np.flipud(ltxt))
        vhi_ltxts.append(np.flipud(hi))
    # note: inverted both ways is computed later when filling the top-left quadrant/section

    # check if lists match; if so there is no need to use a separate SharedTextureList
    src_txt_ids: set[int] = {quick_checksum(txt) for txt in src_textures}
    hi_txt_ids: set[int] = {quick_checksum(txt) for txt in hi_ltxts}
    vi_txt_ids: set[int] = {quick_checksum(txt) for txt in vi_ltxts}
    vhi_txt_ids: set[int] = {quick_checksum(txt) for txt in vhi_ltxts}
    hi_eq_src = src_txt_ids == hi_txt_ids
    vi_eq_src = src_txt_ids == vi_txt_ids
    vhi_eq_src = src_txt_ids == vhi_txt_ids
    del src_txt_ids, hi_txt_ids, vi_txt_ids, vhi_txt_ids

    # init stuff that will have to be released eventually
    parallel = Parallel(n_jobs=4, backend="loky", timeout=None, verbose=0)
    shm_ltxts = SharedTextureList.from_list(src_textures, patching_config.get_patch_kernel())
    shm_hi_ltxts = shm_ltxts if hi_eq_src else SharedTextureList.from_list(hi_ltxts, patching_config.get_patch_kernel())
    shm_vi_ltxts = shm_ltxts if vi_eq_src else SharedTextureList.from_list(vi_ltxts, patching_config.get_patch_kernel())
    shm_vhi_ltxts = shm_ltxts if vhi_eq_src else SharedTextureList.from_list(vhi_ltxts, patching_config.get_patch_kernel())

    try:
        # auxiliary variables
        quad_row_width = out_w / 2
        quad_column_height = out_h / 2
        cols_per_quad = int(ceil((quad_row_width - block_size) / (block_size - overlap)))  # w/o the last full block
        rows_per_quad = int(ceil((quad_column_height - block_size) / (block_size - overlap)))
        # might get 1 more row or column than needed here

        # Patch the central area containing the central block shared by in all the stripes
        gen_res = _generate_texture(
            shm_ltxts,
            patching_config,
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
        args = [
            (shm_ltxts, hor_start_block, patching_config, cols_per_quad - 1, next(seeds), hor_start_block_seams, uicd),
            (shm_hi_ltxts, hor_inv_start_block, patching_config, cols_per_quad - 1, next(seeds), hor_inv_start_block_seams, uicd),
            (shm_ltxts, ver_start_block, patching_config, rows_per_quad - 1, next(seeds), ver_start_block_seams, uicd),
            (shm_vi_ltxts, ver_inv_start_block, patching_config, rows_per_quad - 1, next(seeds), ver_inv_start_block_seams, uicd)
        ]
        funcs = [_pfill_row, _pfill_row, _pfill_column, _pfill_column]

        stripes = parallel(delayed(funcs[i])(*args[i]) for i in range(4))

        text_stripes, smap_stripes = zip(*stripes)
        hs, his, vs, vis = text_stripes
        sm_hs, sm_his, sm_vs, sm_vis = smap_stripes

        del gen_res, central_patch, cp_seams

        # generate the 4 sections (quadrants)
        args = [
            (vis, his, shm_vhi_ltxts, rows_per_quad, cols_per_quad, patching_config, nps, next(seeds), sm_vis, sm_his, uicd),
            (vis, hs, shm_vi_ltxts, rows_per_quad, cols_per_quad, patching_config, nps, next(seeds), sm_vis, sm_hs, uicd),
            (vs, hs, shm_ltxts, rows_per_quad, cols_per_quad, patching_config, nps, next(seeds), sm_vs, sm_hs, uicd),
            (vs, his, shm_hi_ltxts, rows_per_quad, cols_per_quad, patching_config, nps, next(seeds), sm_vs, sm_his, uicd)
        ]
        funcs = [_quad1, _quad2, _quad3, _quad4]
        quads = parallel(delayed(funcs[i])(*args[i]) for i in range(4))

        text_quads, smap_quads = zip(*quads)
        tq1, tq2, tq3, tq4 = text_quads
        sq1, sq2, sq3, sq4 = smap_quads

        texture = np.empty((tq1.shape[0] * 2 - block_size, tq1.shape[1] * 2 - block_size, shm_ltxts.global_channel_count),
                           dtype=shm_ltxts.global_dtype)  # empty here is fine, it won't be passed to any func that computes stuff
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

        # fix potential overvalues due to seams overlap ( technically unneeded; but works as a safety measure)
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


def _run_fill_quad(rows, columns, patching_configs, texture, seams_map,
                   ltxts: SharedTextureList,
                   p_strips, seed: SeedSequence, shm_text: SharedMemory | None, shm_smap: SharedMemory | None,
                   uicd: UiCoordData | None):
    """
    Call fill_quad_ps (when using SHM for p_strips > 1) or _pfill_quad_inplace (when not).
    :return: (texture, seams_map)
    """
    if p_strips > 1:
        _pfill_quad_ps(rows, columns, patching_configs, shm_text.name, shm_smap.name, ltxts, p_strips, seed, uicd)
    else:  # if p_strips == 1
        _pfill_quad_inplace(patching_configs, texture, seams_map, ltxts, seed, uicd)
    return texture, seams_map

# endregion     sub-routines for quadX functions


def _quad1(vis, his,
           vhi_ltxts: SharedTextureList,
           rows: int, columns: int, patching_configs: SquarePatchingConfig, p_strips, seed: SeedSequence,
           sm_vis: np.ndarray, sm_his: np.ndarray,
           uicd: UiCoordData | None):
    """
    :param vis: vertical inverted stripe
    :param his: horizontal inverted stripe
    :param sm_vis: vertical inverted seams map of the computed stripe
    :param sm_his: horizontal inverted seams map of the computed stripe
    :param vhi_ltxts: vertical horizontal inverted lookup textures
    :param rows: number of blocks to compute per row
    :param columns: number of blocks to compute per columns
    :param p_strips: number of processes for building the quadrant (not accounting for the other quadrants)
    """
    vi_hi_s = np.ascontiguousarray(np.flipud(his))  # vertical inversion of the horizontal inverted stripe
    hi_vi_s = np.ascontiguousarray(np.fliplr(vis))

    sm_vi_hi_s = np.ascontiguousarray(np.flipud(sm_his))
    sm_hi_vi_s = np.ascontiguousarray(np.fliplr(sm_vis))

    h, w = vis.shape[0], his.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(h, w, vis.shape[2], vis.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        # place stripes & seams
        texture[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = vi_hi_s[:, :]
        texture[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = hi_vi_s[vi_hi_s.shape[0]:, :]
        seams_map[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = sm_vi_hi_s[:, :]
        seams_map[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = sm_hi_vi_s[vi_hi_s.shape[0]:, :]

        # call fill
        texture, seams_map = _run_fill_quad(
            rows, columns, patching_configs,
            texture, seams_map, vhi_ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 0)
        )

        # make copies and apply final flips before SHM is unlinked when leaving the context
        texture_out = np.ascontiguousarray(np.flip(texture, axis=(0, 1)))
        seams_map_out = np.ascontiguousarray(np.flip(seams_map, axis=(0, 1)))

    return texture_out, seams_map_out


def _quad2(vis, hs,
           vi_ltxts: SharedTextureList,
           rows: int, columns: int, patching_configs: SquarePatchingConfig, p_strips, seed: SeedSequence,
           sm_vis: np.ndarray, sm_hs: np.ndarray,
           uicd: UiCoordData | None):
    vi_hs = np.ascontiguousarray(np.flipud(hs))  # flip the stripe
    sm_vi_hs = np.ascontiguousarray(np.flipud(sm_hs))

    h, w = vis.shape[0], hs.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(h, w, vis.shape[2], vis.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:hs.shape[0], :hs.shape[1]] = vi_hs
        texture[hs.shape[0]:vis.shape[0], :vis.shape[1]] = vis[hs.shape[0]:]
        seams_map[:hs.shape[0], :hs.shape[1]] = sm_vi_hs
        seams_map[hs.shape[0]:vis.shape[0], :vis.shape[1]] = sm_vis[hs.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, patching_configs,
            texture, seams_map, vi_ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 1)
        )

        # copy & final flip (flipud)
        texture_out = np.ascontiguousarray(np.flipud(texture))
        seams_map_out = np.ascontiguousarray(np.flipud(seams_map))

    return texture_out, seams_map_out


def _quad4(vs, his,
           hi_ltxts: SharedTextureList,
           rows: int, columns: int, patching_configs: SquarePatchingConfig, p_strips, seed: SeedSequence,
           sm_vs: np.ndarray, sm_his: np.ndarray,
           uicd: UiCoordData | None):
    hi_vs = np.ascontiguousarray(np.fliplr(vs))
    sm_hi_vs = np.ascontiguousarray(np.fliplr(sm_vs))

    h, w = vs.shape[0], his.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(h, w, hi_vs.shape[2], hi_vs.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:his.shape[0], :his.shape[1]] = his
        texture[his.shape[0]:vs.shape[0], :vs.shape[1]] = hi_vs[his.shape[0]:]

        seams_map[:his.shape[0], :his.shape[1]] = sm_his
        seams_map[his.shape[0]:vs.shape[0], :vs.shape[1]] = sm_hi_vs[his.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, patching_configs,
            texture, seams_map, hi_ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 2)
        )

        # copy & final fliplr
        texture_out = np.ascontiguousarray(np.fliplr(texture))
        seams_map_out = np.ascontiguousarray(np.fliplr(seams_map))

    return texture_out, seams_map_out


def _quad3(vs, hs,
           ltxts: SharedTextureList,
           rows: int, columns: int, patching_config: SquarePatchingConfig, p_strips, seed: SeedSequence,
           sm_vs: np.ndarray, sm_hs: np.ndarray,
           uicd: UiCoordData | None):
    h, w = vs.shape[0], hs.shape[1]
    use_shm = (p_strips > 1)

    with _shm_pair(h, w, vs.shape[2], vs.dtype, use_shm) as (texture, seams_map, shm_text, shm_smap):
        texture[:hs.shape[0], :hs.shape[1]] = hs
        texture[hs.shape[0]:vs.shape[0], :vs.shape[1]] = vs[hs.shape[0]:]
        seams_map[:hs.shape[0], :hs.shape[1]] = sm_hs
        seams_map[hs.shape[0]:vs.shape[0], :vs.shape[1]] = sm_vs[hs.shape[0]:]

        texture, seams_map = _run_fill_quad(
            rows, columns, patching_config,
            texture, seams_map, ltxts,
            p_strips, seed, shm_text, shm_smap,
            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 3)
        )

        # quad3: make copies (no flips)
        texture_out = texture.copy()
        seams_map_out = seams_map.copy()

    return texture_out, seams_map_out


# region        parallel cascading stripes


def _pfill_quad_ps(rows, columns, patching_config: SquarePatchingConfig,
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
            patching_config=patching_config, seed=seed,
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
    patching_config: SquarePatchingConfig
    seed: SeedSequence
    columns: int
    rows: int


@handle_ui_interrupts(auto_close=True, re_raise=True)
def _pfill_rows_ps(pid: int, job: _ParaRowsJobInfo, jobs_events: list, uicd: UiCoordData | None):
    patching_config = job.patching_config
    find_patch_both = get_find_patch_both_method()

    # unwrap data
    block_size, overlap = patching_config.bo
    seed = job.seed
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

        for i in range(1 + pid, rows + 1, total_procs):
            coord_list[pid * 2 + 1] = -b_o  # indicates no column yet processed on this row
            coord_list[pid * 2 + 0] = i

            blk_index_i = i * bmo
            rng = np.random.default_rng(child_seed(seed, blk_index_i))

            for j in range(1, columns + 1):
                # if previous row hasn't processed the adjacent section yet wait for it to advance.
                # -1 is used to shortcircuit this check when the job on the prior row has completed all rows.
                while -1 < coord_list[prior_pid * 2 + 0] < i and \
                        coord_list[prior_pid * 2 + 1] - b_o <= j:
                    jobs_events[prior_pid].wait()
                    jobs_events[prior_pid].clear()
                    check_ui(uicd)

                blk_index_j = j * bmo
                block_idx = np.s_[blk_index_i:blk_index_i + block_size, blk_index_j:blk_index_j + block_size]
                process_block(texture, seams_map, ltxts, block_idx, find_patch_both, get_seam_patched_both,
                              patching_config, rng, uicd)

                coord_list[pid * 2 + 1] = j
                jobs_events[pid].set()
    finally:
        # note: this finally block is not "redundant" with respect to "safety";
        #       the .wait() will get stuck if .set() is never called due to an interrupt (or some other) exception.
        coord_list[pid * 2 + 0] = -1  # set job as completed
        jobs_events[pid].set()


# endregion     parallel cascading stripes

# endregion     _____ PARALLEL IMPLEMENTATION _____


# region    _____ NON-PARALLEL IMPLEMENTATIONS _____

def _generate_texture_step_predictor(patching_config: SquarePatchingConfig, out_h: NumPixels, out_w: NumPixels):
    block_size = patching_config.block_size
    overlap = patching_config.overlap
    n_h = int(ceil((out_h - block_size) / (block_size - overlap)))
    n_w = int(ceil((out_w - block_size) / (block_size - overlap)))
    return (n_h + 1) * (n_w + 1)


def _generate_texture(src_textures: ValidatedTexturesIterator,
                     patching_config: SquarePatchingConfig,
                     out_h: NumPixels, out_w: NumPixels,
                     rng: np.random.Generator,
                     uicd: UiCoordData | None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    block_size, overlap = patching_config.block_size, patching_config.overlap

    n_h = int(ceil((out_h - block_size) / (block_size - overlap)))
    n_w = int(ceil((out_w - block_size) / (block_size - overlap)))

    texture_map = np.zeros(
        ((block_size + n_h * (block_size - overlap)),
         (block_size + n_w * (block_size - overlap)),
         src_textures.global_channel_count), dtype=src_textures.global_dtype)

    seams_map = np.zeros(texture_map.shape[:2], dtype=np.float32)

    # Set starting block
    _, starting_block = _get_random_valid_block(block_size, src_textures, rng)
    texture_map[:block_size, :block_size, :] = starting_block
    check_ui(uicd, to_add=1)

    # --- "Purist" generation: fill only the 1st row and then the rest ---
    _fill_row_inplace(texture_map[:block_size, :], seams_map[:block_size, :], src_textures, patching_config, rng, uicd)
    _fill_quad_purist(patching_config, texture_map, seams_map, src_textures, rng, uicd)

    # fix overvalues due to seams overlap
    np.clip(seams_map, 0, 1, out=seams_map)

    # crop to final size
    return texture_map[:out_h, :out_w], seams_map[:out_h, :out_w]


@step_predictor(_generate_texture_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def generate_texture(src_textures: list[np.ndarray],
                     patching_config: SquarePatchingConfig,
                     out_h: NumPixels, out_w: NumPixels,
                     seed: int,
                     uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param out_h: output's height in pixels
    :param out_w: output's width in pixels
    """
    rng = np.random.default_rng(seed=seed)
    src_textures = TextureList(src_textures, patching_config.get_patch_kernel())
    return _generate_texture(src_textures, patching_config, out_h, out_w, rng, uicd)


@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
def generate_texture_diagonal(src_textures: list[np.ndarray],
                              patching_config: SquarePatchingConfig,
                              out_h: NumPixels, out_w: NumPixels,
                              seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
        I've wondered if iterating diagonally with an overlap sized offset --- so that the corners shared by 4 patches are
        covered by the new patches --- could improve the generation quality, and perhaps help avoid "loose" seams.

        For the most part doesn't seem to make that big of a difference.
    """
    b, o = patching_config.block_size, patching_config.overlap
    assert o * 2 < b

    rng = np.random.default_rng(seed=seed)
    src_textures = TextureList(src_textures, patching_config.get_patch_kernel())

    bm2o = b - 2 * o
    _n_h = n_h = ceil(out_h / bm2o)
    _n_w = n_w = ceil(out_w / bm2o)
    if out_h % bm2o <= o:
        _n_h -= 1
    if out_w % bm2o <= o:
        _n_w -= 1

    n_d = min(_n_h, _n_w)  # number of diagonal iterations

    # select random image for the starting block
    #image = src_textures[rng.integers(len(src_textures))]

    # texture needs to be generated w/ an overlap sized offset due to the auxiliary functions receiving a block sized
    #  patch instead of just the overlap section
    # mind that we need to over-estimate the size of the texture to make room for the adjacent patches to be filled
    # in rows that may not be aligned
    texture_map = np.zeros((o + b * 2 + n_h * bm2o, o + b * 2 + n_w * bm2o, src_textures.global_channel_count),
                           dtype=src_textures.global_dtype)
    seams_map = np.zeros(texture_map.shape[:2], dtype=np.float32)

    # Set starting block
    _, starting_block = _get_random_valid_block(b, src_textures, rng)
    texture_map[o:o + b, o:o + b, :] = starting_block

    # --- generation ----
    d_ixd = o  # texture offset ( both for x & y )

    # fill the 1st row & column
    _fill_row_inplace(texture_map[d_ixd:, d_ixd:], seams_map[d_ixd:, d_ixd:], src_textures, patching_config, rng)
    _fill_column_inplace(texture_map[d_ixd:, d_ixd:], seams_map[d_ixd:, d_ixd:], src_textures, patching_config, rng)

    # fill the rest iterating diagonally
    find_patch = get_find_patch_both_method()
    find_cut = get_seam_patched_both
    for d in range(1, n_d):
        d_ixd += bm2o

        block_idx = np.s_[d_ixd:d_ixd + b, d_ixd:d_ixd + b]
        process_block(texture_map, seams_map, src_textures, block_idx, find_patch, find_cut, patching_config, rng)

        # columns
        for blk_x in range(d_ixd + b - o, texture_map.shape[1] - b + 1, (b - o)):
            block_idx = np.s_[d_ixd:d_ixd + b, blk_x:blk_x + b]
            process_block(texture_map, seams_map, src_textures, block_idx, find_patch, find_cut, patching_config, rng)

        # rows
        for blk_y in range(d_ixd + b - o, texture_map.shape[0] - b + 1, (b - o)):
            block_idx = np.s_[blk_y:blk_y + b, d_ixd:d_ixd + b]
            process_block(texture_map, seams_map, src_textures, block_idx, find_patch, find_cut, patching_config, rng)

    return texture_map[o:o + out_h, o:o + out_w], seams_map[o:o + out_h, o:o + out_w]


# endregion     _____ NON-PARALLEL IMPLEMENTATION _____


# region    _____ PROXY (GUIDED) IMPLEMENTATION _____

@clear_cache_post_exec(*_CACHED_FUNCS)
def _compute_synthesis_map(
        proxy_textures: list[np.ndarray],
        patching_config: SquarePatchingConfig,
        out_h: NumPixels, out_w: NumPixels,
        rng: np.random.Generator,
        uicd: UiCoordData | None) -> tuple[list[PatchIdx], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the data required to obtain the patches from the source textures and the individual patches' masks plus
    the generated proxy textured and its seams map.

    This is returned in the following format:
        1st item: list of (text_idx, y, x) pairs.
        2nd item: ndarray of shape (number of patches, block size, block_size) containing individual patches' masks.
        3rd item: generated proxy texture
        4th item: proxy texture's seams map
    """
    proxy_textures = TextureList(proxy_textures, patching_config.get_patch_kernel())

    b, o = patching_config.block_size, patching_config.overlap
    bmo = b - o

    n_h = int(ceil((out_h - b) / (b - o)))
    n_w = int(ceil((out_w - b) / (b - o)))

    texture_map = np.zeros(
        ((b + n_h * bmo),
         (b + n_w * bmo),
         proxy_textures.global_channel_count), dtype=proxy_textures.global_dtype)

    patch_indices: list[PatchIdx] = []  # text index, row, column
    total_blocks = (n_h + 1) * (n_w + 1)
    patch_masks = np.empty((total_blocks, b, b))
    num_masks_added: int = 0

    def store_patch_mask(mask):
        nonlocal num_masks_added
        patch_masks[num_masks_added] = mask
        num_masks_added += 1

    seams_map = np.zeros(texture_map.shape[:2], dtype=np.float32)

    # Set & Store starting block
    starting_block_idx, starting_block = _get_random_valid_block(b, proxy_textures, rng)
    texture_map[:b, :b, :] = starting_block

    patch_indices.append(starting_block_idx)
    store_patch_mask(np.zeros((b, b), dtype=np.float32))

    def get_block(text_idx, y, x):
        winning_texture = proxy_textures[text_idx]
        return winning_texture[y:y + b, x:x + b]

    def process_block(blk_idx, get_min_cut_patch, ref_block, seams_map_view, patch_idx: PatchIdx):
        patch_indices.append(patch_idx)
        patch_block = get_block(*patch_idx)
        min_cut_patch, patch_weights = get_min_cut_patch(ref_block, patch_block, patching_config)
        store_patch_mask(patch_weights)
        ref_block[:] = min_cut_patch
        seams_map_sub_view = seams_map_view[blk_idx]
        update_seams_map_view(seams_map_sub_view, patch_weights, patching_config.blend_into_patch)
        check_ui(uicd, 1)

    def fill_row_inplace_proxy():
        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            blk_idx = np.s_[:b, blk_x:(blk_x + b)]
            ref_block = texture_map[blk_idx]
            patch_idx = find_patch_vx_idx(
                True, False, False, False,
                ref_block, proxy_textures, patching_config, rng)
            process_block(blk_idx, get_seam_patched_horizontal, ref_block, seams_map, patch_idx)

    def fill_quad_proxy():
        """
            Only requires the first Row already filled.
            A "purist" solution, where there is no apriori column generation.
        """
        for blk_y in range(bmo, texture_map.shape[0] - b + 1 , bmo):
            blk_idx = np.s_[blk_y:blk_y + b, :b]
            ref_block = texture_map[blk_idx]
            patch_idx = find_patch_vx_idx(
                False, False, True, False,
                ref_block, proxy_textures, patching_config, rng)
            process_block(blk_idx, get_seam_patched_vertical, ref_block, seams_map, patch_idx)

            for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
                blk_idx = np.s_[blk_y:blk_y + b, blk_x:blk_x + b]
                ref_block = texture_map[blk_idx]
                patch_idx = find_patch_vx_idx(
                    True, False, True, False,
                    ref_block, proxy_textures, patching_config, rng)
                process_block(blk_idx, get_seam_patched_both, ref_block, seams_map, patch_idx)

        return texture_map, seams_map

    # --- Generate ---
    check_ui(uicd, 1)  # from 1st patch
    fill_row_inplace_proxy()
    fill_quad_proxy()

    np.clip(seams_map, 0, 1, out=seams_map)
    return patch_indices, patch_masks, texture_map[:out_h, :out_w], seams_map[:out_h, :out_w]


def _reconstruct_texture(textures: list[np.ndarray],
                         patches_indices: list[PatchIdx],
                         patches_masks: np.ndarray,
                         out_h: NumPixels, out_w: NumPixels,
                         patching_config: SquarePatchingConfig,
                         uicd: UiCoordData | None
                         ) -> np.ndarray:
    b, o = patching_config.block_size, patching_config.overlap
    bmo = b - o

    n_h = int(ceil((out_h - b) / (b - o)))
    n_w = int(ceil((out_w - b) / (b - o)))

    texture_map = np.empty(
        ((b + n_h * (b - o)),
         (b + n_w * (b - o)),
         textures[0].shape[2]), dtype=textures[0].dtype)

    patch_idx = 0

    # --- Aux Functions ---
    def fetch_masks_at_idx(idx):
        return patches_masks[idx]

    def get_block_at_idx(idx):
        tex_idx, y, x = patches_indices[idx]
        return textures[tex_idx][y:y+b, x:x+b]

    def apply_patch(block_text_idx) -> None:
        nonlocal patch_idx
        patch_idx += 1

        patch_block = get_block_at_idx(patch_idx)
        patch_weights = fetch_masks_at_idx(patch_idx)

        np.subtract(1, patch_weights, out=patch_weights)
        text_view = texture_map[block_text_idx]
        apply_mask(text_view, patch_weights, True)
        np.subtract(1, patch_weights, out=patch_weights)
        text_view += apply_mask(patch_block, patch_weights, False)

        check_ui(uicd, 1)

    def fill_row_inplace():
        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            apply_patch(np.s_[:b, blk_x:blk_x+b])

    def fill_quad_inplace():
        for blk_y in range(bmo, texture_map.shape[0] - b + 1, bmo):
            apply_patch(np.s_[blk_y:blk_y + b, :b])
            for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
                apply_patch(np.s_[blk_y:blk_y + b, blk_x:blk_x + b])

    # --- Generate ---
    texture_map[:b, :b] = get_block_at_idx(0)  # set 1st patch
    check_ui(uicd, 1)
    fill_row_inplace()
    fill_quad_inplace()

    return texture_map[:out_h, :out_w]


def _generate_guided_steps_predictor(patching_config: SquarePatchingConfig, out_h: NumPixels, out_w: NumPixels):
    return 2 * _generate_texture_step_predictor(patching_config=patching_config, out_h=out_h, out_w=out_w)


@step_predictor(_generate_guided_steps_predictor)
@auto_uint8_to_float32
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def generate_guided(
        proxy_textures: list[np.ndarray],
        source_textures: list[np.ndarray],
        patching_config: SquarePatchingConfig,
        out_h: NumPixels, out_w: NumPixels,
        seed: int,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Uses a variant of the source_textures to guide the texture synthesis algorithms and maps the result to the
    source textures.
    Can be used, for example, in noisy images, where a filter can be applied so that the
    generation is not influenced by noise or some other visual artifacts or features.

    :param proxy_textures: textures used to guide the generation;
        the textures order SHOULD MATCH those in the source textures list.
    :param source_textures: textures used to build the final result post proxy synthesis.
    :return:
        item 1: reconstructed synthesis result using the source textures;
        item 2: seams map;
        item 3: synthesis result using the proxy textures.
    """
    # note: ui interrupt exceptions are not caught by _compute_synthesis_map or _reconstruct_texture.
    #       the handle_ui_interrupts wrapper catches the exception here instead, since this is the public method.
    rng = np.random.default_rng(seed=seed)
    patches_idxs, masks, proxy_out_tex, out_cut = _compute_synthesis_map(
        proxy_textures, patching_config, out_h, out_w, rng, uicd)
    out_tex = _reconstruct_texture(
        source_textures, patches_idxs, masks, out_h, out_w, patching_config, uicd)
    return out_tex, out_cut, proxy_out_tex


# endregion _____ PROXY IMPLEMENTATION _____


# region    _____ MAKE SEAMLESS MULTI PATCH _____

def get_numb_of_blocks_to_fill_stripe(block_size, overlap, dim_length):
    return int(ceil((dim_length - block_size) / (block_size - overlap)))


def _seamless_horizontal_multi(image: np.ndarray, patching_config: SquarePatchingConfig, rng: np.random.Generator,
                               lookup_textures: list[np.ndarray] | TextureList = None, seams_map: np.ndarray | None = None,
                               uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray]:
    lookup_textures = [image] if lookup_textures is None else lookup_textures
    if not isinstance(lookup_textures, TextureList):
        lookup_textures = TextureList(lookup_textures, patching_config.get_patch_kernel())
    if seams_map is None:
        seams_map = np.zeros(image.shape[:2], dtype=np.float32)

    block_size, overlap = patching_config.block_size, patching_config.overlap
    bmo = block_size - overlap

    src_h, src_w = image.shape[:2]
    n_h = get_numb_of_blocks_to_fill_stripe(block_size, overlap, src_h)

    texture_map = np.zeros((src_h, src_w, image.shape[-1])).astype(image.dtype)
    texture_map[:] = image

    # center v seam at half block distance of the left corner
    texture_map = np.roll(texture_map, block_size // 2, axis=1)
    ref_block = texture_map[:block_size, :block_size]

    # get 1st patch
    patch_block = find_patch_vx(
        True, True, False, False, ref_block, lookup_textures, patching_config, rng)
    min_cut_patch, patch_weights = get_4way_seam_patched(
        True, True, False, False, ref_block, patch_block, patching_config)

    ref_block[:] = min_cut_patch

    update_seams_map_view(seams_map[:block_size, :block_size], patch_weights, patching_config.blend_into_patch)

    check_ui(uicd, 1)

    for y in range(1, n_h):
        blk_1y = y * bmo  # block top corner y
        blk_2y = blk_1y + block_size  # block bottom corner y

        ref_block = texture_map[blk_1y:blk_2y, :block_size]

        patch_block = find_patch_vx(True, True, True, False, ref_block,
                                    lookup_textures, patching_config, rng)
        min_cut_patch, patch_weights = get_4way_seam_patched(
            True, True, True, False, ref_block, patch_block, patching_config)

        ref_block[:] = min_cut_patch
        update_seams_map_view(seams_map[blk_1y:blk_2y, :block_size], patch_weights, patching_config.blend_into_patch)
        check_ui(uicd, 1)

    # fill last block
    ref_block = texture_map[-block_size:, :block_size]

    patch_block = find_patch_vx(True, True, True, False, ref_block,
                                lookup_textures, patching_config, rng)
    min_cut_patch, patch_weights = get_4way_seam_patched(
        True, True, True, True, ref_block, patch_block, patching_config)
    ref_block[:] = min_cut_patch
    update_seams_map_view(seams_map[-block_size:, :block_size], patch_weights, patching_config.blend_into_patch)
    check_ui(uicd, 1)

    # fix overvalues due to seams overlap
    np.clip(seams_map, 0, 1, out=seams_map)

    return texture_map, seams_map


@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_horizontal_multi(image: np.ndarray, patching_config: SquarePatchingConfig, seed: int,
                              lookup_textures: list[np.ndarray] = None, seams_map: np.ndarray | None = None,
                              uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param image: the image to make seamless; also be used to fetch the patches.
    :param lookup_textures: if provided, the patches will be obtained from the list of "lookup_textures" instead.
    """
    rng = np.random.default_rng(seed=seed)
    return _seamless_horizontal_multi(image, patching_config, rng, lookup_textures, seams_map, uicd)


def _seamless_vertical_multi(image: np.ndarray, patching_config: SquarePatchingConfig, rng: np.random.Generator,
                             lookup_textures: list[np.ndarray] = None, uicd: UiCoordData | None = None):
    texture, seams = _seamless_horizontal_multi(
        np.rot90(image, 1), patching_config, rng=rng, uicd=uicd,
        lookup_textures=None if lookup_textures is None else [np.rot90(txt) for txt in lookup_textures])

    if texture is None:
        return ret_val_on_interrupt
    else:
        # seams should have been already fixed by seamless_horizontal
        return np.rot90(texture, -1).copy(), np.rot90(seams, -1).copy()


@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_vertical_multi(image: np.ndarray, patching_config: SquarePatchingConfig, seed: int,
                            lookup_textures: list[np.ndarray] = None,
                            uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    rng = np.random.default_rng(seed=seed)
    return _seamless_vertical_multi(image, patching_config, rng, lookup_textures, uicd)


@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_both_multi(image, patching_config: SquarePatchingConfig, seed: int,
                        lookup_textures: list[np.ndarray] = None,
                        uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    lookup_textures = [image] if lookup_textures is None else lookup_textures
    block_size = patching_config.block_size

    rng = np.random.default_rng(seed=seed)

    # patch the texture in both directions. the last stripe's endpoints won't loop yet.
    texture, seams = _seamless_vertical_multi(image, patching_config, rng, lookup_textures=lookup_textures, uicd=uicd)

    if texture is None:
        return ret_val_on_interrupt

    # center future seam at stripes interception
    for m in [texture, seams]:
        m[:] = np.roll(m, -block_size // 2, axis=0)

    lookup_textures = TextureList(lookup_textures, patching_config.get_patch_kernel())

    texture, seams = _seamless_horizontal_multi(
        texture, patching_config, rng,
        lookup_textures=lookup_textures,
        seams_map=seams,
        uicd=uicd)

    # center the area to patch 1st, this will make the rolls in the next step easier
    for m in [texture, seams]:
        m[:] = np.roll(m, texture.shape[0] // 2, axis=0)
        m[:] = np.roll(m, (texture.shape[1] - block_size) // 2, axis=1)

    return _patch_horizontal_seam(texture, seams, lookup_textures, patching_config, rng, uicd=uicd)


def _patch_horizontal_seam(texture_to_patch: np.ndarray, seams_map: np.ndarray, lookup_textures: TextureList,
                           patching_config: SquarePatchingConfig, rng: np.random.Generator, uicd: UiCoordData | None = None):
    """Patches the artificial seam at the center of the texture when using make seamless both. """
    block_size, overlap = patching_config.bo

    ys = (texture_to_patch.shape[0] - block_size) // 2
    ye = ys + block_size
    xs = (texture_to_patch.shape[1] - block_size) // 2
    xe = xs + block_size

    # PATCH H SEAM -> LEFT PATCH
    ref_block = texture_to_patch[ys:ye, xs - overlap:xe - overlap]
    patch = find_patch_vx(True, False, True, True, ref_block, lookup_textures, patching_config, rng)
    patch, patch_weights = get_4way_seam_patched(True, False, True, True, ref_block, patch, patching_config)
    ref_block[:] = patch
    update_seams_map_view(seams_map[ys:ye, xs - overlap:xe - overlap], patch_weights, patching_config.blend_into_patch)
    check_ui(uicd, 1)

    # PATCH H SEAM -> RIGHT PATCH
    ref_block = texture_to_patch[ys:ye, xs + overlap:xe + overlap]
    patch = find_patch_vx(True, True, True, True,
                          ref_block, lookup_textures, patching_config, rng)
    patch, patch_weights = get_4way_seam_patched(True, True, True, True,
                                                 ref_block, patch, patching_config)
    ref_block[:] = patch
    update_seams_map_view(seams_map[ys:ye, xs + overlap:xe + overlap], patch_weights, patching_config.blend_into_patch)
    check_ui(uicd, 1)

    np.clip(seams_map, 0, 1, out=seams_map) # fix overvalues
    return texture_to_patch, seams_map


# endregion _____ MAKE SEAMLESS MULTI PATCH _____


# region    _____ MAKE SEAMLESS SINGLE PATCH _____

def _seamless_horizontal_single(image: np.ndarray, lookup_texture: np.ndarray, patching_config: SquarePatchingConfig,
                                rng: np.random.Generator, seams_map=None,
                                uicd: UiCoordData | None = None):
    block_size, overlap = patching_config.bo
    lookup_texture = image if lookup_texture is None else lookup_texture
    image = np.roll(image, +block_size // 2, axis=1)  # move seam to addressable space

    if seams_map is None:
        seams_map = np.zeros(image.shape[:2], dtype=np.float32)

    # left & right overlap errors
    template_method = patching_config.match_template_method
    lo_errs = cv.matchTemplate(image=lookup_texture[:, :-block_size],
                               templ=image[:, :overlap], method=template_method)
    check_ui(uicd, 1)

    ro_errs = cv.matchTemplate(image=np.roll(lookup_texture, -block_size + overlap, axis=1)[:, :-block_size],
                               templ=image[:, block_size - overlap:block_size], method=template_method)
    check_ui(uicd, 1)

    if template_method == cv.TM_CCOEFF_NORMED:
        lo_errs = 1 - lo_errs
        ro_errs = 1 - ro_errs

    err_mat = np.maximum(lo_errs, ro_errs, out=lo_errs)  # could sum instead, this sol. minimizes the worst side
    min_val = np.min(err_mat)  # ignore tolerance in this solution
    y, x = np.nonzero(err_mat <= min_val)  # ignore tolerance here, choose only from the best values
    # still select randomly, it may be the case that there are more than one equally good matches
    # likely super rare, but doesn't costly to keep the option if eventually applicable
    c = rng.integers(len(y))
    y, x = y[c], x[c]

    # "fake" block will only contain the overlap, in order to re-use existing function.
    fake_left_block = np.zeros((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_right_block = np.zeros((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_left_block[:, :overlap] = image[:, :overlap]
    fake_right_block[:, -overlap:] = image[:, block_size - overlap:block_size]
    fake_block_sized_patch = np.zeros((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_block_sized_patch[:, :overlap] = lookup_texture[y:y + image.shape[0], x:x + overlap]
    fake_block_sized_patch[:, -overlap:] = lookup_texture[y:y + image.shape[0], x + block_size - overlap:x + block_size]
    fake_patching_config = dataclasses.replace(patching_config, block_size=image.shape[0])
    left_side_patch , left_weights = get_seam_patched_horizontal(fake_left_block, fake_block_sized_patch, fake_patching_config)
    right_side_patch, right_weights = get_seam_patched_horizontal(
            np.fliplr(fake_right_block),
            np.fliplr(fake_block_sized_patch),
            fake_patching_config
        )
    right_side_patch = np.fliplr(right_side_patch)

    check_ui(uicd, 1)

    # paste vertical stripe patch
    image[:, :block_size] = lookup_texture[y:y + image.shape[0], x:x + block_size]
    image[:, :overlap] = left_side_patch[:, :overlap]
    image[:, block_size - overlap:block_size] = right_side_patch[:, -overlap:]
    update_seams_map_view(seams_map[:, :overlap], left_weights[:, :overlap], patching_config.blend_into_patch)
    update_seams_map_view(seams_map[:, block_size - overlap:block_size], right_weights[:, -overlap:], patching_config.blend_into_patch)

    # fix overvalues due to seams overlap
    np.clip(seams_map, 0, 1, out=seams_map)

    return image, seams_map


def _seamless_vertical_single(image: np.ndarray, lookup_texture: np.ndarray,
                              patching_config: SquarePatchingConfig, rng: np.random.Generator, uicd: UiCoordData | None = None):
    texture, seams = _seamless_horizontal_single(image=np.rot90(image), patching_config=patching_config, rng=rng, uicd=uicd,
                                                 lookup_texture=None if lookup_texture is None else np.rot90(lookup_texture))
    if texture is None:
        return ret_val_on_interrupt
    else:
        # seams should have been already fixed by seamless_horizontal
        return np.rot90(texture, -1).copy(), np.rot90(seams, -1).copy()


@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_horizontal_single(image: np.ndarray, lookup_texture: np.ndarray, patching_config: SquarePatchingConfig,
                               seed: int,
                               uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    rng = np.random.default_rng(seed=seed)
    return _seamless_horizontal_single(image, lookup_texture, patching_config, rng, None, uicd)


@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_vertical_single(image: np.ndarray, lookup_texture: np.ndarray,
                             patching_config: SquarePatchingConfig, seed: int,
                             uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    rng = np.random.default_rng(seed=seed)
    return _seamless_vertical_single(image, lookup_texture, patching_config, rng, uicd)


@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_both_single(image: np.ndarray, lookup_texture: np.ndarray,
                         patching_config: SquarePatchingConfig, seed: int,
                         uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param image: source texture to be made seamless.
    :param lookup_texture: texture from where the patches will be extracted.
    :param patching_config: generation parameters.
    :param seed: seed used for the random number generator (RNG).
    :param uicd: optional element to keep track of the generation or interrupt it.
    :return: the tuple: (texture, seam map)
    """
    lookup_texture = image if lookup_texture is None else lookup_texture
    block_size = patching_config.block_size
    rng = np.random.default_rng(seed=seed)

    texture, seams = _seamless_vertical_single(image, lookup_texture, patching_config, rng, uicd)
    if texture is None:
        return ret_val_on_interrupt
    for m in [texture, seams]:
        m[:] = np.roll(m, -block_size // 2, axis=0)  # center future seam at stripes interception
    texture, seams = _seamless_horizontal_single(texture, lookup_texture, patching_config, rng, seams, uicd)
    if texture is None:
        return ret_val_on_interrupt

    # center seam & patch it
    for m in [texture, seams]:
        m[:] = np.roll(m, texture.shape[0] // 2, axis=0)
        m[:] = np.roll(m, texture.shape[1] // 2 - block_size // 2, axis=1)
    lookup_texture = TextureList([lookup_texture], None)
    texture, seams = _patch_horizontal_seam(texture, seams, lookup_texture, patching_config, rng, uicd)

    # fix overvalues due to seams overlap
    np.clip(seams, 0, 1, out=seams)

    return texture, seams

# endregion _____ MAKE SEAMLESS SINGLE PATCH _____


__all__ = [
    "SquarePatchingConfig",
    "SquarePatchingBlendConfig",
    "SeamsAlgorithm",
    "generate_texture",
    "generate_texture_parallel",
    "generate_texture_diagonal",
    "generate_guided",
    "seamless_horizontal_multi",
    "seamless_vertical_multi",
    "seamless_both_multi",
    "seamless_horizontal_single",
    "seamless_vertical_single",
    "seamless_both_single",
]