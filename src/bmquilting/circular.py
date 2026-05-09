from __future__ import annotations

from ._internal.circular_subroutines import (
    CircularPatchingConfig, CircularPatchParams, AstarVariant,
    set_random_patch_at_location, process_patch_at_location, get_bbox_idx,

    # methods w/ cache
    _get_annular_mask, _get_circle_mask
)
from ._internal.seams_blur import (
    # methods w/ cache
    _circular_kernel, _get_max_possible_gradient_diff, _get_buffers_for_graddiffs_computation,
)

from ._internal.decorators import clear_cache_post_exec, step_predictor, ndarray_identity_cache, auto_uint8_to_float32
from .utils.ui_coord import UiCoordData, handle_ui_interrupts, check_ui, JobInterrupted
from ._internal.common import (
    NumPixels, PatchIdx, TextureList, Percentage, ValidatedTexturesIterator,
    apply_mask, blend_with_mask,
)
from ._internal.hexagonal_lattice import HexagonalLatticeIterator, Vec2_int
from ._internal.shmem_utils import SharedTextureList
from .utils.texture import curate_for_tex_transfer

from joblib.externals.loky import get_reusable_executor
from multiprocessing.shared_memory import SharedMemory
from numpy.random.bit_generator import SeedSequence
from multiprocessing.managers import ListProxy
from joblib import Parallel, delayed
from multiprocessing import Manager

from collections.abc import Iterable, Callable
import numpy as np
import cv2

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

type RetOnInterrupt = tuple[None, None]
ret_val_on_interrupt = (None, None)

type JobID = int
type ProxyPatch = tuple[PatchIdx, np.ndarray, int, int]
""" lookup texture & patch top-left corner coordinates ; masks ; target x, target y """
type ProxyDataCPHL6S = ListProxy[ListProxy[ProxyPatch]]
""" job id -> list of ProxyPatch """

_CACHED_FUNCS = [
    _get_buffers_for_graddiffs_computation,
    _get_max_possible_gradient_diff,
    _get_annular_mask,
    _circular_kernel,
    _get_circle_mask
]

_INTERP_PAD = {
    cv2.INTER_NEAREST:  1,
    cv2.INTER_LINEAR:   2,
    cv2.INTER_CUBIC:    4,
    cv2.INTER_LANCZOS4: 8,
}


def _get_scale_factor(proxy_textures: list[np.ndarray], source_textures: list[np.ndarray]) -> int:
    if not proxy_textures or not source_textures: return 1

    scales = []
    for pt, st in zip(proxy_textures, source_textures):
        ph, pw = pt.shape[:2]
        sh, sw = st.shape[:2]
        if sh % ph != 0 or sw % pw != 0:
            raise ValueError(f"Texture dimensions are not integer multiples: source({sh},{sw}) proxy({ph},{pw})")
        sh_scale = sh // ph
        sw_scale = sw // pw
        if sh_scale != sw_scale:
            raise ValueError(f"Non-uniform scale factor: h_scale={sh_scale}, w_scale={sw_scale}")
        scales.append(sh_scale)

    if len(set(scales)) > 1:
        raise ValueError("Inconsistent scale factors across texture pairs.")

    return scales[0]


def _periodic_resize(src: np.ndarray, dsize: tuple[int, int], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resizes a periodic image by adding a small wrap-around border to provide
    interpolation context, avoiding edge smearing. Much less data-intensive than 3x3 tiling.
    """
    if src.shape[1] == dsize[0] and src.shape[0] == dsize[1]: return src

    # Use wrap-around padding to provide context for periodic interpolation.
    pad = _INTERP_PAD[interpolation]
    padded = cv2.copyMakeBorder(src, pad, pad, pad, pad, borderType=cv2.BORDER_WRAP)

    # Calculate target padded size based on the synchronized scale
    scale_w = dsize[0] // src.shape[1]
    scale_h = dsize[1] // src.shape[0]
    target_padded_size = (dsize[0] + 2 * pad * scale_w, dsize[1] + 2 * pad * scale_h)

    resized_padded = cv2.resize(padded, target_padded_size, interpolation=interpolation)

    # Extract the center
    off_w, off_h = pad * scale_w, pad * scale_h
    return resized_padded[off_h:off_h + dsize[1], off_w:off_w + dsize[0]].copy(order="C")

def _get_proxy_configs(source_config: CircularPatchingConfig, scale: int) -> tuple[
    CircularPatchingConfig, CircularPatchingConfig]:
    """
    Adjusts the source configuration and creates a proxy configuration to ensure
    perfect grid alignment and avoid spatial drift when using scaled proxy textures.

    :param source_config: The original patching configuration.
    :param scale: The integer scale factor (source_size / proxy_size).
    :return: (adjusted_source_config, proxy_config)
    """
    if scale == 1: return source_config, source_config

    s_pp = source_config.patch_params
    radius = s_pp.radius

    # 1. Adjust radius to be divisible by scale factor
    if radius % scale != 0:
        new_radius = int(round(radius / scale) * scale)
        if new_radius == 0: new_radius = int(scale)
        logger.info(f"Guided Variant: Adjusting source radius from {radius} to {new_radius} "
                    f"to be divisible by scale factor {scale}.")
        radius = new_radius

    # 2. Create proxy patch params
    p_radius = radius // scale
    p_diameter = 2 * p_radius + 1
    p_pp = CircularPatchParams(diameter=p_diameter, overlap_ratio=s_pp.overlap_ratio)

    import dataclasses
    proxy_config = dataclasses.replace(source_config, patch_params=p_pp)

    # 3. Adjust source spacing to match proxy spacing exactly (avoid drift)
    # The proxy run uses p_spacing = round(p_radius * spacing_factor)
    # To avoid drift, the source run MUST use p_spacing * scale
    p_spacing = proxy_config.spacing
    expected_s_spacing = p_spacing * scale
    actual_s_spacing = round(radius * source_config.spacing_factor)

    if actual_s_spacing != expected_s_spacing:
        new_f = expected_s_spacing / radius
        logger.info(f"Guided Variant: Adjusting source spacing_factor from {source_config.spacing_factor} "
                    f"to {new_f} to match proxy grid alignment.")
        adj_source_config = dataclasses.replace(source_config,
                                                patch_params=CircularPatchParams(diameter=2 * radius + 1,
                                                                                 overlap_ratio=s_pp.overlap_ratio),
                                                spacing_factor=new_f)
    else:
        adj_source_config = dataclasses.replace(source_config,
                                                patch_params=CircularPatchParams(diameter=2 * radius + 1,
                                                                                 overlap_ratio=s_pp.overlap_ratio))

    return adj_source_config, proxy_config


def _scale_proxy_data_cphl6s(proxy_data_grouped_by_jid: ProxyDataCPHL6S, scale: int, source_block_size: int) -> None:
    """updates provided proxy_data"""
    if scale != 1:
        for jid in range(len(proxy_data_grouped_by_jid)):
            _scale_proxy_patch_list(proxy_data_grouped_by_jid[jid], scale, source_block_size)


def _scale_proxy_patch_list(proxy_data: list[ProxyPatch] | ListProxy[ProxyPatch], scale: int, source_block_size: int) -> None:
    """updates provided proxy_data"""
    if scale != 1:
        proxy_data[:] = [((idx[0], idx[1] * scale, idx[2] * scale),
             cv2.resize(mask, (source_block_size, source_block_size), interpolation=cv2.INTER_LINEAR),
             tx * scale, ty * scale)
             for idx, mask, tx, ty in proxy_data]


def _paste_fromproxy(target: np.ndarray,
                     proxy_data_item: ProxyPatch, pp: CircularPatchParams,
                     source_textures: SharedTextureList | list[np.ndarray] ) -> None:
    patch_idx, mask, tx, ty = proxy_data_item   # tx, ty -> the patch center coordinate on the target texture

    def _get_patch(textures, idx, block_size) -> np.ndarray:
        texture_index = idx[0]
        py, px = idx[1], idx[2]
        block_coords = np.s_[py:py + block_size, px:px + block_size]
        return textures[texture_index][block_coords]

    patch = _get_patch(source_textures, patch_idx, pp.block_size)
    target_idx = get_bbox_idx(tx, ty, pp)
    region = target[target_idx]
    blend_with_mask(patch, region, mask, out=region)


def _shm_mem_array(shape, dtype: str) -> tuple[SharedMemory, dict]:
    """:return: (shared memory, metadata dictionary)"""
    shm_mem = SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize))
    return shm_mem, {
        "name": shm_mem.name,
        "shape": shape,
        "dtype": dtype,
    }


def _get_extended_size(block_size: NumPixels, size: NumPixels) -> NumPixels:
    """
    Get the extended size for a dimension of the image.
    The extended size is used in order to fill the generation space
    with patches near the boundaries.
    """
    return (size // block_size + 3) * block_size


# region ===== GENERATE 6P STRAT FUNCTIONS =====

def _validate_cphl6p_args(n_processes: int, patching_config: CircularPatchingConfig):
    if n_processes <= 0 or n_processes > 6:
        raise ValueError("n_processes must be in the interval [1, 6]")

    critical_spacing_factor = 1.12
    if patching_config.spacing_factor <= critical_spacing_factor:
        logger.warning(
            f"Spacing factor is less than or equal to {critical_spacing_factor}: "
            "patches from different sections may overlap and changing the number of processes may change the output."
        )


def _generate_cphl6p(
        source_textures: list[np.ndarray],
        out_h: NumPixels, out_w: NumPixels,
        patching_config: CircularPatchingConfig,
        seed: int,
        n_processes: int = 1,
        uicd: UiCoordData | None = None,
        _record: Callable[[JobID, ProxyPatch], None] = lambda jid, pp: None,
        _refill_target: np.ndarray|None=None,
        _refill_seams: np.ndarray|None=None,
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    Circular Patches on a Hexagonal Lattice with 6 Partitions (CPHL6P)

    :param source_textures: Textures used for the generation
    :param out_h: Output texture height
    :param out_w: Output texture width
    :param seed: Used for the random selection of patches within the provided tolerance.
        Be mindful that, despite the provided seed, the generation can only ensure deterministic behavior if the
        sections do not overlap. If the spacing factor is not lower than 1.12, there is potential overlap.
    :param n_processes: Number of processes to run in parallel when filling the stripes and sections.
    :param uicd: (Optional) Keeps track of the generation step
    :param _record: Optional function to store or analyze the processed patches.
        The first argument, JobID, is a number that relates to a stripe and a section in the generation
        ( it is independent of the number of processes used ).
    :return: (Texture, Seams) normally, or (proxy_data, Texture, Seams) when _by_proxy=True,
    """
    _validate_cphl6p_args(n_processes, patching_config)

    pp = patching_config.patch_params
    extended_h, extended_w = _get_extended_size(pp.block_size, out_h), _get_extended_size(pp.block_size, out_w)

    margin_x, margin_y = (extended_w - out_w) // 2, (extended_h - out_h) // 2
    ret_idx = np.s_[margin_y:out_h + margin_y, margin_x:out_w + margin_x]
    lookup_texts = SharedTextureList.from_list(source_textures, patching_config.get_patch_kernel())
    del source_textures  # ignore, from here & use lookup_texts

    texture_shm, texture_meta = _shm_mem_array(
        (extended_h, extended_w, lookup_texts.metadata.global_number_of_channels),
        lookup_texts.metadata.global_dtype)
    seams_shm, seams_meta = _shm_mem_array((extended_h, extended_w), "float32")
    filled_shm, filled_meta = _shm_mem_array((extended_h, extended_w), "float32")

    if _refill_target is not None:
        # Handle Special Option where the generation is done on top of an existing texture
        filled_array = np.ndarray(filled_meta["shape"], filled_meta["dtype"], buffer=filled_shm.buf)
        filled_array[ret_idx] = 1
        texture_array = np.ndarray(texture_meta["shape"], texture_meta["dtype"], buffer=texture_shm.buf)
        texture_array[ret_idx] = _refill_target
        del filled_array, texture_array
    if _refill_seams is not None:
        # Handle Special Option where the generation is done on top of an existing texture
        seams_array = np.ndarray(seams_meta["shape"], seams_meta["dtype"], buffer=seams_shm.buf)
        seams_array[ret_idx] = _refill_seams
        del seams_array


    shm_metadata = {
        "texture": texture_meta,
        "seams": seams_meta,
        "filled": filled_meta,
        "uicd": uicd.jobs_shm_name if uicd is not None else None,
        "record": _record,
    }

    def set_1st_patch(center: Vec2_int, shared_data: dict, job_id, seed):
        shm_items = [SharedMemory(name=shared_data[key]['name']) for key in ["texture", "filled"]]
        rng = np.random.default_rng(seed)

        try:
            np_arrays = [np.ndarray(shared_data[key]['shape'], dtype=shared_data[key]['dtype'], buffer=shm_items[i].buf)
                         for i, key in enumerate(["texture", "filled"])]
            x, y = center
            result = set_random_patch_at_location(np_arrays[0], np_arrays[1], lookup_texts, x, y, patching_config, rng)
            shared_data["record"](job_id, result + (x, y))
            check_ui(uicd, 1)

        except JobInterrupted:
            raise JobInterrupted

        finally:
            for _shm in shm_items:
                _shm.close()

    def process_patches_batch(points_batch: Iterable[Vec2_int], shared_data: dict, job_id: int, seed: SeedSequence):
        rng = np.random.default_rng(seed)

        job_uicd = None
        if shared_data["uicd"] is not None:
            job_uicd = UiCoordData(shared_data["uicd"], job_id % n_processes)

        shm_items = [SharedMemory(name=shared_data[key]['name']) for key in ["texture", "filled", "seams"]]
        try:
            np_arrays = [np.ndarray(shared_data[key]['shape'], dtype=shared_data[key]['dtype'], buffer=shm_items[i].buf)
                         for i, key in enumerate(["texture", "filled", "seams"])]
            for x, y in points_batch:
                result = process_patch_at_location(np_arrays[0], np_arrays[1], np_arrays[2], lookup_texts, x, y,
                                                   patching_config, rng)
                shared_data["record"](job_id, result + (x, y))
                check_ui(job_uicd, 1)

        except JobInterrupted:
            raise JobInterrupted

        finally:
            for _shm in shm_items: _shm.close()
            if job_uicd is not None: job_uicd.close()

    try:
        # Setup lattice iterator & Synthesize
        hexa_iter = HexagonalLatticeIterator(
            min_x=margin_x // 2, min_y=margin_y // 2,
            max_x=out_w + margin_x + margin_x // 2, max_y=out_h + margin_y + margin_y // 2,
            spacing=patching_config.spacing,
            first_patch_func=set_1st_patch,
            process_func=process_patches_batch,
            shared_data=shm_metadata
        )
        hexa_iter.process_spiral(n_processes, SeedSequence(seed))

        # Fetch final texture and seams & Crop result
        texture = np.ndarray(texture_meta['shape'], dtype=texture_meta['dtype'], buffer=texture_shm.buf).copy()
        seams = np.ndarray(seams_meta['shape'], dtype=seams_meta['dtype'], buffer=seams_shm.buf).copy()
        np.clip(seams, 0, 1, out=seams)

        return texture[ret_idx], seams[ret_idx]
    finally:
        get_reusable_executor().shutdown(wait=True)
        for shm in [texture_shm, seams_shm, filled_shm]: shm.close(); shm.unlink()
        lookup_texts.release()


def _reconstruct_texture_cphl6p(source_textures: list[np.ndarray],
                                proxy_data: ProxyDataCPHL6S,
                                out_h: NumPixels, out_w: NumPixels,
                                patching_config: CircularPatchingConfig,
                                n_processes: int,
                                uicd: UiCoordData | None,
                                extended_h: int | None = None,
                                extended_w: int | None = None,
                                margin_y: int | None = None,
                                margin_x: int | None = None
                                ) -> np.ndarray | RetOnInterrupt:
    pp: CircularPatchParams = patching_config.patch_params
    if extended_h is None: extended_h = _get_extended_size(pp.block_size, out_h)
    if extended_w is None: extended_w = _get_extended_size(pp.block_size, out_w)

    if margin_x is None: margin_x = (extended_w - out_w) // 2
    if margin_y is None: margin_y = (extended_h - out_h) // 2
    lookup_texts = SharedTextureList.from_list(source_textures, None)
    del source_textures  # ignore, from here & use lookup_texts

    texture_shm, texture_meta = _shm_mem_array(
        (extended_h, extended_w, lookup_texts.metadata.global_number_of_channels),
        lookup_texts.metadata.global_dtype)

    shm_metadata = {
        "texture": texture_meta,
        "uicd": uicd.jobs_shm_name if uicd is not None else None,
    }

    def _apply_patches_batch(job_id: int, shared_data: dict, start_idx: int = 0):
        patches = proxy_data[job_id]
        if not patches or start_idx >= len(patches):
            return

        job_uicd = None
        if shared_data["uicd"] is not None:
            job_uicd = UiCoordData(shared_data["uicd"], job_id % n_processes)

        shm = SharedMemory(name=shared_data["texture"]['name'])
        shm_out_text = np.ndarray(shared_data["texture"]['shape'], dtype=shared_data["texture"]['dtype'],
                                  buffer=shm.buf)

        try:
            for item in patches[start_idx:]:
                _paste_fromproxy(shm_out_text, item, pp, lookup_texts)
                check_ui(job_uicd, 1)
        except JobInterrupted:
            raise JobInterrupted
        finally:
            shm.close()
            if job_uicd is not None: job_uicd.close()

    try:
        # Step 1: Foundation (Center + Neighbors) are always the first 7 patches in proxy_data[0]
        # We apply them sequentially to establish the starting point.
        patches_list_0 = proxy_data[0]
        foundation_count = min(7, len(patches_list_0))

        if foundation_count > 0:
            shm_out_text_main = np.ndarray(texture_meta['shape'], dtype=texture_meta['dtype'], buffer=texture_shm.buf)
            for item in patches_list_0[:foundation_count]:
                _paste_fromproxy(shm_out_text_main, item, pp, lookup_texts)
                check_ui(uicd, 1)

        # Step 2: Parallel reconstruction for the remaining patches in all 6 jobs.
        # proxy_data[0] continues from index 7, others from index 0.
        parallel = Parallel(n_jobs=n_processes, backend="loky", timeout=None, verbose=0)
        parallel(delayed(_apply_patches_batch)(jid, shm_metadata, 7 if jid == 0 else 0)
                 for jid in range(len(proxy_data)))

        # Fetch final texture & Crop result
        texture = np.ndarray(texture_meta['shape'], dtype=texture_meta['dtype'], buffer=texture_shm.buf).copy()
        ret_idx = np.s_[margin_y:out_h + margin_y, margin_x:out_w + margin_x]
        return texture[ret_idx]
    finally:
        get_reusable_executor().shutdown(wait=True)
        for shm in [texture_shm]: shm.close(); shm.unlink()
        lookup_texts.release()


# region -- step predictor methods --

def _generate_cphl6p_step_predictor(patching_config: CircularPatchingConfig, out_h: NumPixels, out_w: NumPixels):
    pp = patching_config.patch_params
    extended_h, extended_w = _get_extended_size(pp.block_size, out_h), _get_extended_size(pp.block_size, out_w)
    margin_x, margin_y = (extended_w - out_w) // 2, (extended_h - out_h) // 2

    hexa_iter = HexagonalLatticeIterator(
        min_x=margin_x // 2, min_y=margin_y // 2,
        max_x=out_w + margin_x + margin_x // 2, max_y=out_h + margin_y + margin_y // 2,
        spacing=patching_config.spacing,
    )

    return sum(1 for inner in hexa_iter.iterate_spiral() for _ in inner)


def _refill_cphl6p_step_predictor(target: np.ndarray, patching_config: CircularPatchingConfig):
    return _generate_cphl6p_step_predictor(patching_config, target.shape[0], target.shape[1])

def _refill_cphl6p_recursive_step_predictor(target: np.ndarray, patching_configs: list[CircularPatchingConfig]):
    return _generate_cphl6p_recursive_step_predictor(patching_configs, target.shape[0], target.shape[1])

def _generate_cphl6p_recursive_step_predictor(patching_configs: list[CircularPatchingConfig],
                                              out_h: NumPixels, out_w: NumPixels):
    return sum((_generate_cphl6p_step_predictor(conf, out_h, out_w) for conf in patching_configs))


def _generate_guided_chlp6p_step_predictor(patching_config: CircularPatchingConfig, out_h: int, out_w: int):
    return _generate_cphl6p_step_predictor(patching_config, out_h, out_w) * 2


# endregion -- step predictor methods --


@step_predictor(_generate_cphl6p_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def generate_cphl6p(
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        out_h: NumPixels, out_w: NumPixels,
        seed: int,
        n_processes: int = 1,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    return _generate_cphl6p(source_textures, out_h, out_w, patching_config, seed, n_processes, uicd)


@step_predictor(_refill_cphl6p_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def refill_cphl6p(
        target: np.ndarray,
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int,
        n_processes: int = 1,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    out_h, out_w = target.shape[:2]
    return _generate_cphl6p(source_textures, out_h, out_w, patching_config, seed, n_processes, uicd, _refill_target=target)


@step_predictor(_refill_cphl6p_recursive_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def refill_cphl6p_recursive(
        target: np.ndarray,
        source_textures: list[np.ndarray],
        patching_configs: list[CircularPatchingConfig],
        seed: int,
        n_processes: int = 1,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    out_h, out_w = target.shape[:2]
    tex, seams = target, np.broadcast_to(np.float32(0.0), target.shape[:2])
    for config in patching_configs:
        tex, seams = _generate_cphl6p(source_textures, out_h, out_w, config, seed, n_processes, uicd,
                                      _refill_target=tex, _refill_seams=seams)
    return tex, seams


@step_predictor(_generate_cphl6p_recursive_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def generate_cphl6p_recursive(
        source_textures: list[np.ndarray],
        patching_configs: list[CircularPatchingConfig],
        out_h: NumPixels, out_w: NumPixels,
        seed: int,
        n_processes: int = 1,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    tex, seams = _generate_cphl6p(source_textures, out_h, out_w, patching_configs[0], seed, n_processes, uicd)
    for config in patching_configs[1:]:
        tex, seams = _generate_cphl6p(source_textures, out_h, out_w, config, seed, n_processes, uicd,
                                      _refill_target=tex, _refill_seams=seams)
    return tex, seams


@step_predictor(_generate_guided_chlp6p_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def generate_cphl6p_guided(
        proxy_textures: list[np.ndarray],
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        out_h: NumPixels, out_w: NumPixels,
        seed: int,
        n_processes: int = 1,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Uses a variant of the source_textures to guide the texture synthesis algorithms and maps the result to the
    source textures.
    Can be used, for example, in noisy images, where a filter can be applied so that the
    generation is not influenced by noise or some other visual artifacts or features.

    proxy_textures and source_textures should match in length and their elements dimensions should be scaled
    by the same factor.

    When using proxy textures with a different size than the source, the patching parameters
    (radius and spacing) may be auto-adjusted to ensure perfect grid alignment and prevent spatial drift.

    :param proxy_textures: Textures used to guide the generation.
    :param source_textures: Textures used for the final generation, guided by the proxy generation obtained with proxy_textures.

    Check generate_cphl6p documentation for the remaining arguments.

    :return:
        item 1: reconstructed synthesis result using the source textures;
        item 2: seams map;
        item 3: synthesis result using the proxy textures.
    """

    # note: ui interrupt exceptions are not caught by _compute_synthesis_map or _reconstruct_texture.

    critical_spacing_factor = 1.12
    if patching_config.spacing_factor <= critical_spacing_factor and n_processes > 1:
        logger.warning(
            f"Spacing factor is less than or equal to {critical_spacing_factor} and number of processes is higher than 1."
            "\nDue to potential overlaps, not only is the output not guaranteed to be deterministic, "
            "the reconstruct procedure after the proxy generation may introduce artifacts due to different patch ordering."
        )
    del critical_spacing_factor

    scale = _get_scale_factor(proxy_textures, source_textures)
    adj_source_config, proxy_config = _get_proxy_configs(patching_config, scale)

    if out_h % scale != 0 or out_w % scale != 0:
        raise ValueError(f"Output dimensions ({out_h}, {out_w}) are not divisible by scale {scale}")
    p_out_h, p_out_w = out_h // scale, out_w // scale

    _proxy_manager = Manager()
    # One result slot per POTENTIAL process (job_id); set_1st_patch always writes to slot 0.
    _proxy_results = _proxy_manager.list([_proxy_manager.list() for _ in range(6)])

    def _record(job_id: int, result: tuple):
        _proxy_results[job_id].append(result)

    proxy_out_tex, out_seams = _generate_cphl6p(
        proxy_textures, p_out_h, p_out_w, proxy_config,
        seed, n_processes, uicd,
        _record=_record
    )

    _scale_proxy_data_cphl6s(_proxy_results, scale, adj_source_config.patch_params.block_size)

    # Reconstruct using consistent extended size and margins from proxy run
    p_pp = proxy_config.patch_params
    p_ext_h = _get_extended_size(p_pp.block_size, p_out_h)
    p_ext_w = _get_extended_size(p_pp.block_size, p_out_w)
    p_margin_y = (p_ext_h - p_out_h) // 2
    p_margin_x = (p_ext_w - p_out_w) // 2

    out_tex = _reconstruct_texture_cphl6p(
        source_textures, _proxy_results,
        out_h, out_w, adj_source_config,
        n_processes, uicd,
        extended_h=p_ext_h * scale,
        extended_w=p_ext_w * scale,
        margin_y=p_margin_y * scale,
        margin_x=p_margin_x * scale
    )

    if scale > 1: out_seams = cv2.resize(out_seams, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return out_tex, out_seams, proxy_out_tex


# endregion ===== GENERATE 6P STRAT FUNCTIONS =====


# region ===== FILL FUNCTIONS =====

def _extend4filling_dims(height: int, width: int, pp: CircularPatchParams, only_horizontally: bool = False):
    """:return: extended_height, extended_width, margin_y, margin_x"""
    extended_height = _get_extended_size(height, pp.block_size)
    extended_width = _get_extended_size(width, pp.block_size)
    margin_y = (extended_height - height) // 2 if not only_horizontally else 0
    margin_x = (extended_width - width) // 2
    return extended_height, extended_width, margin_y, margin_x

@ndarray_identity_cache(array_arg_index=0)
def _extend4filling(ndarray: np.ndarray, pp: CircularPatchParams, only_horizontally: bool = False) -> tuple[
    np.ndarray, NumPixels, NumPixels]:
    """:return: extended_mask, margin_y, margin_x"""
    extended_height, extended_width, margin_y, margin_x = _extend4filling_dims(
        ndarray.shape[0], ndarray.shape[1], pp, only_horizontally)

    # note that BORDER_REPLICATE is required here in order to fill holes near the edges
    extended_ndarray = cv2.copyMakeBorder(ndarray, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_REPLICATE)
    return extended_ndarray, margin_y, margin_x


def _fill_cphl_step_predictor(mask: np.ndarray, patching_config: CircularPatchingConfig):
    extended_mask, margin_y, margin_x = _extend4filling(mask, patching_config.patch_params)
    height, width = mask.shape[:2]
    hexa_iter = HexagonalLatticeIterator(
        min_x=margin_x // 2, min_y=margin_y // 2,
        max_x=width + margin_x + margin_x // 2, max_y=height + margin_y + margin_y // 2,
        spacing=patching_config.spacing,
    )
    return sum(1 for _ in hexa_iter.iterate_row_major(extended_mask))


def _fill_cphl(
        target: np.ndarray,
        mask: np.ndarray,
        source_textures: list[np.ndarray] | ValidatedTexturesIterator,
        patching_config: CircularPatchingConfig,
        seed: int | SeedSequence,
        uicd: UiCoordData | None = None,
        _record: Callable[[ProxyPatch], None] = lambda _: None,
        _seams: np.ndarray | None = None,
        _custom_fill: np.ndarray | None = False,
        _broadcast_zero_mask: bool = False,
        _transfer_tex: tuple[ValidatedTexturesIterator, np.ndarray, Percentage] | None = None, # proxies, target, alpha
) -> tuple[np.ndarray, np.ndarray]:
    if target.shape[0] != mask.shape[0] or target.shape[1] != mask.shape[1]:
        raise ValueError("target and mask must have the same size")

    if not isinstance(source_textures, TextureList):
        source_textures = TextureList(source_textures, patching_config.get_patch_kernel())

    pp = patching_config.patch_params
    rng = np.random.default_rng(seed)
    height, width = mask.shape[:2]

    # setup extended target & mask in case the mask has holes near the edges
    extended_holes_mask, margin_y, margin_x = _extend4filling(mask, pp)
    extended_filled_mask = extended_holes_mask.copy()
    ret_idx = np.s_[margin_y:height + margin_y, margin_x:width + margin_x]

    # refill related options
    if _custom_fill is not None: extended_filled_mask[ret_idx] = _custom_fill
    if _broadcast_zero_mask: extended_holes_mask = np.broadcast_to(np.float32(0.0), extended_holes_mask.shape)

    # extend arrays so that patches can be placed at the edges
    extended_target = cv2.copyMakeBorder(target, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_REPLICATE)
    extended_seams = (
        np.zeros_like(extended_holes_mask)
        if _seams is None
        else cv2.copyMakeBorder(_seams, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_CONSTANT)
    )

    # expand mask hole so that patches properly overlap with the fill mask
    # otherwise the patch may fall near the edge of the filled section
    # and barely blend from the generated to the existing section
    spacing = patching_config.spacing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spacing, spacing))
    extended_holes_mask = cv2.erode(extended_holes_mask, kernel, iterations=1)

    # setup iterator
    hexa_iter = HexagonalLatticeIterator(
        min_x=margin_x // 2, min_y=margin_y // 2,
        max_x=width + margin_x + margin_x // 2, max_y=height + margin_y + margin_y // 2,
        spacing=spacing,
    )

    # fill the holes
    for x, y in hexa_iter.iterate_row_major(extended_holes_mask):
        result = process_patch_at_location(
            extended_target, extended_filled_mask, extended_seams,
            source_textures, x, y, patching_config, rng,
            _tex_transfer= None if _transfer_tex is None else\
            (
                _transfer_tex[0],
                cv2.copyMakeBorder(_transfer_tex[1], margin_y, margin_y, margin_x, margin_x, cv2.BORDER_REPLICATE),
                _transfer_tex[2],
            )
        )
        check_ui(uicd, 1)
        _record(result + (x, y))

    return extended_target[ret_idx], extended_seams[ret_idx]


def _reconstruct_fill_cphl(
        target: np.ndarray,
        source_textures: list[np.ndarray],
        proxy_data: list[ProxyPatch],
        patching_config: CircularPatchingConfig,
        uicd: UiCoordData | None = None,
        margin_y: int | None = None,
        margin_x: int | None = None,
) -> np.ndarray:
    pp = patching_config.patch_params

    # setup extended target & mask
    if margin_y is None or margin_x is None:
        h, w = target.shape[:2]
        _, _, def_margin_y, def_margin_x = _extend4filling_dims(h, w, pp)
        margin_y = margin_y if margin_y is not None else def_margin_y
        margin_x = margin_x if margin_x is not None else def_margin_x

    extended_target = cv2.copyMakeBorder(target, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_REPLICATE)

    # fill the holes using recorded coordinates
    for item in proxy_data:
        # NOTE: tx and ty in proxy_data are already the coordinates relative to the extended target
        _paste_fromproxy(extended_target, item, pp, source_textures)
        check_ui(uicd, 1)

    height, width = target.shape[:2]
    result = extended_target[margin_y:height + margin_y, margin_x:width + margin_x]
    return result


def _guided_fill_cphl_step_predictor(mask, patching_config):
    return _fill_cphl_step_predictor(mask, patching_config) * 2


@step_predictor(_fill_cphl_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(_extend4filling, *_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def fill_cphl(
        target: np.ndarray,
        mask: np.ndarray,
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param mask: Binary mask.
        If provided in float32 format the area to patch should be filled with zeroes, and the remaining area with ones.
        If provided in uint8 format use 255 instead of 1.

    :return: texture, seams
    """
    return _fill_cphl(target, mask, source_textures, patching_config, seed, uicd)


@step_predictor(_guided_fill_cphl_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(_extend4filling, *_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def fill_cphl_guided(
        proxy_target: np.ndarray, target: np.ndarray, mask: np.ndarray,
        proxy_textures: list[np.ndarray], source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int, uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Check generate_cphl6p_guided documentation for more info on the guided approach.

    When using proxy textures with a different size than the source, the patching parameters
    (radius and spacing) may be auto-adjusted to ensure perfect grid alignment and prevent spatial drift.

    :param mask: Binary mask.
        If provided in float32 format the area to patch should be filled with zeroes, and the remaining area with ones.
        If provided in uint8 format use 255 instead of 1.

    :return: texture, seams, synthesised_proxy
    """
    scale = _get_scale_factor(proxy_textures, source_textures)
    adj_source_config, proxy_config = _get_proxy_configs(patching_config, scale)

    if scale > 1:
        ph, pw = proxy_target.shape[:2]
        proxy_mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_LINEAR)
    else:
        proxy_mask = mask

    proxy_data = []

    def record(idx_mask_tuple: ProxyPatch):
        proxy_data.append(idx_mask_tuple)

    proxy_result, seams = _fill_cphl(
        target=proxy_target, mask=proxy_mask, source_textures=proxy_textures,
        patching_config=proxy_config, seed=seed, uicd=uicd, _record=record)

    _scale_proxy_patch_list(proxy_data, scale, adj_source_config.patch_params.block_size)

    # Reconstruct using consistent margins from proxy run
    p_pp = proxy_config.patch_params
    _, p_margin_y, p_margin_x = _extend4filling(mask, p_pp)

    result = _reconstruct_fill_cphl(
        target=target, source_textures=source_textures, proxy_data=proxy_data,
        patching_config=adj_source_config, uicd=uicd,
        margin_y=p_margin_y * scale, margin_x=p_margin_x * scale
    )

    if scale > 1: seams = cv2.resize(seams, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)

    return result, seams, proxy_result


def _refill_cphl_step_predictor(target: np.ndarray, patching_config: CircularPatchingConfig):
    mask = np.broadcast_to(np.float32(0.0), target.shape[:2])
    return _fill_cphl_step_predictor(mask, patching_config)

@step_predictor(_refill_cphl_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(_extend4filling, *_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def refill_cphl(
        target: np.ndarray,
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """:return: texture, seams"""
    mask = np.broadcast_to(np.float32(0.0), target.shape[:2])
    filled = np.broadcast_to(np.float32(1.0), target.shape[:2])
    return _fill_cphl(target, mask, source_textures, patching_config, seed, uicd,
                      _custom_fill=filled, _broadcast_zero_mask=True)


def _refill_cphl_recursive_step_predictor(target: np.ndarray, patching_configs: list[CircularPatchingConfig]):
    mask = np.broadcast_to(np.float32(0.0), target.shape[:2])
    return sum((_fill_cphl_step_predictor(mask, config) for config in patching_configs))

@step_predictor(_refill_cphl_recursive_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(_extend4filling, *_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def refill_cphl_recursive(
        target: np.ndarray,
        source_textures: list[np.ndarray],
        patching_configs: list[CircularPatchingConfig],
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """:return: texture, seams"""
    seams = np.zeros(target.shape[:2], dtype=np.float32)
    tex = target
    mask = np.broadcast_to(np.float32(0.0), target.shape[:2])
    filled = np.broadcast_to(np.float32(1.0), target.shape[:2])
    ss_iter = iter(SeedSequence(seed).spawn(len(patching_configs)))
    for patching_config in patching_configs:
        tex, seams = _fill_cphl(tex, mask, source_textures, patching_config, next(ss_iter), uicd,
                          _seams=seams, _custom_fill=filled, _broadcast_zero_mask=True)
    return tex, seams

# endregion ===== FILL FUNCTIONS =====


# region ==== MORE GENERATE FUNCTIONS ====

def _generate_cphl_step_predictor(out_h: int, out_w: int, patching_config: CircularPatchingConfig):
    mask = np.broadcast_to(np.float32(0.0), (out_h, out_w))
    return _fill_cphl_step_predictor(mask, patching_config)

@step_predictor(_generate_cphl_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(_extend4filling, *_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def generate_cphl(
        lookup_textures: list[np.ndarray],
        out_h: int, out_w: int,
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """:return: texture, seams"""
    texs = TextureList(lookup_textures, patching_config.get_patch_kernel())
    target = np.zeros((out_h, out_w, texs.global_channel_count), dtype=texs.global_dtype)
    mask = np.broadcast_to(np.float32(0.0), target.shape[:2])
    return _fill_cphl(target, mask, texs, patching_config, seed, uicd, _broadcast_zero_mask=True)


def _generate_cphl_recursive_step_predictor(out_h: int, out_w: int, patching_configs: list[CircularPatchingConfig]):
    mask = np.broadcast_to(np.float32(0.0), (out_h, out_w))
    return sum((_fill_cphl_step_predictor(mask, config) for config in patching_configs))

@step_predictor(_generate_cphl_recursive_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(_extend4filling, *_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def generate_cphl_recursive(
        lookup_textures: list[np.ndarray],
        out_h: int, out_w: int,
        patching_configs: list[CircularPatchingConfig],
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """:return: texture, seams"""
    tex_list = TextureList(lookup_textures, patching_configs[0].get_patch_kernel())
    target = np.zeros((out_h, out_w, tex_list.global_channel_count), dtype=tex_list.global_dtype)
    mask = np.broadcast_to(np.float32(0.0), target.shape[:2])
    ss_iter = iter(SeedSequence(seed).spawn(len(patching_configs)))

    tex, seams = _fill_cphl(target, mask, tex_list, patching_configs[0], next(ss_iter), uicd, _broadcast_zero_mask=True)

    filled = np.broadcast_to(np.float32(1.0), target.shape[:2])
    for patching_config in patching_configs[1:]:
        tex, seams = _fill_cphl(tex, mask, tex_list, patching_config, next(ss_iter), uicd,
                          _seams=seams, _custom_fill=filled, _broadcast_zero_mask=True)
    return tex, seams

# endregion ==== MORE GENERATE FUNCTIONS ====


# region ===== MAKE SEAMLESS FUNCTIONS =====

def _fill_hline(
        y: int, x_bounds: tuple[int, int]|None,
        target: np.ndarray, source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int, uicd: UiCoordData | None = None,
        _record: Callable[[ProxyPatch], None] = lambda _: None,
        _seams: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = target.shape[:2]
    pp = patching_config.patch_params
    rng = np.random.default_rng(seed)
    margin_x = pp.block_size if x_bounds is None else 0

    extended_target = cv2.copyMakeBorder(target, 0, 0, margin_x, margin_x, cv2.BORDER_REPLICATE)
    extended_fill_mask = np.ones(extended_target.shape[:2], dtype=np.float32)
    extended_seams = (
        np.zeros_like(extended_fill_mask)
        if _seams is None
        else cv2.copyMakeBorder(_seams, 0, 0, margin_x, margin_x, cv2.BORDER_CONSTANT)
    )

    if x_bounds is None: x_bounds = (pp.radius, extended_target.shape[1]-pp.radius)

    # fill line
    source_textures = TextureList(source_textures, patching_config.get_patch_kernel())
    for x in range(x_bounds[0], x_bounds[1], patching_config.spacing):
        result = process_patch_at_location(
            extended_target, extended_fill_mask, extended_seams,
            source_textures, x, y, patching_config, rng
        )
        check_ui(uicd, 1)
        _record(result + (x, y))

    ret_idx = np.s_[:, margin_x:width + margin_x]
    return extended_target[ret_idx], extended_seams[ret_idx]


def _make_seamless_vertical_circular(
        target: np.ndarray, lookup_textures: list[np.ndarray] | None,
        patching_config: CircularPatchingConfig,
        seed: int | SeedSequence, uicd: UiCoordData | None = None,
        _seams: np.ndarray | None = None,
        _record: Callable[[ProxyPatch], None] = lambda _: None,
) -> tuple[np.ndarray, np.ndarray]:
    """:return: (texture, seams)"""

    if not lookup_textures: lookup_textures = [target]

    row = target.shape[0]//2
    target = np.ascontiguousarray(np.roll(target, row, axis=0))
    return _fill_hline(
        y=row, x_bounds=None,
        target=target, source_textures=lookup_textures,
        patching_config=patching_config,
        seed=seed, uicd=uicd, _seams=_seams, _record=_record
    )

def _make_seamless_horizontal_circular(
        target: np.ndarray, lookup_textures: list[np.ndarray] | None,
        patching_config: CircularPatchingConfig,
        seed: int | SeedSequence, uicd: UiCoordData | None = None,
        _seams: np.ndarray | None = None,
        _record: Callable[[ProxyPatch], None] = lambda _: None,
) -> tuple[np.ndarray, np.ndarray]:
    texture, seams = _make_seamless_vertical_circular(
        # views are copied into C contiguous arrays later; no need to redo the operation here
        target=np.rot90(target), _seams=None if _seams is None else np.rot90(_seams),
        lookup_textures=None if not lookup_textures else [np.rot90(t) for t in lookup_textures],
        patching_config=patching_config,
        seed=seed, uicd=uicd, _record=_record)
    return np.rot90(texture, -1).copy(order="C"), np.rot90(seams, -1).copy(order="C")

def _adjust_seamboth_seams(
        arrays2adjust: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        roll_amount: int | None = None
) -> list[np.ndarray]:
    """
    Slightly move the texture up so that the seams from the vertical patching are not far apart in opposite edges.
    This is done so that any potential following procedure have the seams in a "slightly more usable" layout.
    """
    if roll_amount is None:
        radius = patching_config.patch_params.radius
        roll_amount = round(-radius * 1.1)
    return [np.ascontiguousarray(np.roll(a, roll_amount, axis=0)) for a in arrays2adjust]

def _patch_hseam(
        texture: np.ndarray, seams: np.ndarray,
        lookup_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int | SeedSequence, uicd: UiCoordData | None,
        _record: Callable[[ProxyPatch], None] = lambda _: None,
        y_roll: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    y = y_roll if y_roll is not None else texture.shape[0] // 2
    texture = np.roll(texture, y, axis=0)
    seams = np.roll(seams, y, axis=0)

    space = patching_config.spacing // 2 + 1
    x_center = texture.shape[1] // 2
    x1, x2 = x_center - space, x_center + space * 2

    return _fill_hline(
        y=y, x_bounds=(x1, x2),
        target=texture, source_textures=lookup_textures,
        patching_config=patching_config,
        seed=seed, uicd=uicd,
        _seams=seams, _record=_record
    )

def _guided_make_seamless_vertical(
        proxy_target: np.ndarray, target: np.ndarray,
        proxy_textures: list[np.ndarray] | None, source_textures: list[np.ndarray] | None,
        scale: int,
        adj_source_config: CircularPatchingConfig,
        proxy_config: CircularPatchingConfig,
        seed: int | SeedSequence, uicd: UiCoordData | None = None,
        _proxy_seams: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:param scale: pre-computed source/proxy"""
    source_textures = source_textures if source_textures else [target]
    proxy_textures = proxy_textures if proxy_textures else [proxy_target]

    proxy_data = []
    def record(idx_mask_tuple: ProxyPatch):
        proxy_data.append(idx_mask_tuple)


    # Calculate synchronized roll/center based on proxy scale
    y_proxy = proxy_target.shape[0] // 2
    y_roll = y_proxy * scale

    proxy, p_seams = _make_seamless_vertical_circular(
        proxy_target, proxy_textures,
        proxy_config, seed, uicd,
        _seams=_proxy_seams, _record=record
    )

    _scale_proxy_patch_list(proxy_data, scale, adj_source_config.patch_params.block_size)

    pp = adj_source_config.patch_params
    source_lookup = source_textures if source_textures else [target]
    target_rolled = np.roll(target, y_roll, axis=0)

    # Use scaled margin to match proxy's coordinate system
    s_margin_x = proxy_config.patch_params.block_size * scale
    extended_target = cv2.copyMakeBorder(target_rolled, 0, 0, s_margin_x, s_margin_x, cv2.BORDER_CONSTANT)

    for item in proxy_data:
        _paste_fromproxy(extended_target, item, pp, source_lookup)
        check_ui(uicd, 1)

    ret_idx = np.s_[:, s_margin_x:target.shape[1] + s_margin_x]
    res_texture = extended_target[ret_idx]

    if scale > 1:
        s_seams = _periodic_resize(p_seams, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        s_seams = p_seams

    return res_texture, s_seams, proxy, p_seams

def _guided_make_seamless_horizontal(
        proxy_target: np.ndarray, target: np.ndarray,
        proxy_textures: list[np.ndarray] | None, source_textures: list[np.ndarray] | None,
        scale: int,
        adj_source_config: CircularPatchingConfig,
        proxy_config: CircularPatchingConfig,
        seed: int | SeedSequence, uicd: UiCoordData | None = None,
        _proxy_seams: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    texture, seams, proxy, p_seams = _guided_make_seamless_vertical(
        # arrays are made contiguous further down the line
        proxy_target=np.rot90(proxy_target),
        target=np.rot90(target),
        proxy_textures=None if not proxy_textures else [np.rot90(t) for t in proxy_textures],
        source_textures=None if not source_textures else [np.rot90(t) for t in source_textures],
        scale=scale, adj_source_config=adj_source_config, proxy_config=proxy_config,
        seed=seed, uicd=uicd, _proxy_seams=None if _proxy_seams is None else np.rot90(_proxy_seams)
    )
    return (np.rot90(texture, -1).copy(order="C"), np.rot90(seams, -1).copy(order="C"),
            np.rot90(proxy, -1).copy(order="C"), np.rot90(p_seams, -1).copy(order="C"))


# region  -- step predictor methods --

def _make_seamless_vertical_circular_steps(target: np.ndarray, patching_config: CircularPatchingConfig):
    pp = patching_config.patch_params
    margin = pp.block_size
    extended_target_width = margin * 2 + target.shape[1]
    x_bounds = (pp.radius, extended_target_width-pp.radius)
    return sum(1 for _ in range(x_bounds[0], x_bounds[1], patching_config.spacing))

def _make_seamless_horizontal_circular_steps(target: np.ndarray, patching_config: CircularPatchingConfig):
    return _make_seamless_vertical_circular_steps(np.rot90(target), patching_config)

def _make_seamless_both_circular_steps(target: np.ndarray, patching_config: CircularPatchingConfig):
    return (
        2
        + _make_seamless_vertical_circular_steps(target, patching_config)
        + _make_seamless_horizontal_circular_steps(target, patching_config)
    )

def _guided_make_seamless_vertical_circular_steps(target: np.ndarray, patching_config: CircularPatchingConfig):
    return _make_seamless_vertical_circular_steps(target, patching_config) * 2

def _guided_make_seamless_horizontal_circular_steps(target: np.ndarray, patching_config: CircularPatchingConfig):
    return _make_seamless_horizontal_circular_steps(target, patching_config) * 2

def _guided_make_seameless_both_circular_steps(target: np.ndarray, patching_config: CircularPatchingConfig):
    return _make_seamless_both_circular_steps(target, patching_config) * 2

# endregion  -- step predictor methods --


@step_predictor(_make_seamless_vertical_circular_steps)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def seamless_vertical(
        target: np.ndarray, patching_config: CircularPatchingConfig, seed: int,
        lookup_textures: list[np.ndarray] | None = None,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    return _make_seamless_vertical_circular(target, lookup_textures, patching_config, seed, uicd)


@step_predictor(_make_seamless_horizontal_circular_steps)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def seamless_horizontal(
        target: np.ndarray, patching_config: CircularPatchingConfig, seed: int,
        lookup_textures: list[np.ndarray] | None = None,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    return _make_seamless_horizontal_circular(target, lookup_textures, patching_config, seed, uicd)


@step_predictor(_make_seamless_both_circular_steps)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def seamless_both(
        target: np.ndarray, patching_config: CircularPatchingConfig, seed: int,
        lookup_textures: list[np.ndarray] | None = None,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    lookup_textures = lookup_textures if lookup_textures else [target]

    seed_iterator = iter(SeedSequence(seed).spawn(3))

    texture, seams = _make_seamless_vertical_circular(target, lookup_textures, patching_config, next(seed_iterator), uicd)
    texture, seams = _make_seamless_horizontal_circular(texture, lookup_textures, patching_config, next(seed_iterator), uicd, seams)
    texture, seams = _patch_hseam(texture, seams, lookup_textures, patching_config, next(seed_iterator), uicd)

    texture, seams = _adjust_seamboth_seams([texture, seams], patching_config)
    return texture, seams


@step_predictor(_guided_make_seamless_horizontal_circular_steps)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def seamless_both_guided(
        proxy_target: np.ndarray, target: np.ndarray,
        proxy_textures: list[np.ndarray] | None, source_textures: list[np.ndarray] | None,
        patching_config: CircularPatchingConfig,
        seed: int, uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Check generate_cphl6p_guided documentation for more info on the guided approach.

    When using proxy textures with a different size than the source, the patching parameters
    (radius and spacing) may be auto-adjusted to ensure perfect grid alignment and prevent spatial drift.
    """
    source_textures = source_textures if source_textures else [target]
    proxy_textures = proxy_textures if proxy_textures else [proxy_target]

    seed_iterator = iter(SeedSequence(seed).spawn(3))

    scale = _get_scale_factor(proxy_textures, source_textures)
    adj_source_config, proxy_config = _get_proxy_configs(patching_config, scale)

    # Vertical pass; then Horizontal pass
    texture, seams, proxy, p_seams = _guided_make_seamless_vertical(
        proxy_target, target, proxy_textures, source_textures,
        scale, adj_source_config, proxy_config, next(seed_iterator), uicd)

    texture, seams, proxy, p_seams = _guided_make_seamless_horizontal(
        proxy, texture, proxy_textures, source_textures,
        scale, adj_source_config, proxy_config, next(seed_iterator), uicd, _proxy_seams=p_seams)

    # Calculate synchronized rolls based on proxy scale
    y_proxy = proxy.shape[0] // 2
    y_roll = y_proxy * scale

    # Patch HSEAM (Proxy only)
    proxy_data = []
    def record(idx_mask_tuple: ProxyPatch):
        proxy_data.append(idx_mask_tuple)

    proxy, p_seams = _patch_hseam(proxy, p_seams, proxy_textures, proxy_config, next(seed_iterator), uicd, _record=record, y_roll=y_proxy)

    # Reconstruction
    _scale_proxy_patch_list(proxy_data, scale, adj_source_config.patch_params.block_size)

    # NOTE: texture is already rolled vertically by y_roll from step 1.
    # _patch_hseam rolls it by y_roll AGAIN to perform the horizontal patch line.
    texture = np.roll(texture, y_roll, axis=0)
    pp = adj_source_config.patch_params
    source_lookup = source_textures if source_textures else [target]
    for pd_item in proxy_data:
        _paste_fromproxy(texture, pd_item, pp, source_lookup)
        check_ui(uicd, 1)

    if scale > 1:
        seams = _periodic_resize(p_seams, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        seams = p_seams

    # Final Adjustment
    # Use consistent adjustment roll
    p_adj_roll = round(-proxy_config.patch_params.radius * 1.1)
    s_adj_roll = p_adj_roll * scale

    texture = np.roll(texture, s_adj_roll, axis=0)
    seams = np.roll(seams, s_adj_roll, axis=0)
    proxy = np.roll(proxy, p_adj_roll, axis=0)

    return texture, seams, proxy


@step_predictor(_guided_make_seamless_vertical_circular_steps)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def seamless_vertical_guided(
        proxy_target: np.ndarray, target: np.ndarray,
        patching_config: CircularPatchingConfig,
        seed: int | SeedSequence,
        proxy_textures: list[np.ndarray] | None = None, source_textures: list[np.ndarray] | None = None,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Check generate_cphl6p_guided documentation for more info on the guided approach.

    When using proxy textures with a different size than the source, the patching parameters
    (radius and spacing) may be auto-adjusted to ensure perfect grid alignment and prevent spatial drift.
    """
    scale = _get_scale_factor(proxy_textures, source_textures)
    adj_source_config, proxy_config = _get_proxy_configs(patching_config, scale)
    return _guided_make_seamless_vertical(
        proxy_target, target, proxy_textures, source_textures, scale, adj_source_config, proxy_config, seed, uicd)[:3]


@step_predictor(_guided_make_seamless_horizontal_circular_steps)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def seamless_horizontal_guided(
        proxy_target: np.ndarray, target: np.ndarray,
        patching_config: CircularPatchingConfig,
        seed: int | SeedSequence,
        proxy_textures: list[np.ndarray] | None = None, source_textures: list[np.ndarray] | None = None,
        uicd: UiCoordData | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Check generate_cphl6p_guided documentation for more info on the guided approach.

    When using proxy textures with a different size than the source, the patching parameters
    (radius and spacing) may be auto-adjusted to ensure perfect grid alignment and prevent spatial drift.
    """
    scale = _get_scale_factor(proxy_textures, source_textures)
    adj_source_config, proxy_config = _get_proxy_configs(patching_config, scale)
    return _guided_make_seamless_horizontal(
        proxy_target, target, proxy_textures, source_textures, scale, adj_source_config, proxy_config, seed, uicd)[:3]

# endregion ===== MAKE SEAMLESS FUNCTIONS =====




def _texture_transfer_advanced_step_predictor(curated_target, config_alpha_pairs, target_roi):
    mask = target_roi
    if mask is None: mask = np.broadcast_to(np.float32(0.0), curated_target.shape[:2])
    return sum(_fill_cphl_step_predictor(mask, cfg) for cfg, _ in config_alpha_pairs)


def _texture_transfer_guided_advanced_step_predictor(
        src_textures, proxy_textures, curated_proxy_target, config_alpha_pairs, target_roi):
    mask = target_roi
    if mask is None: mask = np.broadcast_to(np.float32(0.0), curated_proxy_target.shape[:2])

    scale = _get_scale_factor(proxy_textures, src_textures)
    total = 0
    for config, _ in config_alpha_pairs:
        _, proxy_config = _get_proxy_configs(config, scale)
        total += _fill_cphl_step_predictor(mask, proxy_config) * 2  # fill + reconstruction, hence 2x
    return total

def _texture_transfer_step_predictor(src_textures, target, patching_config, alphas, last_diameter, downscale_factor, target_roi):
    config_alpha_pairs = _texture_transfer_auto_config_alpha_pairs(patching_config, alphas, last_diameter)

    if downscale_factor is None or downscale_factor == 1:
        return _texture_transfer_advanced_step_predictor(target, config_alpha_pairs, target_roi)

    # mock data so that prior step predictor can be used without actually resizing stuff
    resized_target_shape = (target.shape[0]//downscale_factor, target.shape[1]//downscale_factor)
    resized_proxy_shape = (src_textures[0].shape[0]//downscale_factor, src_textures[0].shape[1]//downscale_factor)
    curated_proxy_target = np.broadcast_to(np.float32(0.0), resized_target_shape)
    proxy_textures = [np.broadcast_to(np.float32(0.0), resized_proxy_shape)]

    return _texture_transfer_guided_advanced_step_predictor(src_textures, proxy_textures, curated_proxy_target, config_alpha_pairs, target_roi)



def _texture_transfer_advanced(
    src_textures: list[np.ndarray],
    curated_textures: list[np.ndarray],
    curated_target: np.ndarray,
    config_alpha_pairs: list[tuple[CircularPatchingConfig, float]],
    seed: int,
    target_roi: np.ndarray | None = None,
    recompute_valid_area: bool = False,
    uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    :param target_roi: binary mask, where the roi is painted with ones, and the remaining area with zeroes
    """
    if target_roi is None:
        target_roi = np.broadcast_to(np.float32(1.0), curated_target.shape[:2])
        inv_target_roi = np.broadcast_to(np.float32(0.0), curated_target.shape[:2])
    else:
        inv_target_roi = 1 - target_roi

    first_config, first_alpha = config_alpha_pairs[0]
    src_tex_list = TextureList(src_textures, first_config.get_patch_kernel())
    cur_tex_list = TextureList(curated_textures, first_config.get_patch_kernel())

    dst_shape = (curated_target.shape[0], curated_target.shape[1], src_tex_list.global_channel_count)
    tex = np.broadcast_to(np.float32(0.0), dst_shape)
    tex, seams = _fill_cphl(tex, inv_target_roi, src_tex_list, first_config, seed, uicd,
                            _transfer_tex=(cur_tex_list, curated_target, first_alpha))
    filled = np.broadcast_to(np.float32(1.0), target_roi.shape)

    for config, alpha in config_alpha_pairs[1:]:
        if recompute_valid_area:
            src_tex_list = TextureList(src_textures, config.get_patch_kernel())
            cur_tex_list = TextureList(curated_textures, config.get_patch_kernel())
        tex, seams = _fill_cphl(
            tex, inv_target_roi, src_tex_list,
            config, seed, uicd=uicd,
            _seams=seams, _custom_fill=filled,
            _transfer_tex=(cur_tex_list, curated_target, alpha)
        )

    # clear area outside roi
    apply_mask(tex, target_roi, overwrite=True)
    seams *= target_roi

    return tex, seams


def _texture_transfer_auto_config_alpha_pairs(
        config: CircularPatchingConfig,
        alphas:list[float]|None,
        last_diameter:int|None
) -> list[tuple[CircularPatchingConfig, float]]:
    import dataclasses

    if alphas is None: alphas = [.75, .5, .25]
    if last_diameter is None: last_diameter = round(config.patch_params.diameter / 3.6) | 1

    diameters = np.linspace(config.patch_params.diameter, last_diameter, num=len(alphas), dtype=int)
    config_alpha_pairs = []
    for diam, alpha in zip(diameters, alphas):
        new_patch_params = dataclasses.replace(config.patch_params, diameter=(diam|1), overlap_ratio=.5)
        config = dataclasses.replace(config, patch_params=new_patch_params)
        config_alpha_pairs.append((config, alpha))

    return config_alpha_pairs


@step_predictor(_texture_transfer_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def texture_transfer(
    src_textures: list[np.ndarray],
    target: np.ndarray,
    patching_config: CircularPatchingConfig,
    seed: int,
    last_diameter: int | None = None,
    alphas:list[float]|None=None,
    downscale_factor: int | None = None,
    target_roi: np.ndarray | None = None,
    value_range: float = 255.0,
    uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:

    config_alpha_pairs = _texture_transfer_auto_config_alpha_pairs(patching_config, alphas, last_diameter)

    if downscale_factor is not None and downscale_factor > 1:
        resized_target = cv2.resize(target, (target.shape[1]//downscale_factor, target.shape[0]//downscale_factor))
        resized_src_texs = [cv2.resize(t, (t.shape[1]//downscale_factor, t.shape[0]//downscale_factor)) for t in src_textures]
        proxy_textures = [cv2.resize(t, (t.shape[1]//downscale_factor, t.shape[0]//downscale_factor)) for t in src_textures]
        curated_rsz_textures = [curate_for_tex_transfer(t, value_range) for t in resized_src_texs]
        curated_rsz_target = curate_for_tex_transfer(resized_target, value_range)
        cv2.imshow("cr_s", curated_rsz_textures[0])
        cv2.imshow("cr_target", curated_rsz_target)
        cv2.waitKey(0)
        return _texture_transfer_guided_advanced(
            src_textures=src_textures,
            proxy_textures=proxy_textures,
            curated_proxy_textures=curated_rsz_textures,
            curated_proxy_target=curated_rsz_target,
            config_alpha_pairs=config_alpha_pairs,
            seed=seed,
            target_roi=target_roi,
            recompute_valid_area=False,
            uicd=uicd,
        )[:2]

    curated_textures = [curate_for_tex_transfer(t, value_range) for t in src_textures]
    curated_target = curate_for_tex_transfer(target, value_range)
    return _texture_transfer_advanced(
        src_textures, curated_textures, curated_target, config_alpha_pairs, seed, target_roi,
        recompute_valid_area=False, uicd=uicd)


def _texture_transfer_guided_advanced(
    src_textures: list[np.ndarray],
    proxy_textures: list[np.ndarray],
    curated_proxy_textures: list[np.ndarray],
    curated_proxy_target: np.ndarray,
    config_alpha_pairs: list[tuple[CircularPatchingConfig, float]],
    seed: int,
    target_roi: np.ndarray | None = None,  # should be proxy sized
    recompute_valid_area: bool = False,
    uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # DEV NOTE:
    # initially I though of only using the last proxy iteration for reconstructing the texture;
    # however, the reconstruction may have noticeable seams at the edges of a patch,
    # so all the proxy synthesis are recorded and applied.

    if target_roi is None:
        target_roi = np.broadcast_to(np.float32(1.0), curated_proxy_target.shape[:2])
        inv_target_roi = np.broadcast_to(np.float32(0.0), curated_proxy_target.shape[:2])
    else:
        inv_target_roi = 1 - target_roi

    # --- "data" & func used for reconstruction ---
    scale = _get_scale_factor(proxy_textures, src_textures)
    dst_shape = (curated_proxy_target.shape[0]*scale, curated_proxy_target.shape[1]*scale, src_textures[0].shape[2])
    texture = np.broadcast_to(np.float32(0.0), dst_shape)
    mask = cv2.resize(inv_target_roi, dst_shape[:2][::-1])

    proxy_data = []
    def record(idx_mask_tuple: ProxyPatch):
        proxy_data.append(idx_mask_tuple)

    def apply_recording(adj_src_conf: CircularPatchingConfig, proxy_conf:CircularPatchingConfig):
        nonlocal proxy_data, texture, mask, src_textures, scale

        p_pp = proxy_conf.patch_params
        _, p_margin_y, p_margin_x = _extend4filling(inv_target_roi, p_pp)

        # reconstruct using the original texture
        _scale_proxy_patch_list(proxy_data, scale, adj_src_conf.patch_params.block_size)
        texture = _reconstruct_fill_cphl(
            target=texture, source_textures=src_textures, proxy_data=proxy_data,
            patching_config=adj_src_conf, uicd=uicd,
            margin_y=p_margin_y * scale, margin_x=p_margin_x * scale
        )
        proxy_data = []  # clear proxy data for future application


    # --- execute and reconstruct 1st synthesis -----------------------------
    first_config, first_alpha = config_alpha_pairs[0]
    adjust_config, proxy_config = _get_proxy_configs(first_config, scale)
    pxy_tex_list = TextureList(proxy_textures, proxy_config.get_patch_kernel())
    cur_tex_list = TextureList(curated_proxy_textures, proxy_config.get_patch_kernel())

    pxy_dst_shape = (curated_proxy_target.shape[0], curated_proxy_target.shape[1], pxy_tex_list.global_channel_count)
    prx_tex = np.broadcast_to(np.float32(0.0), pxy_dst_shape)

    prx_tex, seams = _fill_cphl(prx_tex, inv_target_roi, pxy_tex_list, proxy_config, seed, uicd,
                            _transfer_tex=(cur_tex_list, curated_proxy_target, first_alpha),
                            _record=record)
    apply_recording(adjust_config, proxy_config)

    # --- execute and reconstruct remaining synthesis -----------------------
    filled = np.broadcast_to(np.float32(1.0), target_roi.shape)
    for config, alpha in config_alpha_pairs[1:]:
        adjust_config, proxy_config = _get_proxy_configs(config, scale)
        if recompute_valid_area:
            pxy_tex_list = TextureList(proxy_textures, proxy_config.get_patch_kernel())
            cur_tex_list = TextureList(curated_proxy_textures, proxy_config.get_patch_kernel())
        prx_tex, seams = _fill_cphl(
            prx_tex, inv_target_roi, pxy_tex_list,
            proxy_config, seed, uicd=uicd,
            _seams=seams, _custom_fill=filled,
            _transfer_tex=(cur_tex_list, curated_proxy_target, alpha),
            _record=record
        )
        apply_recording(adjust_config, proxy_config)

    if scale > 1: seams = cv2.resize(seams, (texture.shape[1], texture.shape[0]), interpolation=cv2.INTER_LINEAR)

    # clear area outside roi
    np.subtract(1, mask, out=mask)
    apply_mask(texture, mask, overwrite=True)
    seams *= mask
    return texture, seams, prx_tex


@step_predictor(_texture_transfer_advanced_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def texture_transfer_advanced(
    src_textures: list[np.ndarray],
    curated_textures: list[np.ndarray],
    curated_target: np.ndarray,
    config_alpha_pairs: list[tuple[CircularPatchingConfig, float]],
    seed: int,
    target_roi: np.ndarray | None = None,
    recompute_valid_area: bool = False,
    uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    :param curated_textures:
        Textures processed in the same way as the target.
        The number of channels should be the same as the target, but can differ from src.
    :param config_alpha_pairs:
        The settings to run in each iteration.
        Typically, both the block sizes and alphas are decreasing.
        Alpha is the weight given to a patch similitude to the target during patch selection:
        if alpha=0, then only the overlap with existing data is used for patch selection;
        if alpha=1, then only the similitude with the target, at the location, is used.
    :param target_roi:
        Mask with the area of interest in the target marked with 1s, and the remaining area with 0s.
    :param recompute_valid_area:
        When using proxies with invalid areas, in order to save on computations
        (and because the block size is expected to decrease, otherwise an invalid patch draw could occur)
        the patch draw-able area is obtained only once using the first configuration in config_alpha_pairs.
        To ignore the default behavior and enforce the recomputation of the invalid areas per configuration,
        set recompute_valid_area to True.

    :return: texture, seams, proxy_texture
    """
    return _texture_transfer_advanced(
        src_textures, curated_textures, curated_target,
        config_alpha_pairs, seed,
        target_roi, recompute_valid_area, uicd
    )

@step_predictor(_texture_transfer_guided_advanced_step_predictor)
@auto_uint8_to_float32
@clear_cache_post_exec(*_CACHED_FUNCS)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def texture_transfer_guided_advanced(
    src_textures: list[np.ndarray],
    proxy_textures: list[np.ndarray],
    curated_proxy_textures: list[np.ndarray],
    curated_proxy_target: np.ndarray,
    config_alpha_pairs: list[tuple[CircularPatchingConfig, float]],
    seed: int,
    target_roi: np.ndarray | None = None,  # should be proxy sized
    recompute_valid_area: bool = False,
    uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    :param curated_proxy_textures:
        Proxy textures processed in the same way as the target.
        The number of channels should be the same as the target, but can differ from the src & proxies.
    :param curated_proxy_target:
        Its resolution should be the same as the proxy textures
    :param config_alpha_pairs:
        The "pseudo" settings to run in each iteration.
        The configs should be set as if they were to run at the src_textures resolution, not at the proxies resolution.
    :param target_roi:
        Mask with the area of interest in the target marked with 1s, and the remaining area with 0s.
        Should have the same size and resolution as curated_proxy_target.

    :return: texture, seams, proxy_texture
    """
    return _texture_transfer_guided_advanced(
        src_textures, proxy_textures, curated_proxy_textures, curated_proxy_target,
        config_alpha_pairs, seed,
        target_roi, recompute_valid_area, uicd
    )



__all__ = [
    "CircularPatchingConfig",
    "AstarVariant",
    "generate_cphl6p",
    "generate_cphl6p_guided",
    "fill_cphl",
    "fill_cphl_guided",
    "seamless_vertical",
    "seamless_horizontal",
    "seamless_both",
    "seamless_both_guided",
    "seamless_vertical_guided",
    "seamless_horizontal_guided",
    "refill_cphl",
    "refill_cphl_recursive",
    "generate_cphl",
    "generate_cphl_recursive"
]