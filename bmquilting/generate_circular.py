from joblib.externals.loky import get_reusable_executor
from multiprocessing.shared_memory import SharedMemory
from numpy.random.bit_generator import SeedSequence
from collections.abc import Iterable, Callable
from multiprocessing import Manager
import numpy as np
import cv2

from .circular_synthesis_subroutines import (
    set_random_patch_at_location, process_patch_at_location, _get_annular_mask, _get_circle_mask, get_bbox_idx)
from .misc.custom_decorators import clear_cache_post_exec, step_predictor, ndarray_identity_cache
from .misc.ui_coord import UiCoordData, handle_ui_interrupts, check_ui, JobInterrupted
from .types import NumPixels, CircularPatchingConfig, PatchIdx, CircularPatchParams
from .seam_smartblur import circular_kernel, get_max_possible_gradient_diff
from .hexa_lattice_iter import HexagonalLatticeIterator, Vec2_int
from .misc.shmem_utils import SharedTextureList
from .misc.dry import blend_with_mask

import logging


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

type RetOnInterrupt = tuple[None, None]
ret_val_on_interrupt = (None, None)

type JobID = int
type ProxyPatch = tuple[PatchIdx, np.ndarray]
type ProxyDataCPHL6S = dict[int, list[ProxyPatch]]
""" job id ->  lookup texture & patch top-left corner coordinates ;  masks """


def _get_patch(textures: list[np.ndarray] | SharedTextureList, idx: PatchIdx, block_size: NumPixels) -> np.ndarray:
    texture_index = idx[0]
    py, px = idx[1], idx[2]
    block_coords = np.s_[py:py + block_size, px:px + block_size]
    return textures[texture_index][block_coords]

def _shm_mem_array(shape, dtype:str) -> tuple[SharedMemory, dict]:
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

def _generate_chlp6p_step_predictor(patching_config: CircularPatchingConfig, out_h: NumPixels, out_w: NumPixels):
    pp = patching_config.patch_params
    extended_h, extended_w = _get_extended_size(pp.block_size, out_h), _get_extended_size(pp.block_size, out_w)
    margin_x, margin_y = (extended_w - out_w) // 2, (extended_h - out_h) // 2

    hexa_iter = HexagonalLatticeIterator(
        min_x=margin_x // 2, min_y=margin_y // 2,
        max_x=out_w + margin_x + margin_x // 2, max_y=out_h + margin_y + margin_y // 2,
        spacing=patching_config.spacing,
    )

    return sum(1 for inner in hexa_iter.iterate_spiral() for _ in inner)

def _validate_cphl6p_args(n_processes: int, patching_config: CircularPatchingConfig):
    if n_processes <= 0 or n_processes > 6:
        raise ValueError("n_processes must be in the interval [1, 6]")

    critical_spacing_factor = 1.12
    if patching_config.spacing_factor <= critical_spacing_factor:
        logger.warning(
            f"Spacing factor is less than or equal to {critical_spacing_factor}: "
            "patches from different sections may overlap and changing the number of processes may change the output."
        )


@step_predictor(_generate_chlp6p_step_predictor)
@clear_cache_post_exec(
    circular_kernel,
    get_max_possible_gradient_diff,
    _get_annular_mask,
    _get_circle_mask
)
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def generate_cphl6p(
        source_textures: list[np.ndarray],
        out_h: NumPixels, out_w: NumPixels,
        patching_config: CircularPatchingConfig,
        seed: int,
        n_processes: int = 1,
        uicd: UiCoordData | None = None,
        _record: Callable[[JobID, ProxyPatch], None] = lambda _: None
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt | tuple[ProxyDataCPHL6S, np.ndarray, np.ndarray]:
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

    margin_x, margin_y = (extended_w-out_w) // 2, (extended_h-out_h) // 2
    lookup_texts = SharedTextureList.from_list(source_textures)
    del source_textures  # ignore, from here & use lookup_texts

    texture_shm, texture_meta = _shm_mem_array(
        (extended_h, extended_w,  lookup_texts.metadata.global_number_of_channels),
        lookup_texts.metadata.global_dtype)
    seams_shm, seams_meta = _shm_mem_array((extended_h, extended_w), "float32")
    filled_shm, filled_meta = _shm_mem_array((extended_h, extended_w), "float32")
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
            shared_data["record"](job_id, result)
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
                result = process_patch_at_location(np_arrays[0], np_arrays[1], np_arrays[2], lookup_texts, x, y, patching_config, rng)
                shared_data["record"](job_id, result)
                check_ui(job_uicd, 1)

        except JobInterrupted:
            raise JobInterrupted

        finally:
            for _shm in shm_items:
                _shm.close()
            if job_uicd is not None:
                job_uicd.close()

    try:
        # Setup lattice iterator & Synthesize
        hexa_iter = HexagonalLatticeIterator(
            min_x=margin_x//2, min_y=margin_y//2,
            max_x=out_w+margin_x+margin_x//2, max_y=out_h+margin_y+margin_y//2,
            spacing = patching_config.spacing,
            first_patch_func=set_1st_patch,
            process_func=process_patches_batch,
            shared_data=shm_metadata
        )
        hexa_iter.process_spiral(n_processes, SeedSequence(seed))

        # Fetch final texture and seams & Crop result
        texture = np.ndarray(texture_meta['shape'], dtype=texture_meta['dtype'], buffer=texture_shm.buf).copy()
        seams = np.ndarray(seams_meta['shape'], dtype=seams_meta['dtype'], buffer=seams_shm.buf).copy()
        np.clip(seams, 0, 1, out=seams)
        ret_idx = np.s_[margin_y:out_h+margin_y, margin_x:out_w+margin_x]

        return texture[ret_idx], seams[ret_idx]
    finally:
        get_reusable_executor().shutdown(wait=True)
        for shm in [texture_shm, seams_shm, filled_shm]:
            shm.close()
            shm.unlink()
        lookup_texts.release()


def _reconstruct_texture_cphl6p(source_textures: list[np.ndarray],
                                proxy_data: ProxyDataCPHL6S,
                                out_h: NumPixels, out_w: NumPixels,
                                patching_config: CircularPatchingConfig,
                                n_processes: int,
                                uicd: UiCoordData | None
                                ) -> np.ndarray | RetOnInterrupt:
    pp: CircularPatchParams = patching_config.patch_params
    block_size = pp.block_size
    extended_h, extended_w = _get_extended_size(pp.block_size, out_h), _get_extended_size(pp.block_size, out_w)

    margin_x, margin_y = (extended_w - out_w) // 2, (extended_h - out_h) // 2
    lookup_texts = SharedTextureList.from_list(source_textures)
    del source_textures  # ignore, from here & use lookup_texts

    texture_shm, texture_meta = _shm_mem_array(
        (extended_h, extended_w, lookup_texts.metadata.global_number_of_channels),
        lookup_texts.metadata.global_dtype)

    counts_shm, counts_meta = _shm_mem_array((6,), "uint32")  # processed patches counts ( for 6 job_ids )

    shm_metadata = {
        "texture": texture_meta,
        "counts": counts_meta,
        "uicd": uicd.jobs_shm_name if uicd is not None else None,
    }

    def _consume(job_id: int, counts: np.ndarray) -> tuple[PatchIdx, np.ndarray]:
        results_list, pos = proxy_data[job_id], counts[job_id]
        counts[job_id] += 1
        return results_list[pos]

    def _apply_patch(shm_out_text: np.ndarray, patch_center: Vec2_int, job_id: int, counts: np.ndarray):
        x, y = patch_center
        patch_idx, mask = _consume(job_id, counts)
        patch = _get_patch(lookup_texts, patch_idx, block_size)
        region_idx = get_bbox_idx(x, y, pp)
        region = shm_out_text[region_idx]
        blend_with_mask(patch, region, mask, out=region)

    def _open_shm(shared_data: dict, item_name: str) -> tuple[SharedMemory, np.ndarray]:
        shm = SharedMemory(name=shared_data[item_name]['name'])
        array = np.ndarray(shared_data[item_name]['shape'], dtype=shared_data[item_name]['dtype'], buffer=shm.buf)
        return shm, array

    def process_patches_batch(points_batch: Iterable[Vec2_int], shared_data: dict, job_id: int, _):
        job_uicd = None
        if shared_data["uicd"] is not None:
            job_uicd = UiCoordData(shared_data["uicd"], job_id % n_processes)

        txt_shm, texture = _open_shm(shared_data, "texture")
        cnt_shm, counts = _open_shm(shared_data, "counts")
        try:
            for point in points_batch:
                _apply_patch(texture, point, job_id, counts)
                check_ui(job_uicd, 1)
        except JobInterrupted:
            raise JobInterrupted
        finally:
            txt_shm.close()
            cnt_shm.close()
            if job_uicd is not None:
                job_uicd.close()

    try:
        # Setup lattice iterator & Synthesize
        hexa_iter = HexagonalLatticeIterator(
            min_x=margin_x // 2, min_y=margin_y // 2,
            max_x=out_w + margin_x + margin_x // 2, max_y=out_h + margin_y + margin_y // 2,
            spacing=patching_config.spacing,
            process_func=process_patches_batch,
            shared_data=shm_metadata
        )
        hexa_iter.process_spiral(n_processes)

        # Fetch final texture and seams & Crop result
        texture = np.ndarray(texture_meta['shape'], dtype=texture_meta['dtype'], buffer=texture_shm.buf).copy()
        ret_idx = np.s_[margin_y:out_h + margin_y, margin_x:out_w + margin_x]
        return texture[ret_idx]
    finally:
        get_reusable_executor().shutdown(wait=True)
        for shm in [texture_shm]:
            shm.close()
            shm.unlink()
        lookup_texts.release()


def _generate_guided_chlp6p_step_predictor(patching_config: CircularPatchingConfig, out_h: int, out_w: int):
    return _generate_chlp6p_step_predictor(patching_config, out_h, out_w) * 2


@step_predictor(_generate_guided_chlp6p_step_predictor)
@clear_cache_post_exec(
    circular_kernel,
    get_max_possible_gradient_diff,
    _get_annular_mask,
    _get_circle_mask
)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def generate_guided_cphl6p(
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

    proxy_textures and source_textures should match in length and in their elements dimensions.

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

    _proxy_manager = Manager()
    # One result slot per POTENTIAL process (job_id); set_1st_patch always writes to slot 0.
    _proxy_results = _proxy_manager.list([_proxy_manager.list() for _ in range(6)])
    def _record(job_id: int, result: tuple):
        _proxy_results[job_id].append(result)

    # remove ui interrup wrapping
    proxy_out_tex, out_seams = generate_cphl6p.__wrapped__.__wrapped__(
        proxy_textures, out_h, out_w, patching_config,
        seed, n_processes, uicd,
        _record=_record
    )
    proxy_data = {i: list(inner_list) for i, inner_list in enumerate(_proxy_results)}

    out_tex = _reconstruct_texture_cphl6p(
        source_textures, proxy_data,
        out_h, out_w, patching_config,
        n_processes, uicd
    )

    return out_tex, out_seams, proxy_out_tex


# region ===== FILL FUNCTIONS =====


@ndarray_identity_cache(0)
def _extend4filling(mask: np.ndarray, pp: CircularPatchParams) -> tuple[np.ndarray, NumPixels, NumPixels]:
    """:return: extended_mask, margin_y, margin_x"""
    height, width = mask.shape[:2]
    extended_height = _get_extended_size(height, pp.block_size)
    extended_width = _get_extended_size(width, pp.block_size)
    margin_y, margin_x = (extended_height - height) // 2, (extended_width - width) // 2

    # note that BORDER_REPLICATE is required here in order to fill holes near the edges
    extended_mask = cv2.copyMakeBorder(mask, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_REPLICATE)
    return extended_mask, margin_y, margin_x


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
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
        _record: Callable[[ProxyPatch], None] = lambda _: None,
        _seams: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if target.shape[0] != mask.shape[0] or target.shape[1] != mask.shape[1]:
        raise ValueError("target and mask must have the same size")

    pp = patching_config.patch_params
    rng = np.random.default_rng(seed)

    # setup extended target & mask in case the mask has holes near the edges
    extended_holes_mask, margin_y, margin_x = _extend4filling(mask, pp)

    extended_target = cv2.copyMakeBorder(target, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_REPLICATE)
    extended_filled_mask = extended_holes_mask.copy()
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
    height, width = mask.shape[:2]
    hexa_iter = HexagonalLatticeIterator(
        min_x=margin_x // 2, min_y=margin_y // 2,
        max_x=width + margin_x + margin_x // 2, max_y=height + margin_y + margin_y // 2,
        spacing=spacing,
    )

    # fill the holes
    for x, y in hexa_iter.iterate_row_major(extended_holes_mask):
        result = process_patch_at_location(
            extended_target, extended_filled_mask, extended_seams,
            source_textures, x, y, patching_config, rng
        )
        check_ui(uicd, 1)
        _record(result)

    ret_idx = np.s_[margin_y:height + margin_y, margin_x:width + margin_x]
    return extended_target[ret_idx], extended_seams[ret_idx]


def _reconstruct_fill_cphl(
        target: np.ndarray,
        mask: np.ndarray,
        source_textures: list[np.ndarray],
        proxy_data: list[ProxyPatch],
        patching_config: CircularPatchingConfig,
        uicd: UiCoordData | None = None,
) -> np.ndarray:
    pp = patching_config.patch_params

    # setup extended target & mask
    extended_holes_mask, margin_y, margin_x = _extend4filling(mask, pp)
    extended_target = cv2.copyMakeBorder(target, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_REPLICATE)

    # setup iterator
    height, width = mask.shape[:2]
    hexa_iter = HexagonalLatticeIterator(
        min_x=margin_x // 2, min_y=margin_y // 2,
        max_x=width + margin_x + margin_x // 2, max_y=height + margin_y + margin_y // 2,
        spacing=patching_config.spacing,
    )

    results_idx = 0
    def get_proxy_patch_data():
        nonlocal results_idx
        result = proxy_data[results_idx]
        results_idx += 1
        return result

    # fill the holes
    for x, y in hexa_iter.iterate_row_major(extended_holes_mask):
        patch_idx, mask = get_proxy_patch_data()
        patch = _get_patch(source_textures, patch_idx, pp.block_size)
        target_idx = get_bbox_idx(x, y, pp)
        region = extended_target[target_idx]
        blend_with_mask(patch, region, mask, out=region)
        check_ui(uicd, 1)

    return extended_target[margin_y:height + margin_y, margin_x:width + margin_x]


def _guided_fill_cphl_step_predictor(mask, patching_config):
    return _fill_cphl_step_predictor(mask, patching_config) * 2


@step_predictor(_fill_cphl_step_predictor)
@clear_cache_post_exec(
    _extend4filling,
    circular_kernel, get_max_possible_gradient_diff, _get_annular_mask, _get_circle_mask
)
@handle_ui_interrupts(return_on_cancel=(None, None), auto_close=True)
def fill_cphl(
        target: np.ndarray,
        mask: np.ndarray,
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """:return: texture, seams"""
    return _fill_cphl(target, mask, source_textures, patching_config, seed, uicd)


@step_predictor(_guided_fill_cphl_step_predictor)
@clear_cache_post_exec(
    _extend4filling,
    circular_kernel, get_max_possible_gradient_diff, _get_annular_mask, _get_circle_mask
)
@handle_ui_interrupts(return_on_cancel=(None, None, None), auto_close=True)
def guided_fill_cphl(
        proxy_target: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        proxy_textures: list[np.ndarray],
        source_textures: list[np.ndarray],
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    proxy_data = []
    def record(idx_mask_tuple: ProxyPatch):
        proxy_data.append(idx_mask_tuple)

    proxy_result, seams = _fill_cphl(
        target=proxy_target, mask=mask, source_textures=proxy_textures,
        patching_config=patching_config, seed=seed, uicd=uicd, _record=record)
    result = _reconstruct_fill_cphl(
        target=target, mask=mask, source_textures=source_textures, proxy_data=proxy_data,
        patching_config=patching_config, uicd=uicd)
    return result, seams, proxy_result

# endregion ===== FILL FUNCTIONS =====


# region ===== MAKE SEAMLESS FUNCTIONS =====

# note: because fill_cphl is re-used here the patches iteration is not optimized; but,
#       this shouldn't be critical. A 1024x1024 w/ 32px sized patches should have ~1k points.

def _make_seamless_vertical_circular(
        target: np.ndarray,
        lookup_textures: list[np.ndarray] | None,
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:return: (texture, seams)"""

    target = target.copy()
    if lookup_textures is None:
        lookup_textures = [target]

    mask = np.ones(target.shape[:2], dtype=np.float32)
    mask[:1, :] = 0
    # mask[-1:, :] = 0
    mask = np.roll(mask, mask.shape[0] // 2, axis=0)
    target = np.roll(target, target.shape[0] // 2, axis=0)
    return _fill_cphl(
        target=target,
        mask=mask,
        source_textures=lookup_textures,
        patching_config=patching_config,
        seed=seed,
        uicd=uicd
    )


def _make_seamless_horizontal_circular(
        target: np.ndarray,
        lookup_textures: list[np.ndarray] | None,
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    texture, seams = _make_seamless_vertical_circular(
        target=np.rot90(target),
        lookup_textures=None if lookup_textures is None else [np.rot90(t) for t in lookup_textures],
        patching_config=patching_config,
        seed=seed, uicd=uicd)

    # if texture is None:
    #    return ret_val_on_interrupt
    # else:
    # seams should have been already fixed by seamless_horizontal
    return np.rot90(texture, -1).copy(), np.rot90(seams, -1).copy()


def make_seamless_both_circular(
        target: np.ndarray,
        lookup_textures: list[np.ndarray] | None,
        patching_config: CircularPatchingConfig,
        seed: int,
        uicd: UiCoordData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO update seams

    texture, _ = _make_seamless_vertical_circular(target, lookup_textures, patching_config, seed, uicd)
    texture, _ = _make_seamless_horizontal_circular(texture, lookup_textures, patching_config, seed, uicd)
    texture = np.roll(texture, texture.shape[0] // 2, axis=0)

    mask = np.ones(target.shape[:2], dtype=np.float32)
    nor = patching_config.patch_params.non_overlap_radius
    x_center = texture.shape[1] // 2
    x1, x2 = x_center - nor, x_center + nor
    mask[texture.shape[0]//2, x1:x2] = 0

    return _fill_cphl(
        target=target,
        mask=mask,
        source_textures=lookup_textures,
        patching_config=patching_config,
        seed=seed,
        uicd=uicd
    )

# endregion ===== MAKE SEAMLESS FUNCTIONS =====
