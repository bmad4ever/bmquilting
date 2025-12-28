from .synthesis_subroutines import get_4way_min_cut_patch, find_patch_vx, update_seams_map_view
from .misc.ui_coord import UiCoordData, handle_ui_interrupts, check_ui
from .generate import RetOnInterrupt, ret_val_on_interrupt
from .misc.custom_decorators import clear_cache_post_exec
from .types import GenParams
from math import ceil
import numpy as np


def get_numb_of_blocks_to_fill_stripe(block_size, overlap, dim_length):
    return int(ceil((dim_length - block_size) / (block_size - overlap)))


def _make_seamless_horizontally(image: np.ndarray, gen_params: GenParams, rng: np.random.Generator,
                               lookup_textures: list[np.ndarray] = None, seams_map: np.ndarray | None = None,
                               uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray]:
    lookup_textures = [image] if lookup_textures is None else lookup_textures
    if seams_map is None:
        seams_map = np.zeros(image.shape[:2], dtype=np.float32)

    block_size, overlap = gen_params.block_size, gen_params.overlap
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
        True, True, False, False, ref_block, lookup_textures, gen_params, rng)
    min_cut_patch, patch_weights = get_4way_min_cut_patch(
        True, True, False, False, ref_block, patch_block, gen_params)

    ref_block[:] = min_cut_patch

    update_seams_map_view(seams_map[:block_size, :block_size], gen_params, patch_weights)

    check_ui(uicd, 1)

    for y in range(1, n_h):
        blk_1y = y * bmo  # block top corner y
        blk_2y = blk_1y + block_size  # block bottom corner y

        ref_block = texture_map[blk_1y:blk_2y, :block_size]

        patch_block = find_patch_vx(True, True, True, False, ref_block,
                                    lookup_textures, gen_params, rng)
        min_cut_patch, patch_weights = get_4way_min_cut_patch(
            True, True, True, False, ref_block, patch_block, gen_params)

        ref_block[:] = min_cut_patch
        update_seams_map_view(seams_map[blk_1y:blk_2y, :block_size], gen_params, patch_weights)
        check_ui(uicd, 1)

    # fill last block
    ref_block = texture_map[-block_size:, :block_size]

    patch_block = find_patch_vx(True, True, True, False, ref_block,
                                lookup_textures, gen_params, rng)
    min_cut_patch, patch_weights = get_4way_min_cut_patch(
        True, True, True, True, ref_block, patch_block, gen_params)
    ref_block[:] = min_cut_patch
    update_seams_map_view(seams_map[-block_size:, :block_size], gen_params, patch_weights)
    check_ui(uicd, 1)

    # fix overvalues due to seams overlap
    np.clip(seams_map, 0, 1, out=seams_map)

    return texture_map, seams_map


@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def make_seamless_horizontally(image: np.ndarray, gen_params: GenParams, rng: np.random.Generator,
                               lookup_textures: list[np.ndarray] = None, seams_map: np.ndarray | None = None,
                               uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param image: the image to make seamless; also be used to fetch the patches.
    :param lookup_textures: if provided, the patches will be obtained from the list of "lookup_textures" instead.
    """
    return _make_seamless_horizontally(image, gen_params, rng, lookup_textures, seams_map, uicd)


def _make_seamless_vertically(image: np.ndarray, gen_params: GenParams, rng: np.random.Generator,
                             lookup_textures: list[np.ndarray] = None, uicd: UiCoordData | None = None):
    texture, seams = _make_seamless_horizontally(
        np.rot90(image, 1), gen_params, rng=rng, uicd=uicd,
        lookup_textures=None if lookup_textures is None else [np.rot90(txt) for txt in lookup_textures])

    if texture is None:
        return ret_val_on_interrupt
    else:
        # seams should have been already fixed by seamless_horizontal
        return np.rot90(texture, -1).copy(), np.rot90(seams, -1).copy()


@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def make_seamless_vertically(image: np.ndarray, gen_params: GenParams, rng: np.random.Generator,
                             lookup_textures: list[np.ndarray] = None,
                             uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    return _make_seamless_vertically(image, gen_params, rng, lookup_textures, uicd)


@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def make_seamless_both(image, gen_params: GenParams, rng: np.random.Generator,
                       lookup_textures: list[np.ndarray] = None,
                       uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    lookup_textures = [image] if lookup_textures is None else lookup_textures
    block_size = gen_params.block_size

    # patch the texture in both directions. the last stripe's endpoints won't loop yet.
    texture, seams = _make_seamless_vertically(image, gen_params, rng, lookup_textures=lookup_textures, uicd=uicd)
    if texture is not None:
        for m in [texture, seams]:
            m[:] = np.roll(m, -block_size // 2, axis=0)  # center future seam at stripes interception
        texture, seams = _make_seamless_horizontally(
            texture, gen_params, rng,
            lookup_textures=lookup_textures,
            seams_map=seams,
            uicd=uicd)

    if texture is None:
        return ret_val_on_interrupt

    # center the area to patch 1st, this will make the rolls in the next step easier
    for m in [texture, seams]:
        m[:] = np.roll(m, texture.shape[0] // 2, axis=0)
        m[:] = np.roll(m, (texture.shape[1] - block_size) // 2, axis=1)

    return _patch_horizontal_seam(texture, seams, lookup_textures, gen_params, rng, uicd=uicd)


def _patch_horizontal_seam(texture_to_patch: np.ndarray, seams_map: np.ndarray, lookup_textures: list[np.ndarray],
                           gen_params: GenParams, rng: np.random.Generator, uicd: UiCoordData | None = None):
    """Patches the artificial seam at the center of the texture when using make seamless both. """
    block_size, overlap = gen_params.bo

    ys = (texture_to_patch.shape[0] - block_size) // 2
    ye = ys + block_size
    xs = (texture_to_patch.shape[1] - block_size) // 2
    xe = xs + block_size

    # PATCH H SEAM -> LEFT PATCH
    ref_block = texture_to_patch[ys:ye, xs - overlap:xe - overlap]
    patch = find_patch_vx(True, False, True, True, ref_block, lookup_textures, gen_params, rng)
    patch, patch_weights = get_4way_min_cut_patch(True, False, True, True, ref_block, patch, gen_params)
    ref_block[:] = patch
    update_seams_map_view(seams_map[ys:ye, xs - overlap:xe - overlap], gen_params, patch_weights)
    check_ui(uicd, 1)

    # PATCH H SEAM -> RIGHT PATCH
    ref_block = texture_to_patch[ys:ye, xs + overlap:xe + overlap]
    patch = find_patch_vx(True, True, True, True,
                          ref_block, lookup_textures, gen_params, rng)
    patch, patch_weights = get_4way_min_cut_patch(True, True, True, True,
                                                  ref_block, patch, gen_params)
    ref_block[:] = patch
    update_seams_map_view(seams_map[ys:ye, xs + overlap:xe + overlap], gen_params, patch_weights)
    check_ui(uicd, 1)

    np.clip(seams_map, 0, 1, out=seams_map) # fix overvalues
    return texture_to_patch, seams_map
