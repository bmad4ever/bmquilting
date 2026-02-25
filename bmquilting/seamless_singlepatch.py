# An alternative approach to making the texture seamless using a single large patch
from .synthesis_subroutines import get_min_cut_patch_horizontal, update_seams_map_view
from .misc.ui_coord import UiCoordData, handle_ui_interrupts, check_ui
from .generate import RetOnInterrupt, ret_val_on_interrupt
from .misc.custom_decorators import clear_cache_post_exec
from .seamless_multipatch import _patch_horizontal_seam
from .types import SquarePatchingConfig
import dataclasses
import numpy as np
import cv2 as cv


def _seamless_horizontal(image: np.ndarray, lookup_texture: np.ndarray, patching_config: SquarePatchingConfig,
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
    fake_patching_config = dataclasses.replace(patching_config)
    fake_patching_config.block_size = image.shape[0]
    left_side_patch , left_weights = get_min_cut_patch_horizontal(fake_left_block, fake_block_sized_patch, fake_patching_config)
    right_side_patch, right_weights = get_min_cut_patch_horizontal(
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


def _seamless_vertical(image: np.ndarray, lookup_texture: np.ndarray,
                       patching_config: SquarePatchingConfig, rng: np.random.Generator, uicd: UiCoordData | None = None):
    texture, seams = _seamless_horizontal(image=np.rot90(image), patching_config=patching_config, rng=rng, uicd=uicd,
                                          lookup_texture=None if lookup_texture is None else np.rot90(lookup_texture))
    if texture is None:
        return ret_val_on_interrupt
    else:
        # seams should have been already fixed by seamless_horizontal
        return np.rot90(texture, -1).copy(), np.rot90(seams, -1).copy()


@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_horizontal(image: np.ndarray, lookup_texture: np.ndarray, patching_config: SquarePatchingConfig,
                        rng: np.random.Generator, seams_map=None,
                        uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    return _seamless_horizontal(image, lookup_texture, patching_config, rng, seams_map, uicd)


@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_vertical(image: np.ndarray, lookup_texture: np.ndarray,
                      patching_config: SquarePatchingConfig, rng: np.random.Generator,
                      uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    return _seamless_vertical(image, lookup_texture, patching_config, rng, uicd)


@clear_cache_post_exec()
@handle_ui_interrupts(return_on_cancel=ret_val_on_interrupt, auto_close=True)
def seamless_both(image: np.ndarray, lookup_texture: np.ndarray,
                  patching_config: SquarePatchingConfig, rng: np.random.Generator,
                  uicd: UiCoordData | None = None) -> tuple[np.ndarray, np.ndarray] | RetOnInterrupt:
    """
    :param image: source texture to be made seamless.
    :param lookup_texture: texture from where the patches will be extracted.
    :param patching_config: generation parameters.
    :param rng: random number generator (RNG).
    :param uicd: optional element to keep track of the generation or interrupt it.
    :return: the tuple: (texture, seam map)
    """
    lookup_texture = image if lookup_texture is None else lookup_texture
    block_size = patching_config.block_size

    texture, seams = _seamless_vertical(image, lookup_texture, patching_config, rng, uicd)
    if texture is None:
        return ret_val_on_interrupt
    for m in [texture, seams]:
        m[:] = np.roll(m, -block_size // 2, axis=0)  # center future seam at stripes interception
    texture, seams = _seamless_horizontal(texture, lookup_texture, patching_config, rng, seams, uicd)
    if texture is None:
        return ret_val_on_interrupt

    # center seam & patch it
    for m in [texture, seams]:
        m[:] = np.roll(m, texture.shape[0] // 2, axis=0)
        m[:] = np.roll(m, texture.shape[1] // 2 - block_size // 2, axis=1)
    texture, seams = _patch_horizontal_seam(texture, seams, [lookup_texture], patching_config, rng, uicd)

    # fix overvalues due to seams overlap
    np.clip(seams, 0, 1, out=seams)

    return texture, seams
