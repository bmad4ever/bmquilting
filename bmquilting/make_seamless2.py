# An alternative approach to making the texture seamless using a single large patch
from .synthesis_subroutines import compute_errors, get_match_template_method, get_min_cut_patch_horizontal, update_seams_map_view
from .make_seamless import patch_horizontal_seam
from .types import UiCoordData, GenParams
import dataclasses
import numpy as np
import cv2 as cv


RETURN_VALUE_WHEN_INTERRUPTED = (None, None)


def seamless_horizontal(image: np.ndarray, lookup_texture: np.ndarray,
                        gen_args: GenParams, rng, seams_map=None, uicd: UiCoordData | None = None):
    block_size, overlap = gen_args.bo
    lookup_texture = image if lookup_texture is None else lookup_texture
    image = np.roll(image, +block_size // 2, axis=1)  # move seam to addressable space

    if seams_map is None:
        seams_map = np.zeros(image.shape[:2], dtype=np.float32)

    # left & right overlap errors
    template_method = get_match_template_method(gen_args.version)
    lo_errs = cv.matchTemplate(image=lookup_texture[:, :-block_size],
                               templ=image[:, :overlap], method=template_method)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return RETURN_VALUE_WHEN_INTERRUPTED

    ro_errs = cv.matchTemplate(image=np.roll(lookup_texture, -block_size + overlap, axis=1)[:, :-block_size],
                               templ=image[:, block_size - overlap:block_size], method=template_method)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return RETURN_VALUE_WHEN_INTERRUPTED

    err_mat = compute_errors([lo_errs, ro_errs], gen_args.version)
    min_val = np.min(err_mat)  # ignore tolerance in this solution
    y, x = np.nonzero(err_mat <= min_val)  # ignore tolerance here, choose only from the best values
    # still select randomly, it may be the case that there are more than one equally good matches
    # likely super rare, but doesn't costly to keep the option if eventually applicable
    c = rng.integers(len(y))
    y, x = y[c], x[c]

    # "fake" block will only contain the overlap, in order to re-use existing function.
    fake_left_block = np.empty((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_right_block = np.empty((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_left_block[:, -overlap:] = image[:, :overlap]
    fake_right_block[:, :overlap] = image[:, block_size - overlap:block_size]
    fake_block_sized_patch = np.empty((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_block_sized_patch[:, :overlap] = lookup_texture[y:y + image.shape[0], x:x + overlap]
    fake_block_sized_patch[:, -overlap:] = lookup_texture[y:y + image.shape[0], x + block_size - overlap:x + block_size]
    fake_gen_args = dataclasses.replace(gen_args)
    fake_gen_args.block_size = image.shape[0]
    left_side_patch , left_weights = get_min_cut_patch_horizontal(fake_left_block, fake_block_sized_patch, fake_gen_args)
    right_side_patch, right_weights = get_min_cut_patch_horizontal(
            np.fliplr(fake_right_block),
            np.fliplr(fake_block_sized_patch),
            fake_gen_args
        )
    right_side_patch = np.fliplr(right_side_patch)

    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return RETURN_VALUE_WHEN_INTERRUPTED

    # paste vertical stripe patch
    image[:, :block_size] = lookup_texture[y:y + image.shape[0], x:x + block_size]
    image[:, :overlap] = left_side_patch[:, :overlap]
    image[:, block_size - overlap:block_size] = right_side_patch[:, -overlap:]
    update_seams_map_view(seams_map[:, :overlap], gen_args, left_weights[:, :overlap])
    update_seams_map_view(seams_map[:, block_size - overlap:block_size], gen_args, right_weights[:, -overlap:])

    # fix overvalues due to seams overlap
    np.clip(seams_map, 0, 1, out=seams_map)

    return image, seams_map


def seamless_vertical(image: np.ndarray, lookup_texture: np.ndarray,
                      gen_args: GenParams, rng, uicd: UiCoordData | None = None):
    texture, seams = seamless_horizontal(image=np.rot90(image), gen_args=gen_args, rng=rng, uicd=uicd,
                                         lookup_texture=None if lookup_texture is None else np.rot90(lookup_texture))
    if texture is None:
        return RETURN_VALUE_WHEN_INTERRUPTED
    else:
        # seams should have been already fixed by seamless_horizontal
        return np.rot90(texture, -1).copy(), np.rot90(seams, -1).copy()


def seamless_both(image: np.ndarray, lookup_texture: np.ndarray,
                  gen_args: GenParams, rng, uicd: UiCoordData | None = None):
    lookup_texture = image if lookup_texture is None else lookup_texture
    block_size = gen_args.block_size

    texture, seams = seamless_vertical(image, lookup_texture, gen_args, rng, uicd)
    if texture is None:
        return RETURN_VALUE_WHEN_INTERRUPTED
    for m in [texture, seams]:
        m[:] = np.roll(m, -block_size // 2, axis=0)  # center future seam at stripes interception
    texture, seams = seamless_horizontal(texture, lookup_texture, gen_args, rng, seams, uicd)
    if texture is None:
        return RETURN_VALUE_WHEN_INTERRUPTED

    # center seam & patch it
    for m in [texture, seams]:
        m[:] = np.roll(m, texture.shape[0] // 2, axis=0)
        m[:] = np.roll(m, texture.shape[1] // 2 - block_size // 2, axis=1)
    texture, seams = patch_horizontal_seam(texture, seams, [lookup_texture], gen_args, rng, uicd)

    # fix overvalues due to seams overlap
    np.clip(seams, 0, 1, out=seams)

    return texture, seams
