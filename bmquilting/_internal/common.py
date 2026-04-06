from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from numpy.random import Generator
import numpy as np
import cv2

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


type NumPixels = int
"""the number of pixels (integer)"""

type Percentage = float

type PatchIdx = tuple[int, int, int]
"""patches indexes (texture index, y coord, x coord), where coords are relative to the top-left corner."""

type _2D_Slice = tuple[slice, slice]


FAKE_OUTLIER: int = 4
"""Value used to fill holes in textures."""

class ValidatedTexturesIterator(Protocol):
    def __iter__(self) -> Iterator[np.ndarray]:
        """Should iterate the textures."""
        ...

    def __getitem__(self, index: int) -> np.ndarray:
        """Should return the texture."""
        ...

    def __len__(self) -> int:
        ...

    def has_mask(self, idx: int) -> bool:
        ...

    def get_mask(self, idx:int) -> np.ndarray:
        ...



class TextureList(ValidatedTexturesIterator):
    """
    Similar to a list of textures, but makes the necessary adjustments in order to be used with match template
    in the eventual presence of invalid data which is interpreted as holes in the texture.
    """

    texs: list[np.ndarray]
    masks: list[np.ndarray|None]

    def __init__(self, texs: list[np.ndarray], patch_kernel: np.ndarray):
        """
        :param texs: textures
        :param patch_kernel: mask with the shape of the full patch
        """

        self.texs = []
        self.masks = []
        for idx, tx in enumerate(texs):
            tx_copy, mask = process_invalid_data(tx, patch_kernel)
            self.texs.append(tx_copy)
            self.masks.append(mask)

            if mask is not None:
                logger.info(f"{self.__class__.__name__}: an invalid-points mask was created for texture {idx:02}.")


    def __getitem__(self, index: int) -> np.ndarray:
        return self.texs[index]

    def __len__(self) -> int:
        return len(self.texs)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.texs.__iter__()

    def has_mask(self, index: int) -> bool:
        return self.masks[index] is not None

    def get_mask(self, index: int) -> np.ndarray:
        return self.masks[index]


def process_invalid_data(texture: np.ndarray, patch_kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Checks for invalid data (NaN/Inf) in the texture.
    If found, fills holes with FAKE_OUTLIER and returns a dilated mask.
    :return: (processed_texture, mask or None)
    """
    valid_mask = np.isfinite(texture)
    if np.all(valid_mask):
        return texture, None

    processed_texture = np.copy(texture)
    invalid_mask = np.logical_not(valid_mask, out=valid_mask)
    processed_texture[invalid_mask] = FAKE_OUTLIER

    # dilate invalid mask by block size
    if invalid_mask.ndim == 3:
        invalid_mask = invalid_mask.any(axis=-1)

    invalid_uint8 = np.uint8(invalid_mask)
    cv2.dilate(invalid_uint8, patch_kernel, anchor=(0, 0), dst=invalid_uint8)
    mask = invalid_uint8 > 0

    return processed_texture, mask[:-patch_kernel.shape[0]+1, :-patch_kernel.shape[1]+1]


# region    ----- MASK UTILITIES -----

def apply_mask(src: np.ndarray, mask: np.ndarray, overwrite: bool = False):
    """
    :param src:  image or latent with shape of length 3 (height, width, channels)
    :param mask: mask with shape of length 2 (height, width)
    :param overwrite: overwrite src w/ the result
    """
    output = src if overwrite else np.empty_like(src)
    for c_i in range(src.shape[-1]):
        np.multiply(src[:, :, c_i], mask, out=output[:, :, c_i])
    return output


def blend_with_mask(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray, out: np.ndarray | None = None):
    """
    :param bg: "background" image  ( can NOT be used as out )
    :param fg: "foreground" image  ( CAN BE used as out )
    :param mask: mask with shape of length 2 (height, width)

    m * fg + (1 - m) * bg

    Blends two buffers using the 1-pass interpolation.
    Here it is assumed that the data is in float format, hence the use of apply_mask.
    This is not optimal for image processing, but will accommodate alternative data inputs such as latent images.

    Note that "background" and "foreground" are merely suggestive names,
    since zeroes are usually associated with the background.
    """
    diff = cv2.subtract(fg, bg, dst=out)
    apply_mask(diff, mask, overwrite=True)
    return cv2.add(bg, diff, dst=diff)

# endregion ----- MASK UTILITIES -----


# region --- PATCH SELECTION METHODS ---

def _get_random_valid_block(block_size: int, lookup_textures: ValidatedTexturesIterator, rng: Generator)\
        -> tuple[PatchIdx, np.ndarray]:
    # Random patch selection
    all_valid_counts = []
    total_valid = 0

    # Identify all valid patches across all textures
    for idx in range(len(lookup_textures)):
        texture = lookup_textures[idx]
        if texture.shape[0] < block_size or texture.shape[1] < block_size:
            all_valid_counts.append(0)
            continue

        if lookup_textures.has_mask(idx):
            mask = lookup_textures.get_mask(idx)
            count = np.count_nonzero(~mask)
        else:
            h, w = texture.shape[:2]
            count = (h - block_size + 1) * (w - block_size + 1)

        all_valid_counts.append(count)
        total_valid += count

    if total_valid == 0:
        raise ValueError("set_random_patch_at_location: No valid patches found in lookup textures.")

    r = int(rng.integers(total_valid))
    accumulated = 0
    for idx, count in enumerate(all_valid_counts):
        if accumulated + count > r:
            target_index = r - accumulated
            texture = lookup_textures[idx]

            if lookup_textures.has_mask(idx):
                mask = lookup_textures.get_mask(idx)
                y_t, x_t = np.nonzero(~mask)
                rand_h = int(y_t[target_index])
                rand_w = int(x_t[target_index])
            else:
                w = texture.shape[1]
                row_len = (w - block_size + 1)
                rand_h = target_index // row_len
                rand_w = target_index % row_len

            rnd_text_idx = idx
            start_block = lookup_textures[rnd_text_idx][rand_h:rand_h + block_size, rand_w:rand_w + block_size]
            return (rnd_text_idx, rand_h, rand_w), start_block
        accumulated += count
    raise RuntimeError("Reached unreachable code: target not found in items.")


def _filter_candidate_patches(err_mats: list[np.ndarray | None],
                              global_min_error: float , tolerance: float) -> list[tuple[int, int, int]]:
    """
    Filter patches with error equal or smaller than (1.0 + tolerance) * global_min_error.
    :param err_mats: error matrices resulting from template matching.
        NOTE: the error matrices indices should match the texture indices.
              if a texture is too small, or has some other problem that invalidates fetching a patch from it,
              its' corresponding err_mat can be set as None to preserve the indices correspondence.
    :param global_min_error: the min error found from all the provided matrices.
        NOTE: the iteration used to obtain the err_mats should also compute the global minimum error,
        there is no need to iterate the list an additional time to obtain this value prior to filtering the candidates.
    :param tolerance: how high, percentage wise, can the error be above the global minimum error to pass as a candidate.
    :return: the list of candidates in the following tuple format: ( index in err_mats, y-coord, x-coord )
    """
    final_candidates: list[tuple[int, int, int]] = []  # (texture_idx, y, x)
    acceptable_error = (1.0 + tolerance) * global_min_error

    for texture_idx, err_mat in enumerate(err_mats):
        if err_mat is None:  # skip small or invalid textures
            logger.warning(f"{texture_idx=} contains no errors; might be invalid or too small.")
            continue
        y_t, x_t = np.nonzero(err_mat <= acceptable_error)  # filter positions with respect to the tolerance threshold
        for y, x in zip(y_t, x_t):   # record all valid patch coordinates
            final_candidates.append((texture_idx, y, x))
    return final_candidates


def _select_a_random_patch(patches: list[PatchIdx], rng: Generator) -> PatchIdx:
    """
    Input and output patches in the format: (texture index, y-coord, x-coord).
    :raises ValueError: raised if no patches are provided.
        Potential edge case if tolerance check fails for every patch (should not be possible).
        global_min_error == np.inf checks when searching for the global minimum should be triggered
        first when it is impossible to find a patch (e.g. the lookup textures are all too small).
        Thus, this is more of a redundant check in case something unexpected occurs.
    """
    if not patches:
        raise ValueError("No patches to select from.")
    c = rng.integers(len(patches))
    return patches[c]

# endregion --- PATCH SELECTION METHODS ---


# region    ----- SEAMS AUXILIARY METHODS -----

def avg_squared_diff(block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
    err = block1 - block2
    err **= 2
    err = err.mean(2)
    return err


def adjust_errors_for_pystar2d_inplace(errors: np.ndarray, block_len: NumPixels) -> None:
    """Adjust the errors so that 1s can be used to create 'free-corridors'"""

    # scale to integer color range and offset by 1 (works for both RGB and LAB)
    #   this is done so that the distance from 0 to the smallest possible error
    #   keeps the same proportion relative to other error values
    #   otherwise there would be a relative penalty mismatch for pixels with error equal to zero
    #   when offsetting the errors, which is required for both pyastar2d (min. value accepted as weight is 1)
    errors *= 255
    errors += 1

    # make the lowest value big enough for 1 to be negligible (paths created w/ 1s become "free-corridors")
    errors *= block_len ** 2


def update_seams_map_view(seams_map_view: np.ndarray, patch_weights: np.ndarray, blends_into_patch: bool):
    seam_map_block = get_seam_mask_from_patch_weights(patch_weights, blends_into_patch)
    clear_seam_overlapped_by_patch(seams_map_view, patch_weights)
    #seams_map_view += seam_map_block
    np.maximum(seams_map_view, seam_map_block, out=seams_map_view)


def get_seam_mask_from_patch_weights(patch_weights: np.ndarray, blends_into_patch: bool) -> np.ndarray:
    """
    Seam Mask here refers to mask containing the boundary or blending area highlighted;
    it is not the same the mask used to merge the source with the patch.
    """
    if blends_into_patch:
        # Has blending, compute distance to 0.5
        seam_mask = 1 - np.abs(.5 - patch_weights) * 2
        np.clip(seam_mask, 0, 1, out=seam_mask)  # just in case
        return seam_mask
    else:
        # No blending, compute gradient to find the seam
        gx = cv2.Sobel(patch_weights, cv2.CV_32F, 1, 0, ksize=cv2.FILTER_SCHARR)
        gy = cv2.Sobel(patch_weights, cv2.CV_32F, 0, 1, ksize=cv2.FILTER_SCHARR)
        np.multiply(gx, gx, out=gx)
        np.multiply(gy, gy, out=gy)
        gx += gy
        mags = gx / 256
        np.clip(mags, 0, 1, out=mags)
        return mags


def clear_seam_overlapped_by_patch(seam_map_view: np.ndarray, patch_weights: np.ndarray):
    seam_map_view *= 1 - patch_weights

# endregion    ----- SEAMS AUXILIARY METHODS -----

