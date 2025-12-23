from .synthesis_subroutines import (
    get_min_cut_patch_horizontal_method, get_min_cut_patch_vertical_method, get_min_cut_patch_both_method,
    update_seams_map_view, find_patch_vx_idx, apply_mask)
from .types import UiCoordData, GenParams, num_pixels, patch_idxs
from .misc.custom_decorators import clear_cache_post_exec
from math import ceil
import numpy as np


@clear_cache_post_exec()
def generate_texture_by_proxy(proxy_textures: list[np.ndarray],
                              gen_args: GenParams,
                              out_h: num_pixels, out_w: num_pixels,
                              rng: np.random.Generator,
                              uicd: UiCoordData | None) -> tuple[list[patch_idxs], np.ndarray, np.ndarray, np.ndarray]:
    """
    @return: The data required to obtain the patches from the source textures and the individual patches' masks plus
    the generated proxy textured and its seams map.

    This is returned in the following format:
        1st item: list of (text_idx, y, x) pairs.
        2nd item: ndarray of shape (number of patches, block size, block_size) containing individual patches' masks.
        3rd item: generated proxy texture
        4th item: proxy texture's seams map
    """
    b, o = gen_args.block_size, gen_args.overlap
    bmo = b - o

    n_h = int(ceil((out_h - b) / (b - o)))
    n_w = int(ceil((out_w - b) / (b - o)))

    # select random image for the starting block
    rand_tex_idx = rng.integers(len(proxy_textures))
    image = proxy_textures[rand_tex_idx]

    texture_map = np.zeros(
        ((b + n_h * (b - o)),
         (b + n_w * (b - o)),
         image.shape[2]), dtype=image.dtype)

    patch_indices: list[tuple[int, int, int]] = []  # text index, row, column
    total_blocks = (n_h + 1) * (n_w + 1)
    patch_masks = np.empty((total_blocks, b, b))
    num_masks_added: int = 0

    def store_patch_mask(mask):
        nonlocal num_masks_added
        patch_masks[num_masks_added] = mask
        num_masks_added += 1

    seams_map = np.zeros(texture_map.shape[:2], dtype=np.float32)

    # Starting index and block
    h, w = image.shape[:2]
    rand_h = rng.integers(h - b)
    rand_w = rng.integers(w - b)

    start_block = image[rand_h:rand_h + b, rand_w:rand_w + b]
    texture_map[:b, :b, :] = start_block

    patch_indices.append((int(rand_tex_idx), int(rand_h), int(rand_w)))
    store_patch_mask(np.zeros((b, b), dtype=np.float32))

    del image, rand_tex_idx, rand_h, rand_w  # access texture list instead after this point

    def get_block(text_idx, y, x):
        winning_texture = proxy_textures[text_idx]
        return winning_texture[y:y + b, x:x + b]

    def fill_row_inplace_proxy(texture_map_view: np.ndarray, seams_map_view: np.ndarray):
        get_min_cut_patch = get_min_cut_patch_horizontal_method(gen_args.version)
        for blk_idx in range(bmo, texture_map_view.shape[1] - b + 1, bmo):
            #ref_block = texture_map_view[:b, (blk_idx - bmo):(blk_idx + o)]
            ref_block = texture_map_view[:b, blk_idx:(blk_idx + b)]
            tex_idx, y, x = find_patch_vx_idx(
                True, False, False, False,
                ref_block, proxy_textures, gen_args, rng)
            patch_indices.append((tex_idx, y, x))
            patch_block = get_block(tex_idx, y, x)
            min_cut_patch, patch_weights = get_min_cut_patch(ref_block, patch_block, gen_args)
            store_patch_mask(patch_weights)
            ref_block[:] = min_cut_patch
            seams_map_sub_view = seams_map_view[:b, blk_idx:(blk_idx + b)]
            update_seams_map_view(seams_map_sub_view, gen_args, patch_weights)

    def fill_quad_proxy(rows: int, columns: int):
        """
            Only requires the first Row already filled.
            A "purist" solution, where there is no apriori column generation.
        """
        get_min_cut_v = get_min_cut_patch_vertical_method(gen_args.version)
        get_min_cut_b = get_min_cut_patch_both_method(gen_args.version)

        for i in range(1, rows + 1):
            blk_index_i = i * (b - o)

            ref_block = texture_map[blk_index_i:(blk_index_i + b), :b]

            tex_idx, y, x = find_patch_vx_idx(
                False, False, True, False,
                ref_block, proxy_textures, gen_args, rng)
            patch_indices.append((tex_idx, y, x))
            patch_block = get_block(tex_idx, y, x)
            min_cut_patch, patch_weights = get_min_cut_v(ref_block, patch_block, gen_args)
            store_patch_mask(patch_weights)

            ref_block[:] = min_cut_patch

            seams_map_view = seams_map[blk_index_i:(blk_index_i + b), :b]
            update_seams_map_view(seams_map_view, gen_args, patch_weights)

            for j in range(1, columns + 1):
                blk_index_j = j * (b - o)
                ref_block = texture_map[blk_index_i:(blk_index_i + b), blk_index_j:(blk_index_j + b)]

                tex_idx, y, x = find_patch_vx_idx(
                    True, False, True, False,
                    ref_block, proxy_textures, gen_args, rng)
                patch_indices.append((tex_idx, y, x))
                patch_block = get_block(tex_idx, y, x)
                min_cut_patch, patch_weights = get_min_cut_b(ref_block, patch_block, gen_args)
                store_patch_mask(patch_weights)

                ref_block[:] = min_cut_patch

                seams_map_view = seams_map[blk_index_i:(blk_index_i + b),
                                 blk_index_j:(blk_index_j + b)]
                update_seams_map_view(seams_map_view, gen_args, patch_weights)

            if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(columns + 1):
                break
        return texture_map, seams_map

    # --- Generate ---
    fill_row_inplace_proxy(texture_map[:b, :], seams_map[:b, :])
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
        return None

    fill_quad_proxy(n_h, n_w)

    np.clip(seams_map, 0, 1, out=seams_map)
    return patch_indices, patch_masks, texture_map[:out_h, :out_w], seams_map[:out_h, :out_w]


def build_texture(textures: list[np.ndarray], patches_indices: list[patch_idxs], patches_masks: np.ndarray,
                  out_h: num_pixels, out_w: num_pixels, gen_args: GenParams):
    b, o = gen_args.block_size, gen_args.overlap
    bmo = b - o

    n_h = int(ceil((out_h - b) / (b - o)))
    n_w = int(ceil((out_w - b) / (b - o)))

    texture_map = np.empty(
        ((b + n_h * (b - o)),
         (b + n_w * (b - o)),
         textures[0].shape[2]), dtype=textures[0].dtype)

    block_idx = 0

    def fetch_masks_at_idx(idx):
        return patches_masks[idx]

    def get_block_at_idx(idx):
        tex_idx, y, x = patches_indices[idx]
        return textures[tex_idx][y:y+b, x:x+b]

    # set 1st patch
    texture_map[:b, :b] = get_block_at_idx(0)

    def fill_row_inplace(texture_map_view: np.ndarray):
        nonlocal block_idx
        for blk_idx in range(bmo, texture_map_view.shape[1] - b + 1, bmo):
            block_idx += 1  # do not mistake for blk_idx, should rename IT!
            patch_block = get_block_at_idx(block_idx)
            patch_weights = fetch_masks_at_idx(block_idx)

            np.subtract(1, patch_weights, out=patch_weights)
            text_view = texture_map_view[:b, blk_idx:(blk_idx + b)]
            apply_mask(text_view, patch_weights, True)
            np.subtract(1, patch_weights, out=patch_weights)
            text_view += apply_mask(patch_block, patch_weights, False)

    # --- Generate ---
    fill_row_inplace(texture_map[:b, :])

    # Seems to be working so far
    # TODO finish implementation

    return texture_map[:out_h, :out_w]