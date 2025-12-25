from .synthesis_subroutines import (
    get_min_cut_patch_horizontal_method, get_min_cut_patch_vertical_method, get_min_cut_patch_both_method,
    update_seams_map_view, find_patch_vx_idx, apply_mask)
from .types import UiCoordData, GenParams, num_pixels, patch_idxs
from .misc.custom_decorators import clear_cache_post_exec
from math import ceil
import numpy as np


@clear_cache_post_exec()
def _compute_synthesis_map(
        proxy_textures: list[np.ndarray],
        gen_params: GenParams,
        out_h: num_pixels, out_w: num_pixels,
        rng: np.random.Generator,
        uicd: UiCoordData | None) -> tuple[list[patch_idxs], np.ndarray, np.ndarray, np.ndarray] | None:
    """
    @return: The data required to obtain the patches from the source textures and the individual patches' masks plus
    the generated proxy textured and its seams map.

    This is returned in the following format:
        1st item: list of (text_idx, y, x) pairs.
        2nd item: ndarray of shape (number of patches, block size, block_size) containing individual patches' masks.
        3rd item: generated proxy texture
        4th item: proxy texture's seams map
    """
    b, o = gen_params.block_size, gen_params.overlap
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

    patch_indices: list[patch_idxs] = []  # text index, row, column
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

    def process_block(blk_idx, get_min_cut_patch, ref_block, seams_map_view, patch_idx: patch_idxs):
        patch_indices.append(patch_idx)
        patch_block = get_block(*patch_idx)
        min_cut_patch, patch_weights = get_min_cut_patch(ref_block, patch_block, gen_params)
        store_patch_mask(patch_weights)
        ref_block[:] = min_cut_patch
        seams_map_sub_view = seams_map_view[blk_idx]
        update_seams_map_view(seams_map_sub_view, gen_params, patch_weights)

    def fill_row_inplace_proxy():
        get_min_cut_patch = get_min_cut_patch_horizontal_method(gen_params.version)
        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            blk_idx = np.s_[:b, blk_x:(blk_x + b)]
            ref_block = texture_map[blk_idx]
            patch_idx = find_patch_vx_idx(
                True, False, False, False,
                ref_block, proxy_textures, gen_params, rng)
            process_block(blk_idx, get_min_cut_patch, ref_block, seams_map, patch_idx)

    def fill_quad_proxy():
        """
            Only requires the first Row already filled.
            A "purist" solution, where there is no apriori column generation.
        """
        get_min_cut_v = get_min_cut_patch_vertical_method(gen_params.version)
        get_min_cut_b = get_min_cut_patch_both_method(gen_params.version)

        for blk_y in range(bmo, texture_map.shape[0] - b + 1 , bmo):
            blk_idx = np.s_[blk_y:blk_y + b, :b]
            ref_block = texture_map[blk_idx]
            patch_idx = find_patch_vx_idx(
                False, False, True, False,
                ref_block, proxy_textures, gen_params, rng)
            process_block(blk_idx, get_min_cut_v, ref_block, seams_map, patch_idx)

            for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
                blk_idx = np.s_[blk_y:blk_y + b, blk_x:blk_x + b]
                ref_block = texture_map[blk_idx]
                patch_idx = find_patch_vx_idx(
                    True, False, True, False,
                    ref_block, proxy_textures, gen_params, rng)
                process_block(blk_idx, get_min_cut_b, ref_block, seams_map, patch_idx)

            if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
                break
        return texture_map, seams_map

    # --- Generate ---
    fill_row_inplace_proxy()
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
        return None

    fill_quad_proxy()

    np.clip(seams_map, 0, 1, out=seams_map)
    return patch_indices, patch_masks, texture_map[:out_h, :out_w], seams_map[:out_h, :out_w]


def _reconstruct_texture(textures: list[np.ndarray],
                         patches_indices: list[patch_idxs],
                         patches_masks: np.ndarray,
                         out_h: num_pixels, out_w: num_pixels,
                         gen_params: GenParams,
                         uicd: UiCoordData | None
                         ) -> np.ndarray:
    b, o = gen_params.block_size, gen_params.overlap
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

    def fill_row_inplace():
        for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
            apply_patch(np.s_[:b, blk_x:blk_x+b])
        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
            return None

    def fill_quad_inplace():
        for blk_y in range(bmo, texture_map.shape[0] - b + 1, bmo):
            apply_patch(np.s_[blk_y:blk_y + b, :b])
            for blk_x in range(bmo, texture_map.shape[1] - b + 1, bmo):
                apply_patch(np.s_[blk_y:blk_y + b, blk_x:blk_x + b])
            if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
                return None

    # --- Generate ---
    texture_map[:b, :b] = get_block_at_idx(0)  # set 1st patch
    fill_row_inplace()
    fill_quad_inplace()
    return texture_map[:out_h, :out_w]


def generate_guided(
        proxy_textures: list[np.ndarray],
        source_textures: list[np.ndarray],
        gen_params: GenParams,
        out_h: num_pixels, out_w: num_pixels,
        rng: np.random.Generator,
        uicd: UiCoordData | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses a variant of the source_textures to guide the texture synthesis algorithms and maps the result to the
    source textures.
    Can be used, for example, in noisy images, where a filter can be applied so that the
    generation is not influenced by noise or some other visual artifacts or features.

    @param proxy_textures: textures used to guide the generation;
        the textures order SHOULD MATCH those in the source textures list.
    @param source_textures: textures used to build the final result post guided synthesis.
    @return:
        item 1: reconstructed synthesis result using the source textures
        item 2: seams map
        item 3: synthesis result using the proxy textures
    """
    patches_idxs, masks, proxy_out_tex, out_cut = _compute_synthesis_map(
        proxy_textures, gen_params, out_h, out_w, rng, uicd
    )
    out_tex = _reconstruct_texture(
        source_textures, patches_idxs, masks, out_h, out_w, gen_params, uicd
    )
    return out_tex, out_cut, proxy_out_tex
