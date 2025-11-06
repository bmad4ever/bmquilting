import numpy as np


def apply_mask(src: np.ndarray, mask: np.ndarray, overwrite: bool = False):
    """
    @param src:  image or latent with shape of length 3 (height, width, channels)
    @param mask: mask with shape of length 2 (height, width)
    @param overwrite: overwrite src w/ the result
    """
    output = src if overwrite else np.empty_like(src)
    for c_i in range(src.shape[-1]):
        np.multiply(src[:, :, c_i], mask, out=output[:, :, c_i])
    return output