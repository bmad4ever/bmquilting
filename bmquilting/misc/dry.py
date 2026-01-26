import numpy as np
import cv2


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
    :param bg: background image
    :param fg: foreground image
    :param mask: mask with shape of length 2 (height, width)
    
    m * fg + (1 - m) * bg

    Blends two buffers using the 1-pass interpolation.
    Here it is assumed that the data is in float format, hence the use apply mask.
    This is not optimal for image processing, but will accommodate alternative data inputs such as latent images.
    """
    diff = cv2.subtract(fg, bg, dst=out)
    apply_mask(diff, mask, overwrite=True)
    return cv2.add(bg, diff, dst=diff)
