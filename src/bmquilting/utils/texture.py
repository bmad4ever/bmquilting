import numpy as np
import hashlib
import cv2

# region OPTIONAL NUMBA NJIT, ONLY USED IF NUMBA IS INSTALLED
try:
    from numba import njit

    def optional_njit(*args, **kwargs):
        return njit(*args, **kwargs)

except ImportError:
    # fallback: dummy decorator
    def optional_njit(*args, **kwargs):
        def decorator(func):
            return func
        # if used as @optional_njit without parentheses
        if args and callable(args[0]):
            return args[0]
        return decorator
# endregion


def get_texture_variants(
        img: np.ndarray,
        original: bool = True,
        flip_h: bool = True,
        flip_v: bool = True,
        flip_both: bool = True,
        transposed: bool = True,
        transpose_flip_h: bool = True,
        transpose_flip_v: bool = True,
        transpose_flip_both: bool = True
):
    """ Compute mirrored and transposed variants of the provided image."""
    texture_variants = []

    if original:
        texture_variants.append(img)
    if flip_h:
        texture_variants.append(np.fliplr(img))
    if flip_v:
        texture_variants.append(np.flipud(img))
    if flip_both:
        texture_variants.append(np.flipud(np.fliplr(img)))
    if transposed:
        texture_variants.append(np.transpose(img, (1, 0, 2)))  # swap x and y
    if transpose_flip_h:
        transposed = np.transpose(img, (1, 0, 2))
        texture_variants.append(np.fliplr(transposed))
    if transpose_flip_v:
        transposed = np.transpose(img, (1, 0, 2))
        texture_variants.append(np.flipud(transposed))
    if transpose_flip_both:
        transposed = np.transpose(img, (1, 0, 2))
        texture_variants.append(np.flipud(np.fliplr(transposed)))

    for idx, tex in enumerate(texture_variants):
        texture_variants[idx] = np.ascontiguousarray(tex)

    return texture_variants


@optional_njit(cache=True)
def _largest_inner_rectangle(mask: np.ndarray):
    H, W = mask.shape
    if H == 0 or W == 0: return (0, 0, 0, 0)

    height = np.zeros(W, dtype=np.int32)
    stack_pos = np.zeros(W + 1, dtype=np.int32)
    stack_val = np.zeros(W + 1, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)

    for row in range(H):
        height += 1
        height[~mask[row]] = 0

        stack_size = 0
        for col in range(W + 1):
            h = height[col] if col < W else 0
            start = col
            while stack_size > 0 and stack_val[stack_size - 1] > h:
                prev_pos = stack_pos[stack_size - 1]
                prev_h = stack_val[stack_size - 1]
                stack_size -= 1
                area = prev_h * (col - prev_pos)
                if area > best_area:
                    best_area = area
                    best = (row - prev_h + 1, prev_pos, prev_h, col - prev_pos)
                start = prev_pos
            stack_pos[stack_size] = start
            stack_val[stack_size] = h
            stack_size += 1

    return best


def _crop_to_largest_inner_area(img: np.ndarray):
    if img.ndim == 3:
        mask = np.any(img != 0, axis=2)
    else:
        mask = img != 0

    if not mask.any():
        return img.copy()

    y, x, h, w = _largest_inner_rectangle(mask)
    return img[y:y+h, x:x+w].copy()


def _rotate_image_cv(img: np.ndarray, angle: float, interpolation) -> np.ndarray:
    """
    Rotate an image by an arbitrary angle using OpenCV and expand canvas,
    then return the rotated image without cropping (cropping happens later).
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute bounding box for rotated image
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust rotation center for the expanded canvas
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated


def get_texture_rotated_variants(
        img: np.ndarray,
        number_of_cardinals: int,
        interpolation=cv2.INTER_LINEAR
) -> list[np.ndarray]:
    """
    Generate N equally-spaced rotations (360° / N).
    Each rotated texture is tightly cropped to remove empty pixels.

    :param img: Input texture (2D or 3D).
    :param number_of_cardinals: Number of equally spaced orientations.
    :param interpolation: OpenCV interpolation flag.
        Examples: cv2.INTER_LINEAR (default), cv2.INTER_NEAREST, cv2.INTER_CUBIC, etc.

    :return: List[np.ndarray]: All rotated & cropped textures.
    """
    if number_of_cardinals <= 0:
        raise ValueError("number_of_cardinals must be >= 1")

    variants = []
    step_angle = 360.0 / number_of_cardinals

    for i in range(number_of_cardinals):
        angle = step_angle * i

        rotated = _rotate_image_cv(img, angle, interpolation)
        cropped = _crop_to_largest_inner_area(rotated)

        variants.append(cropped)

    return variants


def quick_checksum(tensor: np.ndarray, check_ratio: float = 0.05) -> int:
    """
    Computes a quick, position-sensitive checksum by hashing a strategic
    slice (the first N% of the array) of the tensor's raw bytes.
    """
    slice_size_elements = int(tensor.size * check_ratio)
    sample_slice = tensor.flat[:slice_size_elements]
    data_bytes = sample_slice.tobytes()
    hex_digest = hashlib.sha1(data_bytes).hexdigest()
    return int(hex_digest, 16)


def add_salt_and_pepper(image: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5, seed: int = 1):
    """
    Adds salt and pepper noise to an image.
    :param image: Input image
    :param amount: Total proportion of image pixels to replace with noise
    :param salt_vs_pepper: Proportion of salt (white) vs pepper (black)
    :param seed: Seed for the random generated noise
    :return: Noisy image
    """
    out = np.copy(image)

    rng = np.random.default_rng(seed=seed)

    # Add Salt noise (white pixels)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    coords = [rng.integers(0, i - 1, int(num_salt))
              for i in image.shape[:2]]
    out[tuple(coords)] = 1.0

    # Add Pepper noise (black pixels)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    coords = [rng.integers(0, i - 1, int(num_pepper))
              for i in image.shape[:2]]
    out[tuple(coords)] = 0

    return out


def set_invalid_texture_area(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Invalidate an area in the src texture so that no patch can be fetched from it.
    If the source is of integer type (except for signed int8), the values in the returned array will be divided by 255.
    :param src: will not be edited, the output is a new numpy array.
    :param mask: zeroes will be set as invalid
    :return: a copy of src with the invalid texture area set to np.inf
    """
    res = np.float32(src)
    if src.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64]:
        res /= 255.0
    res[mask == 0] = np.inf
    return res


def curate_for_tex_transfer(tex: np.ndarray, value_range=255.0,
                            cvt_code: int|None = cv2.COLOR_BGR2GRAY,
                            mask: np.ndarray|None = None) -> np.ndarray:
    """
    Converts tex to a single channels 2D array, if not already converted.
    Applies the following operations in order: Gaussian Blur; CLAHE; Equalize Histogram.
    :param tex: image
    :param value_range: the  maximum possible value in the tex's format (supposes all channels have the same range)
    :param cvt_code: if set to None, the channels' mean is used; otherwise, cv2.cvtColor(tex, cvt_code) is used.
    :param mask: binary mask (1s & 0s; not 255s in case of uint8 masks).
    :return: the "curated" texture with shape HxWx1 (ndim=3)
    """
    new_tex = None

    if tex.ndim > 3: raise ValueError("Texture dimension is too large")
    if tex.ndim == 3 and tex.shape[-1] > 1:
        new_tex = cv2.cvtColor(tex, cvt_code) if cvt_code is not None else tex.mean(axis=-1)
    if new_tex is None: new_tex = tex.copy()

    cv2.GaussianBlur(new_tex, (5, 5), 1, dst=new_tex)
    tile_grid_size = (tex.shape[0]//10, tex.shape[1]//10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_grid_size)
    if mask is not None and mask.dtype != np.uint8: mask = np.uint8(mask)
    if mask is not None: new_tex *= mask
    new_tex = clahe.apply(np.uint8(new_tex*value_range))
    if mask is not None: new_tex *= mask
    new_tex = cv2.equalizeHist(new_tex).astype(np.float32)
    new_tex /= value_range
    if mask is not None: new_tex *= mask
    new_tex = new_tex[:, :, np.newaxis]
    return new_tex