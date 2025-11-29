import numpy as np
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
        texture_variants[idx] = tex.copy()

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

    Args:
        img (np.ndarray):
            Input texture (2D or 3D).
        number_of_cardinals (int):
            Number of equally spaced orientations.
        interpolation:
            OpenCV interpolation flag.
            Examples: cv2.INTER_LINEAR (default), cv2.INTER_NEAREST, cv2.INTER_CUBIC, etc.

    Returns:
        List[np.ndarray]: All rotated & cropped textures.
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