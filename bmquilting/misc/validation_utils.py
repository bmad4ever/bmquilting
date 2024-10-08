import numpy as np
import inspect
from ..types import GenParams, num_pixels, Orientation


def validate_array_shape(array: np.ndarray, min_height: int = None, min_width: int = None, help_msg: str = ""):
    """
    Validates that the given array has sufficient dimensions for the specified minimum height and width.

    Parameters:
        array (np.ndarray): The input array to validate.
        min_height (int, optional): The minimum required size for the array's first dimension (height). Defaults to None.
        min_width (int, optional): The minimum required size for the array's second dimension (width). Defaults to None.
        help_msg (str, optional): Message appended to the raised exception. Defaults to empty string.

    Raises:
        ValueError: If the array's dimensions do not meet the required conditions.
    """
    valid_height = min_height is None or array.shape[0] >= min_height
    valid_width = min_width is None or array.shape[1] >= min_width

    if not (valid_height and valid_width):
        # Only fetch the variable name if an exception is going to be raised
        caller_frame = inspect.currentframe().f_back
        array_name = None
        for var_name, var_val in caller_frame.f_locals.items():
            if var_val is array:
                array_name = var_name
                break

        array_name = array_name if array_name else "Array"

        if not valid_height:
            raise ValueError(f"The operation requires an height of {min_height},"
                             f" but {array_name} only has {array.shape[0]}.\n{help_msg}")

        if not valid_width:
            raise ValueError(f"The operation requires a width of {min_width},"
                             f" but {array_name} only has {array.shape[1]}.\n{help_msg}")


def validate_gen_args(source, gen_args: GenParams):
    """
    Validates arguments for quilting generation.

    Checks if source has width and height of at least block_size.
    If not, raises ValueError with a helping message
    """
    validate_array_shape(source, min_height=gen_args.block_size, min_width=gen_args.block_size,
                         help_msg="Change the block size.")


def validate_seamless_args(orientation: Orientation, source: np.ndarray, lookup: np.ndarray,
                           block_size: num_pixels, overlap: num_pixels):
    if overlap * 2 > block_size:
        raise ValueError(f"Overlap ({overlap}) needs to be 50% or less of the block size ({block_size}).")

    if orientation == Orientation.H_AND_V:
        validate_array_shape(source, min_height=block_size, min_width=block_size + overlap * 2,
                             help_msg="Change the block size or the overlap.")
    else:
        validate_array_shape(source, min_height=block_size, min_width=block_size,
                             help_msg="Change the block size.")

    if lookup is not None:
        validate_array_shape(lookup, min_height=block_size, min_width=block_size,
                             help_msg="Use a bigger lookup or change the block size.")
        if lookup.dtype != source.dtype:
            raise TypeError("lookup_texture dtype does not match image dtype")