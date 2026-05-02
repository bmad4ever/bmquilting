from functools import wraps
import numpy as np
import weakref
import inspect

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def clear_cache_post_exec(*functions_to_clear):
    """
    Decorator factory that clears the cache of the specified functions
    after the decorated function executes.

    If none is provided, DEFAULT_CACHED_FUNCTIONS list is used instead.

    :param functions_to_clear: A list of functions (with @lru_cache) whose caches should be cleared.
    """
    if not functions_to_clear:
        raise ValueError("No functions to clear were provided.")

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Execute the original function
                return func(*args, **kwargs)
            finally:
                logger.info("--- Starting Cache Cleanup ---")
                for cached_func in functions_to_clear:
                    # Check if the function has the cache_clear method (i.e., is cached)
                    if hasattr(cached_func, 'cache_clear'):
                        cached_func.cache_clear()
                        logger.info(f"  - Cleared cache for: {cached_func.__name__}")
                    else:
                        # Should not happen if used correctly, but good for robust code
                        logger.warning(f"  - WARNING: {cached_func.__name__} does not have cache_clear.")
                logger.info("--- Cache Cleanup Complete ---")

        return wrapper

    return decorator


def step_predictor(calc_func):
    """
    This decorator is meant to be used in functions that use uicd and have a deterministic behavior,
    to enhance them with a function that computes the number of steps they will take.

    Use decorated_function_name.predict_steps(args),
    where in args the relevant arguments to obtain the number of steps are passed.
    """
    def decorator(main_func):
        main_sig = inspect.signature(main_func)
        calc_sig = inspect.signature(calc_func)

        @wraps(main_func)
        def wrapper(*args, **kwargs):
            return main_func(*args, **kwargs)

        def predict(*args, **kwargs):
            # Use bind_partial so we don't need to provide every arg
            bound = main_sig.bind_partial(*args, **kwargs)

            # Filter only what the predictor wants
            filtered_args = {
                k: v for k, v in bound.arguments.items()
                if k in calc_sig.parameters
            }

            return calc_func(**filtered_args)

        wrapper.predict_steps = predict
        return wrapper

    return decorator


def ndarray_identity_cache(array_arg_index: int = 0):
    """
    Cache results based on identity of a numpy ndarray argument.
    """

    def decorator(func):
        cache = {}
        finalizers = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            arr = args[array_arg_index]
            key_id = id(arr)

            per_array = cache.get(key_id)
            if per_array is None:
                per_array = {}
                cache[key_id] = per_array

                # remove cache entry when array dies
                # likely not really needed, but better have some redundant cleanup procedure just in case
                finalizers[key_id] = weakref.ref(
                    arr,
                    lambda _,
                    k=key_id: (
                        cache.pop(k, None),
                        finalizers.pop(k, None)
                    )
                )

            key = (
                args[:array_arg_index] +
                args[array_arg_index + 1:],
                tuple(sorted(kwargs.items()))
            )

            if key in per_array:
                return per_array[key]

            result = func(*args, **kwargs)
            per_array[key] = result
            return result

        # === exposed functions/data ====
        # naming must be the same as the one used in clear_cache_post_exec

        def cache_clear():
            cache.clear()
            finalizers.clear()

        wrapper.cache_clear = cache_clear
        wrapper._cache = cache

        return wrapper

    return decorator


def auto_uint8_to_float32(func):
    """
    Decorator that detects uint8 numpy inputs and automatically converts them to float32 [0, 1].
    The output(s) of the function are converted back to uint8 [0, 255] if any input was uint8.

    Supports:
    - np.ndarray inputs
    - list[np.ndarray] inputs (like source_textures)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        was_uint8 = False

        def convert_in(obj):
            nonlocal was_uint8
            if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
                if not was_uint8:
                    logger.info(f"auto_uint8_to_float32: Detected uint8 input in '{func.__name__}'. "
                                f"Converting to float32 [0, 1] for processing.")
                was_uint8 = True
                return obj.astype(np.float32) / 255.0
            if isinstance(obj, list):
                return [convert_in(item) for item in obj]
            return obj

        def convert_out(obj):
            if isinstance(obj, np.ndarray) and obj.dtype == np.float32:
                # Use np.round and clip to handle potential float precision issues
                return np.clip(np.round(obj * 255.0), 0, 255).astype(np.uint8)
            if isinstance(obj, tuple):
                return tuple(convert_out(item) for item in obj)
            if isinstance(obj, list):
                return [convert_out(item) for item in obj]
            return obj

        # Convert positional arguments
        new_args = tuple(convert_in(arg) for arg in args)

        # Convert keyword arguments
        new_kwargs = {k: convert_in(v) for k, v in kwargs.items()}

        result = func(*new_args, **new_kwargs)

        if was_uint8:
            logger.info(f"auto_uint8_to_float32: Converting output of '{func.__name__}' back to uint8.")
            return convert_out(result)

        return result

    return wrapper
