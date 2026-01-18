from ..seam_smartblur import get_max_possible_gradient_diff, circular_kernel, get_radii_limiter
from ..synthesis_subroutines import patch_blending_vignette, _get_overlap_mask, _get_vignetted_overlap_mask
from functools import wraps
import logging
import inspect

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


DEFAULT_CACHED_FUNCTIONS = [
    circular_kernel,
    get_max_possible_gradient_diff,
    get_radii_limiter,
    _get_overlap_mask,
    _get_vignetted_overlap_mask,
    patch_blending_vignette,
]


def clear_cache_post_exec(*functions_to_clear):
    """
    Decorator factory that clears the cache of the specified functions
    after the decorated function executes.

    If none is provided, DEFAULT_CACHED_FUNCTIONS list is used instead.

    :param functions_to_clear: A list of functions (with @lru_cache) whose caches should be cleared.
    """
    if not functions_to_clear:
        # If the tuple is empty, use the default list
        functions_to_clear = DEFAULT_CACHED_FUNCTIONS

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
