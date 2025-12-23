from ..seam_smartblur import get_max_possible_gradient_diff, circular_kernel, get_radii_limiter
from ..synthesis_subroutines import patch_blending_vignette, get_slice_metadata_for_find_patch
from functools import wraps


DEFAULT_CACHED_FUNCTIONS = [
    circular_kernel,
    get_max_possible_gradient_diff,
    get_radii_limiter,
    get_slice_metadata_for_find_patch,
    patch_blending_vignette,
]


def clear_cache_post_exec(*functions_to_clear):
    """
    Decorator factory that clears the cache of the specified functions
    after the decorated function executes.

    If none is provided, DEFAULT_CACHED_FUNCTIONS list is used instead.

    Args:
        *functions_to_clear: A list of functions (with @lru_cache)
                             whose caches should be cleared.
    """
    if not functions_to_clear:
        # If the tuple is empty, use the standard default list
        functions_to_clear = DEFAULT_CACHED_FUNCTIONS

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Execute the original function
                return func(*args, **kwargs)
            finally:
                print("--- Starting Cache Cleanup ---")
                for cached_func in functions_to_clear:
                    # Check if the function has the cache_clear method (i.e., is cached)
                    if hasattr(cached_func, 'cache_clear'):
                        cached_func.cache_clear()
                        print(f"  - Cleared cache for: {cached_func.__name__}")
                    else:
                        # Should not happen if used correctly, but good for robust code
                        print(f"  - WARNING: {cached_func.__name__} does not have cache_clear.")
                print("--- Cache Cleanup Complete ---")

        return wrapper

    return decorator
