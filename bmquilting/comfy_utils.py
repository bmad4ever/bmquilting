from __future__ import annotations

_import_error_message = (
    "bmq-comfyui-utils is not installed.\n\n"
)

try:
    from bmq_comfyui_utils import *
except ImportError as e:
    if e.msg == "No module named 'bmq_comfyui_utils'":
        raise ImportError(_import_error_message) from e
    else:
        raise


def __getattr__(value):
    try:
        import bmq_comfyui_utils
    except ImportError as e:
        raise ImportError(_import_error_message) from e
    return getattr(bmq_comfyui_utils, value)