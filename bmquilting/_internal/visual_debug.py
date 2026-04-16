"""
Visual debugging module for subroutines.

Uses sys.settrace to intercept function calls and specific line numbers in
seams_blur.py, circular_subroutines.py, and square_subroutines.py, displaying
intermediate numpy array data in a single Tkinter window with dropdown selection.

Arrays are categorized by size:
  - **Block-sized** (min dimension <= 128): patch-scale intermediates
  - **Full-sized**  (min dimension  > 128): texture-scale arrays

Optionally saves debug frames to a "debug_viz" folder.

Usage:
    from bmquilting._internal.visual_debug import enable_visual_debug, disable_visual_debug

    enable_visual_debug(save_frames=True)
    # ... run your quilting code ...
    disable_visual_debug()
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import sys
import cv2



# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_MAX_BLOCK_DIM = 128       # max dimension to classify as "block-sized"
_DISPLAY_MAX_BLOCK = 400   # max display size for block arrays (pixels)
_DISPLAY_MAX_FULL = 512    # max display size for full arrays (pixels)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_tracing_enabled = False
_save_frames = False
_frame_counter = 0
_debug_viz_dir: Path | None = None
_target_modules: set[str] = set()
_call_depth = 0
_in_patch = False
_patch_count = 0

# Per-patch array buffer: list of (display_name, numpy_array)
_patch_arrays: list[tuple[str, np.ndarray]] = []

# ---------------------------------------------------------------------------
# Manual line-level breakpoints for create_adaptive_blend_mask intermediates
# ---------------------------------------------------------------------------
_BLEND_MASK_BREAKPOINTS: dict[int, tuple[list[str], Callable]|list[str]] = {
    # TODO consider these INVALID until version complete
    319: ["tdiff_norm"],
    323: ["tdiff_norm"],
    329: ["blend_radii"],
    332: ["blend_radii", "radii_limiter"],
    342: ["blend_radii"],
    346: ["blend_radii"],
    348: ["dist_to_source"],
    350: ["dist_to_patch"],
    351: ["signed_distance"],
    352: ["signed_distance"],
    355: ["t", "signed_distance", "blend_radii"],
    356: ["t"]
}

_ARRAY_LIKE_KEYWORDS = {"mask", "patch", "source", "block", "img", "image",
                        "texture", "result", "output", "roi", "vignette",
                        "blend", "seam", "tdiff", "radius", "distance",
                        "gradient", "err", "error", "patched"}

_FNS_TO_EXCLUDE = {
    "create_adaptive_blend_mask",   # manually check via _BLEND_MASK_BREAKPOINTS
}

_PATCH_BOUNDARY_FNS = {
    "process_patch_at_location",
    "set_random_patch_at_location",
    "find_patch_vx",
    "find_patch_vx_idx",
    "get_4way_seam_patched",
    "_make_seamless_vertical_circular",
    "_make_seamless_horizontal_circular",
    "_make_seamless_both_circular",
}

# ---------------------------------------------------------------------------
# Tkinter debug viewer (single window, dropdown-driven)
# ---------------------------------------------------------------------------

class _DebugViewer:
    """Singleton Tkinter viewer for patch-level debug arrays."""

    def __init__(self):
        self._root = None
        self._patch_num: int = 0
        self._block_arrays: list[tuple[str, np.ndarray]] = []
        self._full_arrays: list[tuple[str, np.ndarray]] = []
        self._block_combo = None
        self._full_combo = None
        self._block_label = None
        self._full_label = None
        self._saved_geometry: tuple[int, int] | None = None  # (x, y) screen position
        self._saved_combos_idx: tuple[int, int] = (0, 0)
        self._action: str = "next"  # "next" | "stop"

    def show_patch(self, patch_num: int, arrays: list[tuple[str, np.ndarray]]) -> str:
        """Blocking call: displays all arrays in a single window with dropdowns.

        Returns the user action:
          - "next" : advance to the next patch (default keypress)
          - "stop" : stop debugging, disable trace
        """
        self._action = "next"

        # Categorize by size
        self._block_arrays = [(n, a) for n, a in arrays if min(a.shape[:2]) <= _MAX_BLOCK_DIM]
        self._full_arrays  = [(n, a) for n, a in arrays if min(a.shape[:2]) > _MAX_BLOCK_DIM]

        if not self._block_arrays and not self._full_arrays:
            raise RuntimeError("Unexpected error; nothing within _block_arrays or _full_arrays.")

        # De-duplicate names (append index when duplicates exist)
        self._block_arrays = _dedup_names(self._block_arrays)
        self._full_arrays = _dedup_names(self._full_arrays)

        self._patch_num = patch_num

        import tkinter as tk
        from tkinter import ttk

        self._root = tk.Tk()
        self._root.title(f"Patch #{patch_num}  –  bmquilting debug viewer")
        self._root.protocol("WM_DELETE_WINDOW", self._on_stop)
        self._root.resizable(True, True)

        self._build_ui(ttk)

        # Restore previous window position
        self._restore_geometry()

        # Auto-select first items
        if self._block_arrays:
            combo_idx = self._saved_combos_idx[0] if self._saved_combos_idx[0] < len(self._block_combo["values"]) else 0
            self._block_combo.current(combo_idx)
            self._display_for(self._block_label, "block")
        if self._full_arrays:
            combo_idx = self._saved_combos_idx[1] if self._saved_combos_idx[1] < len(self._full_combo["values"]) else 0
            self._full_combo.current(combo_idx)
            self._display_for(self._full_label, "full")

        # Key bindings
        self._root.bind("<Return>", self._on_advance)
        self._root.bind("<space>", self._on_advance)
        self._root.bind("<Escape>", self._on_stop)

        self._root.mainloop()
        return self._action

    # ---- UI construction ----

    def _build_ui(self, ttk):
        main = ttk.Frame(self._root, padding=10)
        main.pack(fill="both", expand=True)

        # Title bar
        ttk.Label(main, text="Use the dropdowns to inspect each intermediate array.").pack(anchor="w")

        # Two-column layout
        content = ttk.Frame(main)
        content.pack(fill="both", expand=True, pady=8)

        # --- Block-sized column ---
        if self._block_arrays:
            left = ttk.LabelFrame(content, text=f"Block-sized  ({len(self._block_arrays)})", padding=6)
            left.pack(side="left", fill="both", expand=True, padx=4)

            # Dropdown row
            row_top = ttk.Frame(left)
            row_top.pack(fill="x")

            names = [n for n, _ in self._block_arrays]
            self._block_combo = ttk.Combobox(row_top, values=names, state="readonly", width=50)
            self._block_combo.pack(side="left", fill="x", expand=True)
            self._block_combo.bind("<<ComboboxSelected>>",
                                   lambda e: self._display_for(self._block_label, "block"))

            ttk.Button(row_top, text="Save \u2192", width=8,
                       command=lambda: self._save_selected("block")).pack(side="left", padx=(4, 0))

            self._block_label = ttk.Label(left, relief="sunken", anchor="center")
            self._block_label.pack(fill="both", expand=True, pady=5)

        # --- Full-sized column ---
        if self._full_arrays:
            right = ttk.LabelFrame(content, text=f"Full-sized  ({len(self._full_arrays)})", padding=6)
            right.pack(side="left", fill="both", expand=True, padx=4)

            # Dropdown row
            row_top = ttk.Frame(right)
            row_top.pack(fill="x")

            names = [n for n, _ in self._full_arrays]
            self._full_combo = ttk.Combobox(row_top, values=names, state="readonly", width=50)
            self._full_combo.pack(side="left", fill="x", expand=True)
            self._full_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self._display_for(self._full_label, "full"))

            ttk.Button(row_top, text="Save \u2192", width=8,
                       command=lambda: self._save_selected("full")).pack(side="left", padx=(4, 0))

            self._full_label = ttk.Label(right, relief="sunken", anchor="center")
            self._full_label.pack(fill="both", expand=True, pady=5)

        # Bottom bar
        bottom = ttk.Frame(main)
        bottom.pack(fill="x")

        # Left side: label + Stop button
        left_bar = ttk.Frame(bottom)
        left_bar.pack(side="left")
        ttk.Label(left_bar, text="Space / Enter \u2192 next patch").pack(side="left")
        ttk.Button(left_bar, text="Stop Debugging", command=self._on_stop,
                   style="TButton").pack(side="left", padx=8)

        # Right side: action buttons
        btn_row = ttk.Frame(bottom)
        btn_row.pack(side="right")
        ttk.Button(btn_row, text="Save All (Patch)",
                   command=self._save_all_patch).pack(side="right", padx=(0, 6))
        ttk.Button(btn_row, text="Next Patch \u2192", command=self._on_advance).pack(side="right")

    # ---- Display helpers ----

    def _display_for(self, label_widget, panel):
        idx = (self._block_combo if panel == "block" else self._full_combo).current()
        if idx < 0:
            return
        arrays = self._block_arrays if panel == "block" else self._full_arrays
        max_px = _DISPLAY_MAX_BLOCK if panel == "block" else _DISPLAY_MAX_FULL
        _, arr = arrays[idx]
        # Single-channel block arrays get a heatmap + colorbar for easier reading
        if panel == "block" and _is_single_channel(arr):
            _set_photo_heatmap(label_widget, arr, max_px)
        else:
            _set_photo(label_widget, arr, max_px)

    def _on_advance(self, event=None):
        self._action = "next"
        if self._root and self._root.winfo_exists():
            self._saved_geometry = (self._root.winfo_x(), self._root.winfo_y())
            self._saved_combos_idx = (self._block_combo.current(), self._full_combo.current())
            self._root.quit()
            self._root.destroy()
        self._root = None

    def _on_stop(self, event=None):
        self._action = "stop"
        if self._root and self._root.winfo_exists():
            self._saved_geometry = (self._root.winfo_x(), self._root.winfo_y())
            self._root.quit()
            self._root.destroy()
        self._root = None

    def _restore_geometry(self):
        if self._saved_geometry is not None and self._root:
            x, y = self._saved_geometry
            self._root.geometry(f"+{x}+{y}")

    # ---- Save helpers ----

    def _get_panel_arrays(self, panel: str) -> list[tuple[str, np.ndarray]]:
        return self._block_arrays if panel == "block" else self._full_arrays

    def _save_selected(self, panel: str) -> None:
        """Save currently selected array from the dropdown as .npy and .png."""
        from tkinter import filedialog

        idx = (self._block_combo if panel == "block" else self._full_combo).current()
        if idx < 0:
            return
        arrays = self._get_panel_arrays(panel)
        name, arr = arrays[idx]

        # Sanitize filename
        safe_name = name.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")[:120]

        # File dialog for .npy
        save_path = filedialog.asksaveasfilename(
            title="Save array",
            initialfile=f"patch{self._patch_num}_{safe_name}",
            defaultextension=".npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")]
        )
        if not save_path:
            return
        np.save(save_path, arr)
        print(f"[visual_debug] Saved .npy → {save_path}")

        # Also save a .png for visual reference
        png_path = save_path + ".png"
        viz = _normalise_for_display(arr)
        if viz is not None:
            cv2.imwrite(png_path, viz)
            print(f"[visual_debug] Saved .png → {png_path}")

    def _save_all_patch(self) -> None:
        """Save all arrays for the current patch into debug_viz/saved_arrays/patch_N/."""
        all_arrays = self._block_arrays + self._full_arrays
        if not all_arrays:
            return

        save_dir = Path.cwd() / "debug_viz" / "saved_arrays" / f"patch_{self._patch_num}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, arr in all_arrays:
            safe_name = name.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")[:120]
            np.save(save_dir / f"{safe_name}.npy", arr)
            viz = _normalise_for_display(arr)
            if viz is not None:
                cv2.imwrite(str(save_dir / f"{safe_name}.png"), viz)

        print(f"[visual_debug] Saved {len(all_arrays)} arrays (.npy + .png) → {save_dir}")


# Module-level viewer singleton
_viewer = _DebugViewer()


def _dedup_names(pairs: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Append numeric suffix to duplicate names so dropdown entries are unique."""
    seen: dict[str, int] = {}
    result = []
    for name, arr in pairs:
        if name in seen:
            seen[name] += 1
            result.append((f"{name}  (#{seen[name]})", arr))
        else:
            seen[name] = 0
            result.append((name, arr))
    return result


def _is_single_channel(arr: np.ndarray) -> bool:
    """Return True when *arr* is effectively a scalar/grayscale image."""
    return arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)


def _make_heatmap_with_scale(arr: np.ndarray, max_px: int) -> np.ndarray | None:
    """Render a single-channel array as a false-colour heatmap with a colorbar.

    The colorbar strip sits to the right of the image and carries three tick
    labels: the true data maximum (top), midpoint (middle), and minimum
    (bottom), expressed in the *original* array units (not 0-255).

    Returns a uint8 BGR image that can be encoded directly with cv2.imencode.
    """
    if arr.ndim < 2:
        return None

    _COLORMAP = cv2.COLORMAP_TURBO
    _HEATMAP_GAMMA = 1.5  # gamma > 1 compresses low values (similar dark colours) and stretches

    data = arr[:, :, 0] if arr.ndim == 3 else arr   # always 2-D from here on
    data = data.copy()

    inf_idxs = np.isinf(data)
    data[inf_idxs] = np.nan

    if np.any(inf_idxs):
        _COLORMAP = cv2.COLORMAP_INFERNO
        _HEATMAP_GAMMA = .65

    data[np.isnan(data)] = 0  #hi*1.25
    lo, hi = float(data.min()), float(data.max())
    rng = hi - lo

    if np.abs(1-hi) <= .02 and lo == 0:
        _COLORMAP = cv2.COLORMAP_BONE

    if rng > 0:
        norm = ((data - lo) / rng * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(data, dtype=np.uint8)

    norm = (norm.astype(np.float32) / 255.0) ** _HEATMAP_GAMMA
    norm = (norm * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, _COLORMAP)  # H × W × 3 BGR

    # ---- Scale heatmap to fit display box ----
    h, w = heatmap.shape[:2]
    scale = min(max_px / w, max_px / h, 1.0)
    if scale < 0.99:
        heatmap = cv2.resize(heatmap, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
    h, w = heatmap.shape[:2]

    # ---- Build colorbar ----
    BAR_W   = 24          # width of the gradient strip (px)
    PAD     = 6           # gap between image and gradient strip (px)
    LABEL_W = 64          # reserved width for tick-mark text (px)

    # Colormap Vertical Bar
    gradient_1d = (np.linspace(1.0, 0.0, h, dtype=np.float32) ** _HEATMAP_GAMMA * 255).astype(np.uint8).reshape(h, 1)
    bar_img = cv2.applyColorMap(gradient_1d, _COLORMAP)  # H × 1 × 3
    bar_img = cv2.resize(bar_img, (BAR_W, h), interpolation=cv2.INTER_LINEAR)

    # Canvas: dark background to the right of the heatmap
    canvas_w = PAD + BAR_W + LABEL_W
    canvas = np.full((h, canvas_w, 3), 30, dtype=np.uint8)
    canvas[:, PAD: PAD + BAR_W] = bar_img

    # Tick-mark labels: hi at top, mid in centre, lo at bottom
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1
    txt_color  = (210, 210, 210)
    tx         = PAD + BAR_W + 4          # x start for all labels

    def _fmt(v: float) -> str:
        # Use fixed-point when values are in a "normal" range, else sci notation
        return f"{v:.3g}"

    mid = (lo + hi) / 2.0
    tick_specs = [
        (0,      _fmt(hi)),
        (h // 2, _fmt(mid)),
        (h - 1,  _fmt(lo)),
    ]
    for y_anchor, text in tick_specs:
        (_, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        y_put = max(th + 1, min(h - 2, y_anchor + th // 2))
        cv2.putText(canvas, text, (tx, y_put),
                    font, font_scale, txt_color, thickness, cv2.LINE_AA)

    # Thin border on the left edge of the gradient strip for clarity
    cv2.line(canvas, (PAD, 0), (PAD, h - 1), (80, 80, 80), 1)
    cv2.line(canvas, (PAD + BAR_W, 0), (PAD + BAR_W, h - 1), (80, 80, 80), 1)

    return np.hstack([heatmap, canvas])


def _set_photo_heatmap(label_widget, arr: np.ndarray, max_px: int) -> None:
    """Encode a heatmap+colorbar composite and attach it to *label_widget*."""
    viz = _make_heatmap_with_scale(arr, max_px)
    if viz is None:
        # Graceful fallback: use the standard renderer
        _set_photo(label_widget, arr, max_px)
        return

    _, png_bytes = cv2.imencode(".png", viz)
    import tkinter as tk
    photo = tk.PhotoImage(data=png_bytes.tobytes())
    label_widget.configure(image=photo)
    label_widget.image = photo  # prevent GC


def _set_photo(label_widget, arr: np.ndarray, max_px: int) -> None:
    """Render a numpy array as a Tkinter PhotoImage and attach to a label."""
    viz = _normalise_for_display(arr)
    if viz is None:
        return

    # Scale to fit display box
    h, w = viz.shape[:2]
    scale = min(max_px / w, max_px / h, 1.0)
    if scale < 0.99:
        viz = cv2.resize(viz, (round(w * scale), round(h * scale)))

    # Encode as PNG → PhotoImage (no PIL dependency)
    _, png_bytes = cv2.imencode(".png", viz)
    import tkinter as tk
    photo = tk.PhotoImage(data=png_bytes.tobytes())
    label_widget.configure(image=photo)
    label_widget.image = photo  # prevent GC


# ---------------------------------------------------------------------------
# Array normalisation (shared)
# ---------------------------------------------------------------------------

def _normalise_for_display(arr: np.ndarray) -> np.ndarray | None:
    """Convert any numeric array to uint8 BGR suitable for display / encoding."""
    if arr.ndim < 2:
        return None

    arr = arr.copy()

    if arr.ndim > 2:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[-1] == 2:
            arr = np.dstack([arr, np.zeros_like(arr[:, :, 0])])
        elif arr.shape[-1] > 3:
            arr = arr[:, :, :3]

    if arr.ndim == 2:
        cv2.normalize(arr, arr, 0, 255, cv2.NORM_MINMAX)
    elif arr.ndim == 3:
        arr *= 255.0

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    return arr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_save_frame(arr: np.ndarray, label: str) -> None:
    global _frame_counter
    if not _save_frames or _debug_viz_dir is None:
        return
    viz = _normalise_for_display(arr)
    if viz is None:
        return
    out_path = _debug_viz_dir / f"{_frame_counter:05d}_{label}.png"
    cv2.imwrite(str(out_path), viz)
    _frame_counter += 1


def _collect_array(name: str, arr: np.ndarray, filename: str, func_name: str, adjust_data_for_viz_func: Callable|None=None) -> None:
    """Buffer a named array for the current patch (copied to freeze state)."""
    arr = arr.copy()
    _maybe_save_frame(arr, f"{func_name}_{name}")

    custom_viz_note = ""
    if adjust_data_for_viz_func:
        arr = adjust_data_for_viz_func(arr)
        custom_viz_note = "(vizmod)"

    bare = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name
    display_name = f"[{bare}]  {name}  {arr.shape} {custom_viz_note}"
    _patch_arrays.append((display_name, arr))


def _flush_arrays() -> str:
    """At patch boundary: show all buffered arrays in the viewer, then clear.

    Returns the user action: "next", "stop", or "skip_end".
    """
    global _patch_count
    if not _patch_arrays:
        return "next"

    _patch_count += 1
    print(f"\n[visual_debug] === Patch #{_patch_count}  ({len(_patch_arrays)} arrays) ===")
    action = _viewer.show_patch(_patch_count, _patch_arrays)
    _patch_arrays.clear()
    return action


# ---------------------------------------------------------------------------
# Trace function
# ---------------------------------------------------------------------------

def _trace_calls(frame, event, arg):
    global _call_depth, _in_patch

    filename = frame.f_code.co_filename
    is_target = any(mod in filename for mod in _target_modules)
    func_name = frame.f_code.co_qualname or frame.f_code.co_name
    bare_name = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name

    if event == "call":
        if not is_target:
            return None
        _call_depth += 1
        if bare_name in _PATCH_BOUNDARY_FNS:
            _in_patch = True
        return _trace_calls

    elif event == "line":
        if not is_target:
            return _trace_calls
        lineno = frame.f_lineno
        if lineno in _BLEND_MASK_BREAKPOINTS:
            item = _BLEND_MASK_BREAKPOINTS[lineno]

            vars_list, adjust_data_for_viz_func = item if isinstance(item, tuple) else (item, None)

            for vname in vars_list:
                val = frame.f_locals.get(vname)
                if isinstance(val, np.ndarray) and val.size >= 16:
                    _collect_array(vname, val, filename, func_name, adjust_data_for_viz_func)
        return _trace_calls

    elif event == "return":
        if is_target and _call_depth <= 6:
            _inspect_locals(frame.f_locals, func_name, filename)
        _call_depth -= 1
        if _call_depth < 0:
            _call_depth = 0

        # Exit from patch-boundary function → flush buffered arrays
        if bare_name in _PATCH_BOUNDARY_FNS and _in_patch:
            _in_patch = False
            action = _flush_arrays()
            if action == "stop":
                disable_visual_debug()
            elif action == "skip_end":
                # Drop all intermediate arrays from here onward until generation ends
                _patch_arrays.clear()
                # Still collect arrays from the very last patch by tracking a flag
                _skip_mode[0] = True

        return _trace_calls

    return None


def _inspect_locals(locals_dict: dict[str, Any], func_name: str, filename: str) -> None:
    if _call_depth > 6:
        return
    for name, val in locals_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        name_lower = name.lower()
        if not any(kw in name_lower for kw in _ARRAY_LIKE_KEYWORDS):
            continue
        if val.size < 16:
            continue
        if func_name in _FNS_TO_EXCLUDE:
            continue
        _collect_array(name, val, filename, func_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enable_visual_debug(save_frames: bool = False) -> None:
    """
    Activate visual debugging via sys.settrace.

    :param save_frames: If True, every displayed frame is also saved as a PNG
        into a folder called "debug_viz" in the current working directory.
    """
    global _tracing_enabled, _save_frames, _frame_counter, _debug_viz_dir, \
           _target_modules, _patch_arrays, _patch_count, _in_patch, _call_depth, _skip_mode

    if _tracing_enabled:
        print("[visual_debug] Already enabled. Call disable_visual_debug() first.")
        return

    _save_frames = save_frames
    _frame_counter = 0
    _patch_arrays = []
    _patch_count = 0
    _in_patch = False
    _call_depth = 0

    if save_frames:
        _debug_viz_dir = Path.cwd() / "debug_viz"
        _debug_viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"[visual_debug] Frames will be saved to: {_debug_viz_dir}")

    import bmquilting._internal.seams_blur as sb
    import bmquilting._internal.circular_subroutines as cs
    import bmquilting._internal.square_subroutines as sq

    _target_modules = {sb.__file__, cs.__file__, sq.__file__}

    sys.settrace(_trace_calls)
    _tracing_enabled = True
    print("[visual_debug] Enabled.  A single Tkinter window with dropdowns will appear"
          " at each patch boundary.  Press Space/Enter/Esc to advance.")


def disable_visual_debug() -> None:
    """Deactivate visual debugging and restore normal execution."""
    global _tracing_enabled
    if not _tracing_enabled:
        return
    sys.settrace(None)
    _tracing_enabled = False
    # Flush anything remaining
    if _patch_arrays:
        _flush_arrays()
    _patch_arrays.clear()
    # Destroy viewer if still alive
    if _viewer._root and _viewer._root.winfo_exists():
        _viewer._root.quit()
        _viewer._root.destroy()
        _viewer._root = None
    print("[visual_debug] Disabled.")