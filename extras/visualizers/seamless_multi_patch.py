# run as module: python -m extras.visualizers.seamless_multi_patch

from ..tkinter_ui_templates import SquarePatchGenApp

import bmquilting.square as square

import tkinter as tk
import numpy as np
import cv2

import logging
logging.basicConfig(level=logging.DEBUG)


class ConvertToSeamlessTextureApp(SquarePatchGenApp):
    def __init__(self, root: tk.Tk):
        self.src_img = None
        self.result_img = None
        self.seams_map = None
        self.grid_img = None
        self.current_display = "result"
        super().__init__(root, "Convert to Seamless Texture")

    def view_options(self) -> list[tuple[str, str]]:
        return [("Result", "result"), ("Seams Map", "seams"), ("2x2 Grid", "grid")]

    def setup_params(self, params_frame):
        # Direction selection
        dir_frame = tk.Frame(params_frame)
        dir_frame.pack(anchor=tk.W, pady=5)
        tk.Label(dir_frame, text="Direction:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.direction_var = tk.StringVar(value="B")
        directions = [("Both", "B"), ("Horizontal", "H"), ("Vertical", "V")]
        for text, value in directions:
            tk.Radiobutton(dir_frame, text=text, variable=self.direction_var,
                           value=value).pack(side=tk.LEFT)

        super().setup_params(params_frame)

    def on_auto_block_size(self, src_img_f32: np.ndarray):
        return round(min(src_img_f32.shape[:2]) / 2.5)

    def on_auto_overlap_size(self, block_size):
        return round(block_size / 2.8)

    def process_texture(self) -> None:
        try:
            src_img_float = self.src_img.astype(np.float32) / 255.0
            src_img_bgr = cv2.cvtColor(src_img_float, cv2.COLOR_RGB2BGR)

            # Setup random generator w/ the provided seed
            rand_gen = np.random.default_rng(seed=self.seed_var.get())

            # Fetch Gen Params
            block_size = self.get_block_size(src_img_float)
            overlap = self.get_overlap_size(block_size)
            patching_config = self.get_blend_config(block_size, overlap)

            # Select function based on direction
            seamless_multi_patch_functions = {
                "H": square.seamless_horizontal_multi,
                "V": square.seamless_vertical_multi,
                "B": square.seamless_both_multi,
            }

            direction = self.direction_var.get()
            seamless_texture, seams_map = seamless_multi_patch_functions[direction](
                image=src_img_bgr,
                patching_config=patching_config,
                rng=rand_gen,
                lookup_textures=self.get_lookup_textures(src_img_bgr)
            )

            # Convert back to RGB and to uint8
            self.result_img = (cv2.cvtColor(seamless_texture, cv2.COLOR_BGR2RGB) * 255).astype(np.uint8)
            self.seams_map = (seams_map * 255).astype(np.uint8)

            # Create 2x2 grid
            H, W = self.result_img.shape[:2]
            self.grid_img = np.empty((H * 2, W * 2, 3), dtype=np.uint8)
            self.grid_img[:H, :W] = self.result_img
            self.grid_img[:H, -W:] = self.result_img
            self.grid_img[-H:, :W] = self.result_img
            self.grid_img[-H:, -W:] = self.result_img

            # Update UI in main thread
            self.root.after(0, self.on_generation_complete)

        except Exception as e:
            self.root.after(0, lambda: self.on_generation_error(str(e)))

    def get_image(self, name):
        match name:
            case "result": return self.result_img
            case "seams": return self.seams_map
            case "grid": return self.grid_img
            case _: return None


if __name__ == "__main__":
    root = tk.Tk()
    app = ConvertToSeamlessTextureApp(root)
    root.mainloop()