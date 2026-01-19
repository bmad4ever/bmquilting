from bmquilting.generate import generate_texture_parallel, generate_texture, generate_texture_diagonal
from bmquilting.guess_block_size import guess_nice_block_size
from bmquilting.extras.tkinter_ui_templates import SquarePatchGenApp

import tkinter as tk
import numpy as np
import cv2


class GenerateTextureApp(SquarePatchGenApp):
    def __init__(self, root: tk.Tk):
        self.src_img = None
        self.result_img = None
        self.seams_map = None
        self.current_display = "result"
        super().__init__(root, "Generate Texture")

    def view_options(self) -> list[tuple[str, str]]:
        return [("Result", "result"), ("Seams", "seams")]

    def setup_params(self, params_frame):
        # Output Width
        width_frame = tk.Frame(params_frame)
        width_frame.pack(anchor=tk.W, pady=5)
        tk.Label(width_frame, text="Output Width:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.out_width_var = tk.IntVar(value=512)
        tk.Entry(width_frame, textvariable=self.out_width_var, width=10).pack(side=tk.LEFT, padx=5)

        # Output Height
        height_frame = tk.Frame(params_frame)
        height_frame.pack(anchor=tk.W, pady=5)
        tk.Label(height_frame, text="Output Height:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.out_height_var = tk.IntVar(value=512)
        tk.Entry(height_frame, textvariable=self.out_height_var, width=10).pack(side=tk.LEFT, padx=5)

        # Parallel lvl
        nps_frame = tk.Frame(params_frame)
        nps_frame.pack(anchor=tk.W, pady=5)
        tk.Label(nps_frame, text="Parallel Streams:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.nps_var = tk.IntVar(value=1)
        tk.Entry(nps_frame, textvariable=self.nps_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(nps_frame, text="(0=sequential)", fg="gray").pack(side=tk.LEFT)

        super().setup_params(params_frame)

    def on_auto_block_size(self, src_img_f32: np.ndarray):
        grayscale = cv2.cvtColor(src_img_f32, cv2.COLOR_BGR2GRAY)
        return guess_nice_block_size(grayscale, True)

    def on_auto_overlap_size(self, block_size):
        return round(block_size / 2.5)

    def process_texture(self) -> None:
        try:
            src_img_float = self.src_img.astype(np.float32) / 255.0
            src_img_bgr = cv2.cvtColor(src_img_float, cv2.COLOR_RGB2BGR)

            # Setup random generator
            seed = self.seed_var.get()
            rand_gen = np.random.default_rng(seed=seed)

            # Fetch Gen Params
            block_size = self.get_block_size(src_img_float)
            overlap = self.get_overlap_size(block_size)
            gen_params = self.get_blend_config(block_size, overlap)

            nps = min(self.nps_var.get(), 3)
            self.nps_var.set(nps)

            if nps < 0:
                seamless_texture, seams_map = generate_texture_diagonal(
                    src_textures=self.get_lookup_textures(src_img_bgr),
                    gen_params=gen_params,
                    out_h=self.out_height_var.get(),
                    out_w=self.out_width_var.get(),
                    rng=rand_gen,
                    uicd=None
                )
            elif nps == 0:
                seamless_texture, seams_map = generate_texture(
                    src_textures=self.get_lookup_textures(src_img_bgr),
                    gen_params=gen_params,
                    out_h=self.out_height_var.get(),
                    out_w=self.out_width_var.get(),
                    rng=rand_gen,
                    uicd=None
                )
            else:
                seamless_texture, seams_map = generate_texture_parallel(
                    src_textures=self.get_lookup_textures(src_img_bgr),
                    gen_params=gen_params,
                    out_h=self.out_height_var.get(),
                    out_w=self.out_width_var.get(),
                    nps=nps,
                    rng=rand_gen,
                    uicd=None
                )

            # Convert to uint8
            self.result_img = (cv2.cvtColor(seamless_texture, cv2.COLOR_BGR2RGB) * 255).astype(np.uint8)
            print(f"LE MAX = {np.min(seams_map), np.max(seams_map)}")
            self.seams_map = np.clip(seams_map * 255, 0, 255).astype(np.uint8)

            self.root.after(0, self.on_generation_complete)

        except Exception as e:
            self.root.after(0, lambda: self.on_generation_error(str(e)))

    def get_image(self, name):
        match name:
            case "result": return self.result_img
            case "seams": return self.seams_map
            case _: return None


if __name__ == "__main__":
    root = tk.Tk()
    app = GenerateTextureApp(root)
    root.mainloop()
