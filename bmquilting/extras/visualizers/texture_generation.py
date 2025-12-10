import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from bmquilting.generate import GenParams, generate_texture_parallel, generate_texture, generate_texture_diagonal
from bmquilting.guess_block_size import guess_nice_block_size
from bmquilting.seam_smartblur import auto_blend_config_2
from bmquilting.types import BlendConfig
import numpy as np
import cv2
import threading


class TextureGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Texture Generator")
        self.root.geometry("1200x800")

        self.src_img = None
        self.result_img = None
        self.seams_map = None
        self.current_display = "result"

        # -------------------------------
        # ADVANCED SETTINGS DEFAULT VARS
        # -------------------------------
        self.seed_var = tk.IntVar(value=123)
        self.version_var = tk.IntVar(value=2)
        self.vignette_match_var = tk.BooleanVar(value=True)
        self.freq_analysis_var = tk.BooleanVar(value=False)

        self.enable_blend_var = tk.BooleanVar(value=True)
        self.blend_mode_var = tk.StringVar(value="auto")

        # Auto blend config
        self.auto_sobel_kernel_var = tk.IntVar(value=7)
        self.auto_min_blur_var = tk.IntVar(value=3)
        self.auto_use_vignette_var = tk.BooleanVar(value=False)

        # Manual blend config
        self.use_vignette_var = tk.BooleanVar(value=True)
        self.sobel_kernel_var = tk.IntVar(value=5)
        self.min_blur_var = tk.IntVar(value=1)
        self.max_blur_var = tk.IntVar(value=10)

        # --------------------------------
        # Build UI
        # --------------------------------
        self.setup_ui()

    def setup_ui(self):
        # Main container with two columns
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side - controls with scrollbar
        left_container = tk.Frame(main_container, width=400)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        left_container.pack_propagate(False)

        canvas = tk.Canvas(left_container)
        scrollbar = tk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        left_side = scrollable_frame

        # Right side - image display
        right_side = tk.Frame(main_container)
        right_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        # Control panel
        control_frame = tk.Frame(left_side, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text="Select Image", command=self.select_image,
                  bg="#4CAF50", fg="white", padx=20, pady=5).pack(side=tk.TOP, anchor=tk.W, pady=2)

        self.file_label = tk.Label(control_frame, text="No image selected", fg="gray")
        self.file_label.pack(side=tk.TOP, anchor=tk.W, pady=2)

        self.generate_btn = tk.Button(control_frame, text="Generate Texture",
                                      command=self.generate_texture, state=tk.DISABLED,
                                      bg="#2196F3", fg="white", padx=20, pady=5)
        self.generate_btn.pack(side=tk.TOP, anchor=tk.W, pady=5)

        self.progress = ttk.Progressbar(control_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.TOP, anchor=tk.W, pady=2)

        params_frame = tk.LabelFrame(left_side, text="Generation Parameters", padx=10, pady=10)
        params_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=5)

        basic_frame = tk.Frame(params_frame)
        basic_frame.pack(side=tk.TOP, fill=tk.X)

        # Output Width
        width_frame = tk.Frame(basic_frame)
        width_frame.pack(anchor=tk.W, pady=5)
        tk.Label(width_frame, text="Output Width:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.out_width_var = tk.IntVar(value=512)
        tk.Entry(width_frame, textvariable=self.out_width_var, width=10).pack(side=tk.LEFT, padx=5)

        # Output Height
        height_frame = tk.Frame(basic_frame)
        height_frame.pack(anchor=tk.W, pady=5)
        tk.Label(height_frame, text="Output Height:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.out_height_var = tk.IntVar(value=512)
        tk.Entry(height_frame, textvariable=self.out_height_var, width=10).pack(side=tk.LEFT, padx=5)

        # Block size
        block_frame = tk.Frame(basic_frame)
        block_frame.pack(anchor=tk.W, pady=5)
        tk.Label(block_frame, text="Block Size:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.block_size_var = tk.StringVar(value="auto")
        tk.Entry(block_frame, textvariable=self.block_size_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(block_frame, text="(auto/pixels)", fg="gray").pack(side=tk.LEFT)

        # Overlap
        overlap_frame = tk.Frame(basic_frame)
        overlap_frame.pack(anchor=tk.W, pady=5)
        tk.Label(overlap_frame, text="Overlap:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.overlap_var = tk.StringVar(value="auto")
        tk.Entry(overlap_frame, textvariable=self.overlap_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(overlap_frame, text="(auto/pixels)", fg="gray").pack(side=tk.LEFT)

        # Tolerance
        tol_frame = tk.Frame(basic_frame)
        tol_frame.pack(anchor=tk.W, pady=5)
        tk.Label(tol_frame, text="Tolerance:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=0.1)
        tk.Scale(tol_frame, from_=0.0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.tolerance_var, length=150).pack(side=tk.LEFT, padx=5)

        # Parallel streams
        nps_frame = tk.Frame(basic_frame)
        nps_frame.pack(anchor=tk.W, pady=5)
        tk.Label(nps_frame, text="Parallel Streams:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.nps_var = tk.IntVar(value=1)
        tk.Entry(nps_frame, textvariable=self.nps_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(nps_frame, text="(0=sequential)", fg="gray").pack(side=tk.LEFT)

        # Advanced Settings Button
        adv_btn_frame = tk.Frame(params_frame)
        adv_btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        tk.Button(adv_btn_frame, text="⚙ Advanced Settings", command=self.open_advanced_settings,
                  bg="#9C27B0", fg="white", padx=15, pady=5).pack(anchor=tk.W)

        # Texture variant selection
        variants_frame = tk.LabelFrame(left_side, text="Texture Variants to Use", padx=5, pady=5)
        variants_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.variant_vars = {}
        variants = [
            ("Original (x, y)", "original"),
            ("Flip Horizontal (-x, y)", "flip_h"),
            ("Flip Vertical (x, -y)", "flip_v"),
            ("Flip Both (-x, -y)", "flip_both"),
            ("Transpose (y, x)", "transpose"),
            ("Transpose + Flip H (-y, x)", "transpose_flip_h"),
            ("Transpose + Flip V (y, -x)", "transpose_flip_v"),
            ("Transpose + Flip Both (-y, -x)", "transpose_flip_both")
        ]

        for text, key in variants:
            var = tk.BooleanVar(value=(key == "original" or key == "flip_h"))
            self.variant_vars[key] = var
            cb = tk.Checkbutton(variants_frame, text=text, variable=var)
            cb.pack(anchor=tk.W)

        btn_frame = tk.Frame(variants_frame)
        btn_frame.pack(anchor=tk.W, pady=5)
        tk.Button(btn_frame, text="Select All", command=self.select_all_variants,
                  padx=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Only Original", command=self.select_only_original,
                  padx=10).pack(side=tk.LEFT, padx=2)

        # Right side view
        view_frame = tk.Frame(right_side, padx=10, pady=5)
        view_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(view_frame, text="View:").pack(side=tk.LEFT, padx=5)
        self.view_var = tk.StringVar(value="result")
        for text, value in [("Result", "result"), ("Seams Map", "seams")]:
            tk.Radiobutton(view_frame, text=text, variable=self.view_var,
                           value=value, command=self.update_display).pack(side=tk.LEFT)

        self.save_btn = tk.Button(view_frame, text="Save Current View",
                                  command=self.save_image, state=tk.DISABLED,
                                  bg="#FF9800", fg="white", padx=15, pady=3)
        self.save_btn.pack(side=tk.RIGHT, padx=5)

        display_frame = tk.Frame(right_side, bg="gray")
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(display_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def open_advanced_settings(self):
        """Open the Advanced Settings window (variables already exist)."""
        adv_window = tk.Toplevel(self.root)
        adv_window.title("Advanced Settings")
        adv_window.geometry("500x600")
        adv_window.transient(self.root)
        adv_window.grab_set()

        notebook = ttk.Notebook(adv_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ------------------------------
        # Tab 1 – General
        # ------------------------------
        general_tab = tk.Frame(notebook, padx=10, pady=10)
        notebook.add(general_tab, text="General")

        seed_frame = tk.Frame(general_tab)
        seed_frame.pack(anchor=tk.W, pady=5)
        tk.Label(seed_frame, text="Random Seed:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(seed_frame, textvariable=self.seed_var, width=15).pack(side=tk.LEFT, padx=5)

        version_frame = tk.Frame(general_tab)
        version_frame.pack(anchor=tk.W, pady=5)
        tk.Label(version_frame, text="Version:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(version_frame, textvariable=self.version_var, width=15).pack(side=tk.LEFT, padx=5)

        vignette_frame = tk.Frame(general_tab)
        vignette_frame.pack(anchor=tk.W, pady=10)
        tk.Checkbutton(vignette_frame, text="Use vignette on match template",
                       variable=self.vignette_match_var).pack(side=tk.LEFT)

        freq_frame = tk.Frame(general_tab)
        freq_frame.pack(anchor=tk.W, pady=5)
        tk.Checkbutton(freq_frame, text="ONLY use frequency analysis for block size",
                       variable=self.freq_analysis_var).pack(side=tk.LEFT)

        # ------------------------------
        # Tab 2 – Blend Config
        # ------------------------------
        blend_tab = tk.Frame(notebook, padx=10, pady=10)
        notebook.add(blend_tab, text="Blend Config")

        tk.Checkbutton(blend_tab, text="Enable blending (if unchecked, blend_config will be None)",
                       variable=self.enable_blend_var).pack(anchor=tk.W, pady=5)

        blend_mode_frame = tk.LabelFrame(blend_tab, text="Blend Mode", padx=10, pady=10)
        blend_mode_frame.pack(fill=tk.X, pady=10)

        tk.Radiobutton(blend_mode_frame, text="Auto Config 2", variable=self.blend_mode_var,
                       value="auto").pack(anchor=tk.W)
        tk.Radiobutton(blend_mode_frame, text="Manual", variable=self.blend_mode_var,
                       value="manual").pack(anchor=tk.W)

        # Auto Config Area
        auto_frame = tk.LabelFrame(blend_tab, text="Auto Config 2 Parameters", padx=10, pady=10)
        auto_frame.pack(fill=tk.X, pady=5)

        auto_sobel_frame = tk.Frame(auto_frame)
        auto_sobel_frame.pack(anchor=tk.W, pady=2)
        tk.Label(auto_sobel_frame, text="Sobel Kernel Size:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(auto_sobel_frame, textvariable=self.auto_sobel_kernel_var, width=10).pack(side=tk.LEFT, padx=5)

        auto_minblur_frame = tk.Frame(auto_frame)
        auto_minblur_frame.pack(anchor=tk.W, pady=2)
        tk.Label(auto_minblur_frame, text="Min Blur Diameter:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(auto_minblur_frame, textvariable=self.auto_min_blur_var, width=10).pack(side=tk.LEFT, padx=5)

        auto_vig_frame = tk.Frame(auto_frame)
        auto_vig_frame.pack(anchor=tk.W, pady=2)
        tk.Checkbutton(auto_vig_frame, text="Use vignette", variable=self.auto_use_vignette_var).pack(side=tk.LEFT)

        # Manual Config
        manual_frame = tk.LabelFrame(blend_tab, text="Manual Blend Parameters", padx=10, pady=10)
        manual_frame.pack(fill=tk.X, pady=5)

        vig_frame = tk.Frame(manual_frame)
        vig_frame.pack(anchor=tk.W, pady=2)
        tk.Checkbutton(vig_frame, text="Use vignette", variable=self.use_vignette_var).pack(side=tk.LEFT)

        sobel_frame = tk.Frame(manual_frame)
        sobel_frame.pack(anchor=tk.W, pady=2)
        tk.Label(sobel_frame, text="Sobel Kernel Size:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(sobel_frame, textvariable=self.sobel_kernel_var, width=10).pack(side=tk.LEFT, padx=5)

        minblur_frame = tk.Frame(manual_frame)
        minblur_frame.pack(anchor=tk.W, pady=2)
        tk.Label(minblur_frame, text="Min Blur Diameter:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(minblur_frame, textvariable=self.min_blur_var, width=10).pack(side=tk.LEFT, padx=5)

        maxblur_frame = tk.Frame(manual_frame)
        maxblur_frame.pack(anchor=tk.W, pady=2)
        tk.Label(maxblur_frame, text="Max Blur Diameter:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(maxblur_frame, textvariable=self.max_blur_var, width=10).pack(side=tk.LEFT, padx=5)

        tk.Button(adv_window, text="Close", command=adv_window.destroy,
                  bg="#4CAF50", fg="white", padx=20, pady=5).pack(pady=10)

    # -------------------------------------------------------------------
    # The rest of the file remains unchanged
    # -------------------------------------------------------------------

    def select_all_variants(self):
        for var in self.variant_vars.values():
            var.set(True)

    def select_only_original(self):
        for key, var in self.variant_vars.items():
            var.set(key == "original")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Texture Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.src_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if self.src_img is None:
                    raise ValueError("Failed to load image")

                self.src_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB)
                self.file_label.config(text=file_path.split("/")[-1], fg="black")
                self.generate_btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"Image loaded: {self.src_img.shape[1]}x{self.src_img.shape[0]}")
                self.display_image(self.src_img)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.status_label.config(text="Error loading image")

    def generate_texture(self):
        if self.src_img is None:
            return

        selected_variants = [k for k, v in self.variant_vars.items() if v.get()]
        if not selected_variants:
            messagebox.showwarning("No Variants Selected",
                                   "Please select at least one texture variant.")
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_label.config(text="Generating texture...")

        thread = threading.Thread(target=self.process_texture)
        thread.start()

    def process_texture(self):
        try:
            src_img_float = self.src_img.astype(np.float32) / 255.0
            src_img_bgr = cv2.cvtColor(src_img_float, cv2.COLOR_RGB2BGR)

            # Direct – vars are always initialized
            seed = self.seed_var.get()
            version = self.version_var.get()
            vignette_on_match = self.vignette_match_var.get()

            rand_gen = np.random.default_rng(seed=seed)

            # Block size
            block_size_str = self.block_size_var.get().strip().lower()
            if block_size_str == "auto":
                grayscale = cv2.cvtColor(src_img_bgr, cv2.COLOR_BGR2GRAY)
                block_size = guess_nice_block_size(grayscale, self.freq_analysis_var.get())
            else:
                block_size = int(block_size_str)
                if block_size <= 0:
                    raise ValueError("Block size must be positive")

            # Overlap
            overlap_str = self.overlap_var.get().strip().lower()
            if overlap_str == "auto":
                overlap = round(block_size / 2.5)
            else:
                overlap = int(overlap_str)
                if overlap < 0 or overlap >= block_size:
                    raise ValueError("Overlap must be between 0 and block_size")

            tolerance = self.tolerance_var.get()

            # Blend config
            blend_config = None
            if self.enable_blend_var.get():
                if self.blend_mode_var.get() == "auto":
                    blend_config = auto_blend_config_2(
                        self.auto_sobel_kernel_var.get(),
                        overlap,
                        self.auto_min_blur_var.get(),
                        self.auto_use_vignette_var.get()
                    )
                else:
                    blend_config = BlendConfig(
                        use_vignette=self.use_vignette_var.get(),
                        sobel_kernel_size=self.sobel_kernel_var.get(),
                        min_blur_diameter=self.min_blur_var.get(),
                        max_blur_diameter=self.max_blur_var.get()
                    )

            lookup_textures = []
            for variant, enabled in self.variant_vars.items():
                if enabled.get():
                    tex = src_img_bgr
                    if variant == "flip_h": tex = np.fliplr(tex)
                    elif variant == "flip_v": tex = np.flipud(tex)
                    elif variant == "flip_both": tex = np.flipud(np.fliplr(tex))
                    elif variant == "transpose": tex = np.transpose(tex, (1, 0, 2))
                    elif variant == "transpose_flip_h":
                        tex = np.fliplr(np.transpose(tex, (1, 0, 2)))
                    elif variant == "transpose_flip_v":
                        tex = np.flipud(np.transpose(tex, (1, 0, 2)))
                    elif variant == "transpose_flip_both":
                        tex = np.flipud(np.fliplr(np.transpose(tex, (1, 0, 2))))
                    lookup_textures.append(tex)

            gen_args = GenParams(
                block_size=block_size,
                overlap=overlap,
                tolerance=tolerance,
                vignette_on_match_template=vignette_on_match,
                blend_config=blend_config,
                version=version
            )

            nps = min(self.nps_var.get(), 3)
            self.nps_var.set(nps)

            if nps < 0:
                seamless_texture, seams_map = generate_texture_diagonal(
                    src_textures=lookup_textures,
                    gen_args=gen_args,
                    out_h=self.out_height_var.get(),
                    out_w=self.out_width_var.get(),
                    rng=rand_gen,
                    uicd=None
                )
            elif nps == 0:
                seamless_texture, seams_map = generate_texture(
                    src_textures=lookup_textures,
                    gen_args=gen_args,
                    out_h=self.out_height_var.get(),
                    out_w=self.out_width_var.get(),
                    rng=rand_gen,
                    uicd=None
                )
            else:
                seamless_texture, seams_map = generate_texture_parallel(
                    src_textures=lookup_textures,
                    gen_args=gen_args,
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

    def on_generation_complete(self):
        self.progress.stop()
        self.generate_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Generation complete!")
        self.update_display()
        messagebox.showinfo("Success", "Seamless texture generated successfully!")

    def on_generation_error(self, msg):
        self.progress.stop()
        self.generate_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Generation failed")
        messagebox.showerror("Error", f"Failed to generate texture:\n{msg}")

    def update_display(self):
        if self.view_var.get() == "result" and self.result_img is not None:
            self.display_image(self.result_img)
        elif self.view_var.get() == "seams" and self.seams_map is not None:
            self.display_image(self.seams_map)

    def display_image(self, img_array):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 800, 500

        if len(img_array.shape) == 2:
            pil_img = Image.fromarray(img_array, mode='L')
        else:
            pil_img = Image.fromarray(img_array, mode='RGB')

        img_w, img_h = pil_img.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.photo)

    def save_image(self):
        view = self.view_var.get()
        img = None
        default_name = ""

        if view == "result" and self.result_img is not None:
            img = self.result_img
            default_name = "seamless_result.png"
        elif view == "seams" and self.seams_map is not None:
            img = self.seams_map
            default_name = "seams_map.png"

        if img is None:
            messagebox.showwarning("Warning", "No image to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )

        if file_path:
            try:
                Image.fromarray(img).save(file_path)
                self.status_label.config(text=f"Saved to {file_path}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TextureGeneratorApp(root)
    root.mainloop()
