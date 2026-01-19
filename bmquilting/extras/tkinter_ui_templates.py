from bmquilting.seam_smartblur import auto_blend_config_2
from bmquilting.types import BlendConfig, SquarePatchingBlendConfig, NumPixels
from bmquilting.synthesis_subroutines import (
    ignore_min_cut_patch, get_min_cut_patch_mask_horizontal_jena2020, get_min_cut_patch_mask_horizontal_astar)
from bmquilting.generate import GenParams

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from abc import ABC, abstractmethod
from dataclasses import asdict
from numpy import  ndarray
import numpy as np
import threading
import cv2


class TextureAppTemplate(ABC):
    @abstractmethod
    def __init__(self):
        self.setup_ui()

    @abstractmethod
    def setup_params(self, params_frame: tk.Frame) -> None:
        """
        :param params_frame: scrollable frame to the left.
        """
        pass

    @abstractmethod
    def view_options(self) -> list[tuple[str, str]]:
        """ list of: (display name, view identifier string) pairs """
        pass

    @abstractmethod
    def exec_function(self) -> None:
        pass

    def setup_ui(self):
        # Main container with two columns
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side - controls with scrollbar
        left_container = tk.Frame(main_container, width=400)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        left_container.pack_propagate(False)

        # Create canvas and scrollbar for left side
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

        # Enable mouse wheel scrolling
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

        # File selection
        tk.Button(control_frame, text="Select Image", command=self.select_image,
                  bg="#4CAF50", fg="white", padx=20, pady=5).pack(side=tk.TOP, anchor=tk.W, pady=2)

        self.file_label = tk.Label(control_frame, text="No image selected", fg="gray")
        self.file_label.pack(side=tk.TOP, anchor=tk.W, pady=2)

        # Generate button
        self.exec_btn = tk.Button(control_frame, text="RUN",
                                  command=self.exec_function, state=tk.DISABLED,
                                  bg="#2196F3", fg="white", padx=20, pady=5)
        self.exec_btn.pack(side=tk.TOP, anchor=tk.W, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.TOP, anchor=tk.W, pady=2)

        # Parameters frame
        params_frame = tk.LabelFrame(left_side, text="Generation Parameters", padx=10, pady=10)
        params_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=5)

        # Left column
        left_params = tk.Frame(params_frame)
        left_params.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.setup_params(params_frame)

        # View options and image display
        view_frame = tk.Frame(right_side, padx=10, pady=5)
        view_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(view_frame, text="View:").pack(side=tk.LEFT, padx=5)
        self.view_var = tk.StringVar(value="result")
        view_options = self.view_options()
        for text, value in view_options:
            tk.Radiobutton(view_frame, text=text, variable=self.view_var,
                           value=value, command=self.update_display).pack(side=tk.LEFT)

        # Save button
        self.save_btn = tk.Button(view_frame, text="Save Current View",
                                  command=self.save_image, state=tk.DISABLED,
                                  bg="#FF9800", fg="white", padx=15, pady=3)
        self.save_btn.pack(side=tk.RIGHT, padx=5)

        # Image display area
        display_frame = tk.Frame(right_side, bg="gray")
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(display_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_label = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

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
                self.exec_btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"Image loaded: {self.src_img.shape[1]}x{self.src_img.shape[0]}")

                # Display source image
                self.display_image(self.src_img)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.status_label.config(text="Error loading image")


    def enable_widget(self, widget):
        """Recursively enable a widget and its children"""
        try:
            widget.configure(state=tk.NORMAL)
        except:
            pass
        for child in widget.winfo_children():
            self.enable_widget(child)

    def disable_widget(self, widget):
        """Recursively disable a widget and its children"""
        try:
            widget.configure(state=tk.DISABLED)
        except:
            pass
        for child in widget.winfo_children():
            self.disable_widget(child)

    def select_all_variants(self):
        for var in self.variant_vars.values():
            var.set(True)

    def select_only_original(self):
        for key, var in self.variant_vars.items():
            var.set(key == "original")

    def on_generation_complete(self):
        self.progress.stop()
        self.exec_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Generation complete!")
        self.update_display()
        messagebox.showinfo("Success", "Texture generated successfully!")

    def on_generation_error(self, error_msg):
        self.progress.stop()
        self.exec_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Generation failed")
        messagebox.showerror("Error", f"Failed to generate texture:\n{error_msg}")

    @abstractmethod
    def get_image(self, name):
        pass

    def update_display(self):
        view = self.view_var.get()
        for _, name in self.view_options():
            if view == name:
                img_to_display = self.get_image(name)
                if img_to_display is not None:
                    self.display_image(img_to_display)
                break

    def display_image(self, img_array):
        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 800, 500

        # Convert to PIL Image
        if len(img_array.shape) == 2:
            pil_img = Image.fromarray(img_array, mode='L')
        else:
            pil_img = Image.fromarray(img_array, mode='RGB')

        # Resize to fit canvas while maintaining aspect ratio
        img_w, img_h = pil_img.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Display on canvas
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.photo)

    def save_image(self):
        view = self.view_var.get()
        default_name, img_to_save = None, None

        for _, name in self.view_options():
            if view == name:
                default_name = f"{name}.png"
                img_to_save = self.get_image(name)
                break

        if img_to_save is None:
            messagebox.showwarning("Warning", "No image to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )

        if file_path:
            try:
                pil_img = Image.fromarray(img_to_save)
                pil_img.save(file_path)
                self.status_label.config(text=f"Saved to {file_path}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")


class SquarePatchGenApp(TextureAppTemplate):
    @abstractmethod
    def __init__(self, root: tk.Tk, title:str, dims:str="1200x800"):
        self.root = root
        self.root.title(title)
        self.root.geometry(dims)
        super().__init__()


    def setup_params(self, params_frame):
        # Block size
        block_frame = tk.Frame(params_frame)
        block_frame.pack(anchor=tk.W, pady=5)
        tk.Label(block_frame, text="Block Size:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.block_size_var = tk.StringVar(value="auto")
        tk.Entry(block_frame, textvariable=self.block_size_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(block_frame, text="(auto or pixel value)", fg="gray").pack(side=tk.LEFT)

        # Overlap
        overlap_frame = tk.Frame(params_frame)
        overlap_frame.pack(anchor=tk.W, pady=5)
        tk.Label(overlap_frame, text="Overlap:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.overlap_var = tk.StringVar(value="auto")
        tk.Entry(overlap_frame, textvariable=self.overlap_var, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(overlap_frame, text="(auto or pixel value)", fg="gray").pack(side=tk.LEFT)

        # Tolerance
        tol_frame = tk.Frame(params_frame)
        tol_frame.pack(anchor=tk.W, pady=5)
        tk.Label(tol_frame, text="Tolerance:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=0.05)
        tk.Scale(tol_frame, from_=0.0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.tolerance_var, length=200).pack(side=tk.LEFT, padx=5)

        # Random seed
        seed_frame = tk.Frame(params_frame)
        seed_frame.pack(anchor=tk.W, pady=5)
        tk.Label(seed_frame, text="Random Seed:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.seed_var = tk.IntVar(value=123)
        tk.Entry(seed_frame, textvariable=self.seed_var, width=10).pack(side=tk.LEFT, padx=5)

        # Seam Method
        seam_method_frame = tk.Frame(params_frame)
        seam_method_frame.pack(anchor=tk.W, pady=5)
        tk.Label(
            seam_method_frame,
            text="Method:",
            width=15,
            anchor="w"
        ).pack(side="left")
        self.seam_method_var = tk.StringVar(value="astar")
        seam_method_combo = ttk.Combobox(
            seam_method_frame,
            width=27,
            textvariable=self.seam_method_var
        )
        seam_method_combo['values'] = (
            "astar", "purist", "none"
        )
        seam_method_combo.pack(side=tk.LEFT)
        seam_method_combo.current(0)

        # Vignette on match template
        vignette_frame = tk.Frame(params_frame)
        vignette_frame.pack(anchor=tk.W, pady=5)
        self.vignette_match_var = tk.BooleanVar(value=False)
        tk.Checkbutton(vignette_frame, text="Use vignette on match template",
                       variable=self.vignette_match_var).pack(side=tk.LEFT)

        # Blend configuration section
        blend_frame = tk.LabelFrame(params_frame, text="Blend Configuration", padx=5, pady=5)
        blend_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        # Enable blending
        self.enable_blend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(blend_frame, text="Enable blending (if unchecked, blend_config will be None)",
                       variable=self.enable_blend_var, command=self.toggle_blend_options).pack(anchor=tk.W)

        self.blend_options_frame = tk.Frame(blend_frame)
        self.blend_options_frame.pack(fill=tk.X, padx=(20, 0))

        # Blend mode selection
        blend_mode_frame = tk.Frame(self.blend_options_frame)
        blend_mode_frame.pack(anchor=tk.W, pady=2)
        tk.Label(blend_mode_frame, text="Blend Mode:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        self.blend_mode_var = tk.StringVar(value="auto")
        tk.Radiobutton(blend_mode_frame, text="Auto Config 2", variable=self.blend_mode_var,
                       value="auto", command=self.toggle_blend_mode).pack(side=tk.LEFT)
        tk.Radiobutton(blend_mode_frame, text="Manual", variable=self.blend_mode_var,
                       value="manual", command=self.toggle_blend_mode).pack(side=tk.LEFT)

        # Auto config 2 parameters frame
        self.auto_blend_frame = tk.Frame(self.blend_options_frame)
        self.auto_blend_frame.pack(fill=tk.X, padx=(20, 0))

        # Sobel kernel size (for auto config 2)
        auto_sobel_frame = tk.Frame(self.auto_blend_frame)
        auto_sobel_frame.pack(anchor=tk.W, pady=2)
        tk.Label(auto_sobel_frame, text="Sobel Kernel Size:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        self.auto_sobel_kernel_var = tk.IntVar(value=5)
        tk.Entry(auto_sobel_frame, textvariable=self.auto_sobel_kernel_var, width=10).pack(side=tk.LEFT, padx=5)

        # Use vignette (for auto config 2)
        auto_vig_frame = tk.Frame(self.auto_blend_frame)
        auto_vig_frame.pack(anchor=tk.W, pady=2)
        self.auto_use_vignette_var = tk.BooleanVar(value=True)
        tk.Checkbutton(auto_vig_frame, text="Use vignette", variable=self.auto_use_vignette_var).pack(side=tk.LEFT)

        # Manual blend config options
        self.manual_blend_frame = tk.Frame(self.blend_options_frame)
        self.manual_blend_frame.pack(fill=tk.X, padx=(20, 0))

        # Use vignette
        vig_frame = tk.Frame(self.manual_blend_frame)
        vig_frame.pack(anchor=tk.W, pady=2)
        self.use_vignette_var = tk.BooleanVar(value=True)
        tk.Checkbutton(vig_frame, text="Use vignette", variable=self.use_vignette_var).pack(side=tk.LEFT)

        # Sobel kernel size
        sobel_frame = tk.Frame(self.manual_blend_frame)
        sobel_frame.pack(anchor=tk.W, pady=2)
        tk.Label(sobel_frame, text="Sobel Kernel Size:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        self.sobel_kernel_var = tk.IntVar(value=5)
        tk.Entry(sobel_frame, textvariable=self.sobel_kernel_var, width=10).pack(side=tk.LEFT, padx=5)

        # Min blur diameter
        minblur_frame = tk.Frame(self.manual_blend_frame)
        minblur_frame.pack(anchor=tk.W, pady=2)
        tk.Label(minblur_frame, text="Min Blur Diameter:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        self.min_blur_var = tk.IntVar(value=1)
        tk.Entry(minblur_frame, textvariable=self.min_blur_var, width=10).pack(side=tk.LEFT, padx=5)

        # Max blur diameter
        maxblur_frame = tk.Frame(self.manual_blend_frame)
        maxblur_frame.pack(anchor=tk.W, pady=2)
        tk.Label(maxblur_frame, text="Max Blur Diameter:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        self.max_blur_var = tk.IntVar(value=10)
        tk.Entry(maxblur_frame, textvariable=self.max_blur_var, width=10).pack(side=tk.LEFT, padx=5)

        self.toggle_blend_mode()  # Initialize visibility

        # Texture variants selection
        variants_frame = tk.LabelFrame(params_frame, text="Texture Variants to Use", padx=5, pady=5)
        variants_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
            var = tk.BooleanVar(value=(key == "original"))
            self.variant_vars[key] = var
            cb = tk.Checkbutton(variants_frame, text=text, variable=var)
            cb.pack(anchor=tk.W)

        # Select all / none buttons
        btn_frame = tk.Frame(variants_frame)
        btn_frame.pack(anchor=tk.W, pady=5)
        tk.Button(btn_frame, text="Select All", command=self.select_all_variants,
                  padx=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Only Original", command=self.select_only_original,
                  padx=10).pack(side=tk.LEFT, padx=2)

    def toggle_blend_options(self):
        """Enable/disable blend configuration options"""
        if self.enable_blend_var.get():
            for child in self.blend_options_frame.winfo_children():
                self.enable_widget(child)
            self.toggle_blend_mode()
        else:
            for child in self.blend_options_frame.winfo_children():
                self.disable_widget(child)

    def toggle_blend_mode(self):
        """Show/hide manual blend options based on mode selection"""
        if not self.enable_blend_var.get():
            return

        if self.blend_mode_var.get() == "manual":
            # Hide auto, show manual
            for child in self.auto_blend_frame.winfo_children():
                self.disable_widget(child)
            for child in self.manual_blend_frame.winfo_children():
                self.enable_widget(child)
        else:  # auto
            # Show auto, hide manual
            for child in self.auto_blend_frame.winfo_children():
                self.enable_widget(child)
            for child in self.manual_blend_frame.winfo_children():
                self.disable_widget(child)


    #region Fetch Inputs Aux Funcs  START_______________

    @abstractmethod
    def on_auto_block_size(self, src_img_f32: ndarray):
        pass

    @abstractmethod
    def on_auto_overlap_size(self, block_size):
        pass

    def get_lookup_textures(self, src_img_bgr: ndarray) -> list[ndarray]:
        lookup_textures = []
        selected_variants = [key for key, var in self.variant_vars.items() if var.get()]

        # This check should never trigger due to pre-check in generate_texture(),
        # but kept as a safety measure
        if not selected_variants:
            raise ValueError("Please select at least one texture variant")

        for variant in selected_variants:
            match variant:
                case "original":
                    lookup_textures.append(src_img_bgr); break
                case "flip_h":
                    lookup_textures.append(np.fliplr(src_img_bgr)); break
                case "flip_v":
                    lookup_textures.append(np.flipud(src_img_bgr)); break
                case "flip_both":
                    lookup_textures.append(np.flipud(np.fliplr(src_img_bgr))); break
                case "transpose":
                    lookup_textures.append(np.transpose(src_img_bgr, (1, 0, 2))); break
                case "transpose_flip_h":
                    transposed = np.transpose(src_img_bgr, (1, 0, 2))
                    lookup_textures.append(np.fliplr(transposed))
                    break
                case "transpose_flip_v":
                    transposed = np.transpose(src_img_bgr, (1, 0, 2))
                    lookup_textures.append(np.flipud(transposed))
                    break
                case "transpose_flip_both":
                    transposed = np.transpose(src_img_bgr, (1, 0, 2))
                    lookup_textures.append(np.flipud(np.fliplr(transposed)))
        return lookup_textures

    def get_seam_method(self):
        match self.seam_method_var.get():
            case "astar":
                return get_min_cut_patch_mask_horizontal_astar
            case "purist":
                return get_min_cut_patch_mask_horizontal_jena2020
            case _:
                return ignore_min_cut_patch

    def get_overlap_size(self, block_size) -> NumPixels:
        overlap_str = self.overlap_var.get().strip().lower()
        if overlap_str == "auto":
            overlap = self.on_auto_overlap_size(block_size)
        else:
            try:
                overlap = int(overlap_str)
                if overlap < 0 or overlap >= block_size:
                    raise ValueError("Overlap must be between 0 and block_size")
            except ValueError as e:
                raise ValueError(f"Invalid overlap: {overlap_str}. Use 'auto' or an integer less than block_size.")
        return overlap

    def get_block_size(self, src_img_float: ndarray) -> NumPixels:
        block_size_str = self.block_size_var.get().strip().lower()
        if block_size_str == "auto":
            block_size = self.on_auto_block_size(src_img_float)
        else:
            try:
                block_size = int(block_size_str)
                if block_size <= 0:
                    raise ValueError("Block size must be positive")
            except ValueError as e:
                raise ValueError(f"Invalid block size: {block_size_str}. Use 'auto' or a positive integer.")
        return block_size

    def get_blend_config(self, block_size, overlap) -> GenParams:
        blend_config = None
        if self.enable_blend_var.get():
            blend_mode = self.blend_mode_var.get()
            if blend_mode == "auto":
                sobel_kernel = self.auto_sobel_kernel_var.get()
                use_vignette = self.auto_use_vignette_var.get()
                blend_config = auto_blend_config_2(sobel_kernel, overlap, use_vignette)
            else:  # manual
                use_vignette = self.use_vignette_var.get()
                sobel_kernel = self.sobel_kernel_var.get()
                min_blur = self.min_blur_var.get()
                max_blur = self.max_blur_var.get()

                blend_config = BlendConfig(
                    use_vignette=use_vignette,
                    sobel_kernel_size=sobel_kernel,
                    min_blur_diameter=min_blur,
                    max_blur_diameter=max_blur
                )
            blend_config = SquarePatchingBlendConfig(**asdict(blend_config))

        gen_params = GenParams(
            block_size=block_size,
            overlap=overlap,
            tolerance=self.tolerance_var.get(),
            vignette_on_match_template=self.vignette_match_var.get(),
            blend_config=blend_config,
            min_cut_search_method=self.get_seam_method()
        )
        return gen_params

    #endregion Fetch Inputs Aux Funcs END_______________


    @abstractmethod
    def process_texture(self) -> None:
        pass

    def exec_function(self):
        if self.src_img is None:
            return

        # Check if at least one texture variant is selected
        selected_variants = [key for key, var in self.variant_vars.items() if var.get()]
        if not selected_variants:
            messagebox.showwarning("No Variants Selected",
                                   "Please select at least one texture variant before generating.\n\n"
                                   "Click 'Only Original' to use just the source texture.")
            return

        self.exec_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_label.config(text="Generating texture...")

        # Run in separate thread to keep UI responsive
        thread = threading.Thread(target=self.process_texture)
        thread.start()
