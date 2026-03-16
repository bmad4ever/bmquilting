from bmquilting.square import SquarePatchingConfig, generate_texture, generate_guided
from bmquilting.guess_block_size import guess_nice_block_size
from bmquilting.misc.texture_utils import add_salt_and_pepper
from bmquilting.types import SquarePatchingBlendConfig
from bmquilting.synthesis_subroutines import ignore_min_cut_patch

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading



class GuidedTextureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Guided Texture Generator Demo")
        self.root.geometry("1400x900")

        self.src_img = None
        self.images = {}  # Store all generated images

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side - controls
        left_side = tk.Frame(main_container, width=350)
        left_side.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_side.pack_propagate(False)

        # Right side - image grid
        right_side = tk.Frame(main_container)
        right_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        # === LEFT SIDE CONTROLS ===
        control_frame = tk.Frame(left_side)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text="Select Image", command=self.select_image,
                  bg="#4CAF50", fg="white", padx=20, pady=5).pack(fill=tk.X, pady=2)

        self.file_label = tk.Label(control_frame, text="No image selected", fg="gray", wraplength=300)
        self.file_label.pack(fill=tk.X, pady=2)

        self.generate_btn = tk.Button(control_frame, text="Generate & Compare",
                                      command=self.generate_comparison, state=tk.DISABLED,
                                      bg="#2196F3", fg="white", padx=20, pady=8)
        self.generate_btn.pack(fill=tk.X, pady=5)

        self.progress_label = tk.Label(control_frame, text="", fg="blue")
        self.progress_label.pack(fill=tk.X, pady=2)

        # Parameters
        params_frame = tk.LabelFrame(left_side, text="Parameters", padx=10, pady=10)
        params_frame.pack(fill=tk.X, pady=10)

        # Output size
        size_frame = tk.Frame(params_frame)
        size_frame.pack(fill=tk.X, pady=3)
        tk.Label(size_frame, text="Output Size:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.out_size_var = tk.IntVar(value=512)
        tk.Entry(size_frame, textvariable=self.out_size_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(size_frame, text="px", fg="gray").pack(side=tk.LEFT)

        # Noise level
        noise_frame = tk.Frame(params_frame)
        noise_frame.pack(fill=tk.X, pady=3)
        tk.Label(noise_frame, text="Noise Level:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.noise_var = tk.DoubleVar(value=0.2)
        tk.Scale(noise_frame, from_=0.0, to=0.5, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.noise_var, length=150).pack(side=tk.LEFT)

        # Median blur iterations
        blur_frame = tk.Frame(params_frame)
        blur_frame.pack(fill=tk.X, pady=3)
        tk.Label(blur_frame, text="Blur Iterations:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.blur_iter_var = tk.IntVar(value=3)
        tk.Scale(blur_frame, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL,
                variable=self.blur_iter_var, length=150).pack(side=tk.LEFT)

        # Tolerance
        tol_frame = tk.Frame(params_frame)
        tol_frame.pack(fill=tk.X, pady=3)
        tk.Label(tol_frame, text="Tolerance:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=0.1)
        tk.Scale(tol_frame, from_=0.0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.tolerance_var, length=150).pack(side=tk.LEFT)

        # Seed
        seed_frame = tk.Frame(params_frame)
        seed_frame.pack(fill=tk.X, pady=3)
        tk.Label(seed_frame, text="Random Seed:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.seed_var = tk.IntVar(value=123)
        tk.Entry(seed_frame, textvariable=self.seed_var, width=8).pack(side=tk.LEFT, padx=5)

        # Info
        info_frame = tk.LabelFrame(left_side, text="About", padx=10, pady=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        info_text = ("This demo compares guided vs. non-guided texture synthesis.\n\n"
                     "Process:\n"
                     "1. Add salt & pepper noise to source\n"
                     "2. Create proxy via median blur\n"
                     "3. Generate with guidance (proxy)\n"
                     "4. Generate without guidance\n\n"
                     "Guided synthesis uses the proxy to help preserve structure, ignoring the noise in the source.")
        tk.Label(info_frame, text=info_text, justify=tk.LEFT, wraplength=300).pack()

        # === RIGHT SIDE IMAGE GRID ===
        # Create 4x2 grid for images
        self.canvas_frames = {}
        self.canvases = {}

        grid_labels = [
            ("src", "Source Image"),
            ("noisy", "Noisy Source"),
            ("proxy", "Proxy (Blurred)"),
            ("out_proxy", "Output Proxy"),
            ("no_guided", "Result: WITHOUT Guidance"),
            ("guided", "Result: WITH Guidance"),
        ]

        for i, (key, label) in enumerate(grid_labels):
            row = i // 2
            col = i % 2

            frame = tk.LabelFrame(right_side, text=label, padx=5, pady=5)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            canvas = tk.Canvas(frame, bg="white", width=300, height=200)
            canvas.pack(fill=tk.BOTH, expand=True)

            self.canvas_frames[key] = frame
            self.canvases[key] = canvas

        # Configure grid weights
        for i in range(4):
            right_side.grid_rowconfigure(i, weight=1)
        for i in range(2):
            right_side.grid_columnconfigure(i, weight=1)

        # Status bar
        self.status_label = tk.Label(self.root, text="Ready - Select an image to begin",
                                     bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Texture Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.src_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if self.src_img is None:
                    raise ValueError("Failed to load image")

                self.src_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB)
                filename = file_path.split("/")[-1]
                self.file_label.config(text=f"Loaded: {filename}", fg="black")
                self.generate_btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"Image loaded: {self.src_img.shape[1]}x{self.src_img.shape[0]}")

                # Display source
                self.display_image("src", self.src_img)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.status_label.config(text="Error loading image")

    def generate_comparison(self):
        if self.src_img is None:
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Generating textures...")
        self.progress_label.config(text="Processing...")

        thread = threading.Thread(target=self.process_comparison)
        thread.start()

    def process_comparison(self):
        try:
            # Prepare source
            src_float = self.src_img.astype(np.float32) / 255.0
            src_bgr = cv2.cvtColor(src_float, cv2.COLOR_RGB2BGR)

            # Parameters
            seed = self.seed_var.get()
            rand_gen = np.random.default_rng(seed=seed)
            out_size = self.out_size_var.get()

            # Block size calculation
            self.update_progress("Calculating block size...")
            grayscale = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
            block_size = guess_nice_block_size(grayscale, False)
            overlap = round(block_size / 3.0)

            # Blend config
            #blend_config = auto_blend_config_2(9, overlap, False)
            #**asdict(blend_config)
            blend_config = SquarePatchingBlendConfig(
                use_vignette=True
            )

            patching_config = SquarePatchingConfig(
                vignette_on_match_template=True,
                block_size=block_size,
                overlap=overlap,
                tolerance=self.tolerance_var.get(),
                blend_config=blend_config,
                min_cut_search_method=ignore_min_cut_patch
            )

            # Add noise
            self.update_progress("Adding noise...")
            noise_level = self.noise_var.get()
            noisy_src = add_salt_and_pepper(
                image=src_bgr,
                amount=noise_level,
                salt_vs_pepper=0.5,
                seed=seed)
            self.display_bgr_image("noisy", noisy_src)

            # Create proxy
            self.update_progress("Creating proxy (median blur)...")
            proxy = noisy_src.copy()
            blur_iters = self.blur_iter_var.get()
            for _ in range(blur_iters):
                proxy = cv2.medianBlur(proxy, 5)
            self.display_bgr_image("proxy", proxy)

            # Generate with guidance
            self.update_progress("Generating WITH guidance...")
            out_tex, out_cut, out_proxy = generate_guided(
                proxy_textures=[proxy],
                source_textures=[noisy_src],
                patching_config=patching_config,
                out_h=out_size,
                out_w=out_size,
                rng=rand_gen,
                uicd=None
            )
            self.display_bgr_image("out_proxy", out_proxy)
            self.display_bgr_image("guided", out_tex)

            # Generate without guidance
            self.update_progress("Generating WITHOUT guidance...")
            no_guide_tex, no_guide_cut = generate_texture(
                src_textures=[noisy_src],
                patching_config=patching_config,
                out_h=out_size,
                out_w=out_size,
                rng=rand_gen,
                uicd=None
            )
            self.display_bgr_image("no_guided", no_guide_tex)

            self.root.after(0, self.on_complete)

        except Exception as e:
            self.root.after(0, lambda: self.on_error(str(e)))

    def update_progress(self, msg):
        self.root.after(0, lambda: self.progress_label.config(text=msg))

    def display_image(self, key, img_array):
        """Display RGB image"""
        self.root.after(0, lambda: self._display_image_main_thread(key, img_array))

    def display_bgr_image(self, key, img_array):
        """Display BGR image (convert to RGB)"""
        img_rgb = (cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) * 255).astype(np.uint8)
        self.display_image(key, img_rgb)

    def _display_image_main_thread(self, key, img_array):
        canvas = self.canvases.get(key)
        if canvas is None:
            return

        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w <= 1:
            canvas_w, canvas_h = 300, 200

        # Convert to PIL
        if len(img_array.shape) == 2:
            pil_img = Image.fromarray(img_array, mode='L')
        else:
            pil_img = Image.fromarray(img_array, mode='RGB')

        # Resize to fit
        img_w, img_h = pil_img.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Display
        photo = ImageTk.PhotoImage(pil_img)
        self.images[key] = photo  # Keep reference
        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo)

    def on_complete(self):
        self.progress_label.config(text="Complete!", fg="green")
        self.generate_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Generation complete - Compare the results!")
        messagebox.showinfo("Success", "Texture comparison generated successfully!\n\n"
                                       "Compare the 'WITH Guidance' vs 'WITHOUT Guidance' results.")

    def on_error(self, error_msg):
        self.progress_label.config(text="Error!", fg="red")
        self.generate_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Generation failed")
        messagebox.showerror("Error", f"Failed to generate:\n{error_msg}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GuidedTextureApp(root)
    root.mainloop()