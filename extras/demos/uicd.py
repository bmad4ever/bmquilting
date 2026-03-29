import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import cv2
from bmquilting.circular import generate_cphl6p, _generate_cphl6p, CircularPatchingConfig, CircularPatchParams
from bmquilting.utils.ui_coord import UiCoordData, JobMemoryManager, JobInterrupted
from numpy.random.bit_generator import SeedSequence

class UICDDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UICD Progress Demo")
        self.root.geometry("400x250")

        self.setup_ui()
        self.worker_thread = None
        self.monitor_thread = None
        self.jmm = None
        self.is_running = False

    def setup_ui(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="UICD Progress Demonstration", font=("Arial", 14, "bold")).pack(pady=(0, 20))

        # Progress Section
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.pack(pady=10)

        self.status_label = tk.Label(main_frame, text="Ready")
        self.status_label.pack()

        # Buttons
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=20)

        self.start_btn = tk.Button(btn_frame, text="Start Generation", command=self.start_generation, bg="#4CAF50", fg="white", width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.cancel_generation, state=tk.DISABLED, bg="#F44336", fg="white", width=15)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

    def start_generation(self):
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Starting...")
        self.progress_var.set(0)

        # Create dummy source texture (noise)
        source = np.random.rand(256, 256, 3).astype(np.float32)

        # 1. Define patch parameters
        params = CircularPatchParams(diameter=65, overlap_ratio=0.25)

        # 2. Configure: 0.1 tolerance and 1.12 spacing factor
        config = CircularPatchingConfig.with_seams(params, tolerance=0.1, spacing_factor=1.12)
        out_h, out_w = 512, 512

        # Step prediction for progress bar
        self.total_steps = generate_cphl6p.predict_steps(patching_config=config, out_h=out_h, out_w=out_w)
        self.progress_bar.config(maximum=self.total_steps)

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self.run_generation,
            args=([source], out_h, out_w, config)
        )
        self.worker_thread.start()

    def run_generation(self, sources, h, w, config):
        # We use JobMemoryManager to handle shared memory for progress tracking
        num_jobs = 4 # n_processes=1 usually needs 4 slots according to get_number_of_jobs_for(1)
        with JobMemoryManager(num_jobs=num_jobs) as jmm:
            self.jmm = jmm
            uicd = UiCoordData(jmm.name, job_id=0)

            # Start monitoring in a separate thread
            self.monitor_thread = threading.Thread(target=self.monitor_progress, args=(jmm,))
            self.monitor_thread.start()

            def slow_record(jid, pp):
                # This is called for each patch. Sleep to make the demo visible.
                time.sleep(0.05)

            try:
                # Call the internal variant to pass _record
                texture, seams = _generate_cphl6p(
                    source_textures=sources,
                    out_h=h, out_w=w,
                    patching_config=config,
                    seed=42,
                    n_processes=1,
                    uicd=uicd,
                    _record=slow_record
                )
                self.root.after(0, lambda: self.finish_generation("Success!"))
            except JobInterrupted:
                self.root.after(0, lambda: self.finish_generation("Cancelled"))
            except Exception as e:
                self.root.after(0, lambda: self.finish_generation(f"Error: {str(e)}"))
            finally:
                self.is_running = False

    def monitor_progress(self, jmm):
        while self.is_running:
            if jmm.is_interrupted():
                break

            progress = jmm.get_progress()
            self.root.after(0, lambda p=progress: self.update_progress(p, self.total_steps))
            time.sleep(0.1)

    def update_progress(self, current, total):
        self.progress_var.set(current)
        self.status_label.config(text=f"Progress: {current} / {total} patches")

    def cancel_generation(self):
        if self.jmm:
            self.jmm.stop_all()
            self.status_label.config(text="Cancelling...")
            self.cancel_btn.config(state=tk.DISABLED)

    def finish_generation(self, message):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.status_label.config(text=message)
        if message == "Success!":
            messagebox.showinfo("Done", "Generation completed successfully!")
        elif message == "Cancelled":
            messagebox.showwarning("Interrupted", "Generation was cancelled by user.")

if __name__ == "__main__":
    root = tk.Tk()
    app = UICDDemoApp(root)
    root.mainloop()
