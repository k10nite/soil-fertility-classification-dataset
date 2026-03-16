"""
Soil Fertility Dataset Augmentation - GUI Application
User-friendly interface with manual operation selection
"""

import os
import sys
import threading
import logging
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict
import queue
from datetime import datetime

# Add search paths for operation_based_pipeline.py
# Priority 1: Same directory as this script (for portable/standalone tool)
# Priority 2: Dataset root directory (for integrated use)
script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent

# Try same directory first (portable), then parent directory (integrated)
sys.path.insert(0, str(script_dir))
sys.path.insert(1, str(parent_dir))

# Try importing dependencies
try:
    from operation_based_pipeline import (
        PipelineSettings,
        OperationPipeline,
        OperationRegistry,
        ImageProcessor
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure operation_based_pipeline.py is either:")
    print(f"  1. In the same folder as this GUI: {script_dir}")
    print(f"  2. In the dataset root directory: {parent_dir}")
    sys.exit(1)


class TextQueueHandler(logging.Handler):
    """Logging handler that sends logs to a queue"""
    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)


class AugmentationGUI:
    """Main GUI application with operation selection"""

    def __init__(self, root):
        self.root = root
        self.root.title("Soil Fertility Augmentation Tool")
        self.root.geometry("1100x800")

        # Variables
        self.input_dir_var = StringVar()
        self.output_dir_var = StringVar()
        self.res_1920_var = BooleanVar(value=True)
        self.res_1280_var = BooleanVar(value=False)
        self.res_640_var = BooleanVar(value=False)
        self.status_var = StringVar(value="Ready")
        self.op_progress_var = DoubleVar(value=0)
        self.overall_progress_var = DoubleVar(value=0)

        self.running = False

        # Category checkboxes - select entire operation types
        self.category_vars = {}  # {category_name: BooleanVar}
        self.operations_by_category = {}  # Will be populated by load_operations()

        # Create log handler
        self.log_handler = TextQueueHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        self.log_handler.setFormatter(formatter)

        # Load operations from registry
        self.load_operations()

        self.build_ui()

        # Start log updater
        self.update_logs()

    def load_operations(self):
        """Load operations from the pipeline registry"""
        # Initialize empty category dict (in case of error)
        self.operations_by_category = {
            'Brightness': [],
            'Contrast': [],
            'Rotation': [],
            'Flip': [],
            'Hue Shift': [],
            'Saturation': [],
            'CLAHE': [],
            'Noise': [],
            'Perspective': [],
            'Blur': [],
            'Sharpen': []
        }

        try:
            # Create a temporary registry to get operation list
            temp_settings = PipelineSettings(
                INPUT_DIR=".",
                OUTPUT_BASE=".",
                TARGET_SIZE=(1920, 1080)
            )
            temp_registry = OperationRegistry(temp_settings)

            # Categorize operations
            for op in temp_registry.operations:
                if 'bright' in op.name:
                    self.operations_by_category['Brightness'].append(op)
                elif 'contrast' in op.name:
                    self.operations_by_category['Contrast'].append(op)
                elif 'rot' in op.name:
                    self.operations_by_category['Rotation'].append(op)
                elif 'flip' in op.name:
                    self.operations_by_category['Flip'].append(op)
                elif 'hue' in op.name:
                    self.operations_by_category['Hue Shift'].append(op)
                elif 'sat' in op.name:
                    self.operations_by_category['Saturation'].append(op)
                elif 'clahe' in op.name:
                    self.operations_by_category['CLAHE'].append(op)
                elif 'noise' in op.name:
                    self.operations_by_category['Noise'].append(op)
                elif 'perspective' in op.name:
                    self.operations_by_category['Perspective'].append(op)
                elif 'blur' in op.name:
                    self.operations_by_category['Blur'].append(op)
                elif 'sharpen' in op.name:
                    self.operations_by_category['Sharpen'].append(op)

            # Create checkbox variables for each category (not individual operations)
            for category, ops in self.operations_by_category.items():
                if ops:  # Only create checkbox if category has operations
                    self.category_vars[category] = BooleanVar(value=False)  # All unselected by default

        except Exception as e:
            self.log(f"Warning: Could not load operations: {e}", "WARNING")

    def build_ui(self):
        # Header
        header = Frame(self.root, bg="#2c3e50", height=70)
        header.pack(fill=X)
        header.pack_propagate(False)

        Label(header, text="🌱 Soil Fertility Augmentation",
              font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=(12,3))
        Label(header, text="Select operations and resolutions for batch processing",
              font=("Arial", 9), bg="#2c3e50", fg="#ecf0f1").pack()

        # Main container with two columns
        main_container = Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Left column - Settings and Operations
        left_col = Frame(main_container)
        left_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0,5))

        # Directories
        dir_frame = LabelFrame(left_col, text="Directories", padx=10, pady=10)
        dir_frame.pack(fill=X, pady=(0,10))

        Label(dir_frame, text="Input:", width=8, anchor=W).grid(row=0, column=0, sticky=W, pady=3)
        Entry(dir_frame, textvariable=self.input_dir_var, state='readonly', font=("Arial", 9)).grid(
            row=0, column=1, sticky=EW, padx=5)
        Button(dir_frame, text="Browse", command=self.browse_input, width=8).grid(row=0, column=2)

        Label(dir_frame, text="Output:", width=8, anchor=W).grid(row=1, column=0, sticky=W, pady=3)
        Entry(dir_frame, textvariable=self.output_dir_var, state='readonly', font=("Arial", 9)).grid(
            row=1, column=1, sticky=EW, padx=5)
        Button(dir_frame, text="Browse", command=self.browse_output, width=8).grid(row=1, column=2)

        dir_frame.columnconfigure(1, weight=1)

        # Resolutions
        res_frame = LabelFrame(left_col, text="Output Resolutions", padx=10, pady=8)
        res_frame.pack(fill=X, pady=(0,10))

        res_opts = Frame(res_frame)
        res_opts.pack(fill=X)
        Checkbutton(res_opts, text="1920×1080", variable=self.res_1920_var, font=("Arial", 9)).pack(side=LEFT, padx=10)
        Checkbutton(res_opts, text="1280×720", variable=self.res_1280_var, font=("Arial", 9)).pack(side=LEFT, padx=10)
        Checkbutton(res_opts, text="640×480", variable=self.res_640_var, font=("Arial", 9)).pack(side=LEFT, padx=10)

        # Operations List
        ops_frame = LabelFrame(left_col, text="Operation Types (Select types to include all variations)", padx=10, pady=10)
        ops_frame.pack(fill=BOTH, expand=True)

        # Control buttons
        control_frame = Frame(ops_frame)
        control_frame.pack(fill=X, pady=(0,5))

        Button(control_frame, text="✓ Select All", command=self.select_all_ops,
               bg="#27ae60", fg="white", font=("Arial", 9)).pack(side=LEFT, padx=2)
        Button(control_frame, text="✗ Deselect All", command=self.deselect_all_ops,
               bg="#e74c3c", fg="white", font=("Arial", 9)).pack(side=LEFT, padx=2)
        Button(control_frame, text="Geometric Only", command=lambda: self.select_category(['Rotation', 'Flip']),
               font=("Arial", 9)).pack(side=LEFT, padx=2)
        Button(control_frame, text="Photometric Only", command=lambda: self.select_category(['Brightness', 'Contrast', 'Hue Shift', 'Saturation']),
               font=("Arial", 9)).pack(side=LEFT, padx=2)

        # Scrollable operations list
        ops_canvas = Canvas(ops_frame, height=300)
        ops_scrollbar = Scrollbar(ops_frame, orient=VERTICAL, command=ops_canvas.yview)
        ops_scrollable = Frame(ops_canvas)

        ops_scrollable.bind("<Configure>", lambda e: ops_canvas.configure(scrollregion=ops_canvas.bbox("all")))
        ops_canvas.create_window((0, 0), window=ops_scrollable, anchor=NW)
        ops_canvas.configure(yscrollcommand=ops_scrollbar.set)

        # Populate operation categories (not individual variations)
        for category, ops in self.operations_by_category.items():
            if ops:
                # Create checkbox for entire category
                if category in self.category_vars:
                    # Build description of variations
                    variations_text = f"({len(ops)} variations)"

                    cb = Checkbutton(ops_scrollable,
                                   text=f"{category} {variations_text}",
                                   variable=self.category_vars[category],
                                   font=("Arial", 10),
                                   anchor=W)
                    cb.pack(fill=X, padx=10, pady=3)

        ops_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        ops_scrollbar.pack(side=RIGHT, fill=Y)

        # Right column - Progress and Log
        right_col = Frame(main_container, width=400)
        right_col.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5,0))
        right_col.pack_propagate(False)

        # Progress
        prog_frame = LabelFrame(right_col, text="Progress", padx=10, pady=10)
        prog_frame.pack(fill=X, pady=(0,10))

        Label(prog_frame, textvariable=self.status_var, font=("Arial", 9, "bold"), anchor=W).pack(fill=X, pady=(0,8))

        Label(prog_frame, text="Current Operation:", font=("Arial", 8)).pack(anchor=W)
        self.op_bar = ttk.Progressbar(prog_frame, variable=self.op_progress_var, maximum=100)
        self.op_bar.pack(fill=X, pady=(2,8))

        Label(prog_frame, text="Overall Progress:", font=("Arial", 8)).pack(anchor=W)
        self.overall_bar = ttk.Progressbar(prog_frame, variable=self.overall_progress_var, maximum=100)
        self.overall_bar.pack(fill=X, pady=2)

        # Log
        log_frame = LabelFrame(right_col, text="Log Output", padx=10, pady=10)
        log_frame.pack(fill=BOTH, expand=True)

        scroll = Scrollbar(log_frame)
        scroll.pack(side=RIGHT, fill=Y)

        self.log_text = Text(log_frame, height=15, bg="#1e1e1e", fg="#d4d4d4",
                            yscrollcommand=scroll.set, state='disabled', font=("Consolas", 8))
        self.log_text.pack(fill=BOTH, expand=True)
        scroll.config(command=self.log_text.yview)

        self.log_text.tag_config('INFO', foreground='#4ec9b0')
        self.log_text.tag_config('WARNING', foreground='#ce9178')
        self.log_text.tag_config('ERROR', foreground='#f48771')

        # Bottom buttons
        btn_frame = Frame(self.root, pady=10)
        btn_frame.pack(fill=X, padx=15)

        self.start_btn = Button(btn_frame, text="▶ Run Selected Operations",
                                command=self.start_aug, bg="#27ae60", fg="white",
                                font=("Arial", 12, "bold"), padx=25, pady=10)
        self.start_btn.pack(side=LEFT, padx=5)

        # Show selected count
        self.selected_count_label = Label(btn_frame, text="No operations selected",
                                          font=("Arial", 9), fg="#555")
        self.selected_count_label.pack(side=LEFT, padx=15)

        Button(btn_frame, text="Exit", command=self.root.quit,
              bg="#95a5a6", fg="white", font=("Arial", 10), padx=20, pady=10).pack(side=RIGHT, padx=5)

        # Update selected count periodically
        self.update_selected_count()

    def select_all_ops(self):
        """Select all operation categories"""
        for var in self.category_vars.values():
            var.set(True)
        self.log("Selected all operation types", "INFO")

    def deselect_all_ops(self):
        """Deselect all operation categories"""
        for var in self.category_vars.values():
            var.set(False)
        self.log("Deselected all operation types", "INFO")

    def select_category(self, categories: List[str]):
        """Select only specific operation categories"""
        self.deselect_all_ops()
        for category in categories:
            if category in self.category_vars:
                self.category_vars[category].set(True)
        self.log(f"Selected: {', '.join(categories)}", "INFO")

    def update_selected_count(self):
        """Update the count of selected operation types and total variations"""
        # Count selected categories
        selected_categories = [cat for cat, var in self.category_vars.items() if var.get()]
        # Count total operations (all variations)
        total_ops = sum(len(self.operations_by_category[cat]) for cat in selected_categories)

        cat_count = len(selected_categories)
        if cat_count == 0:
            self.selected_count_label.config(text="No operations selected")
        else:
            self.selected_count_label.config(
                text=f"{cat_count} type{'s' if cat_count != 1 else ''} selected ({total_ops} total variations)"
            )
        self.root.after(500, self.update_selected_count)

    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Directory")
        if folder:
            self.input_dir_var.set(folder)
            self.log(f"Input: {folder}", "INFO")

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir_var.set(folder)
            self.log(f"Output: {folder}", "INFO")

    def log(self, msg, level="INFO"):
        self.log_handler.log_queue.put(f"[{level}] {msg}")

    def update_logs(self):
        try:
            while True:
                msg = self.log_handler.log_queue.get_nowait()

                tag = 'INFO'
                if 'ERROR' in msg:
                    tag = 'ERROR'
                elif 'WARNING' in msg:
                    tag = 'WARNING'

                self.log_text.config(state='normal')
                self.log_text.insert(END, msg + '\n', tag)
                self.log_text.see(END)
                self.log_text.config(state='disabled')
        except queue.Empty:
            pass

        self.root.after(100, self.update_logs)

    def validate(self):
        if not self.input_dir_var.get():
            messagebox.showerror("Error", "Select input directory")
            return False
        if not os.path.exists(self.input_dir_var.get()):
            messagebox.showerror("Error", "Input directory doesn't exist")
            return False
        if not self.output_dir_var.get():
            messagebox.showerror("Error", "Select output directory")
            return False
        if not any([self.res_1920_var.get(), self.res_1280_var.get(), self.res_640_var.get()]):
            messagebox.showerror("Error", "Select at least one resolution")
            return False

        # Check for images
        p = Path(self.input_dir_var.get())
        imgs = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
        if not imgs:
            messagebox.showerror("Error", "No images found in input directory")
            return False

        # Check selected categories
        selected_categories = [cat for cat, var in self.category_vars.items() if var.get()]
        if len(selected_categories) == 0:
            messagebox.showerror("Error", "Select at least one operation type to run")
            return False

        return True

    def start_aug(self):
        if self.running:
            messagebox.showwarning("Warning", "Already running")
            return

        if not self.validate():
            return

        # Get selected categories
        selected_categories = [cat for cat, var in self.category_vars.items() if var.get()]
        # Count total operations
        total_ops = sum(len(self.operations_by_category[cat]) for cat in selected_categories)

        # Confirm
        resolutions = []
        if self.res_1920_var.get():
            resolutions.append("1920x1080")
        if self.res_1280_var.get():
            resolutions.append("1280x720")
        if self.res_640_var.get():
            resolutions.append("640x480")

        msg = f"Run augmentation with selected settings?\n\n"
        msg += f"Input: {self.input_dir_var.get()}\n"
        msg += f"Output: {self.output_dir_var.get()}\n"
        msg += f"Resolutions: {', '.join(resolutions)}\n"
        msg += f"Operation Types: {', '.join(selected_categories)}\n"
        msg += f"Total Variations: {total_ops}\n\n"
        msg += f"This may take a while depending on the number of images and operations."

        if not messagebox.askyesno("Confirm", msg):
            return

        # Clear log
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, END)
        self.log_text.config(state='disabled')

        self.op_progress_var.set(0)
        self.overall_progress_var.set(0)

        self.running = True
        self.start_btn.config(state='disabled')

        # Start thread
        thread = threading.Thread(target=self.run_augmentation,
                                 args=(resolutions, selected_categories), daemon=True)
        thread.start()

    def run_augmentation(self, res_names, selected_categories):
        try:
            resolutions = []
            if "1920x1080" in res_names:
                resolutions.append((1920, 1080, "1920x1080"))
            if "1280x720" in res_names:
                resolutions.append((1280, 720, "1280x720"))
            if "640x480" in res_names:
                resolutions.append((640, 480, "640x480"))

            total_res = len(resolutions)

            # Count total operations from selected categories
            total_ops = sum(len(self.operations_by_category[cat]) for cat in selected_categories)

            self.log(f"Starting augmentation: {len(selected_categories)} types ({total_ops} variations) × {total_res} resolution(s)", "INFO")

            for idx, (width, height, name) in enumerate(resolutions):
                self.status_var.set(f"Processing {name}... ({idx+1}/{total_res})")
                self.log(f"\n{'='*60}", "INFO")
                self.log(f"Resolution: {name}", "INFO")
                self.log(f"{'='*60}\n", "INFO")

                output_dir = Path(self.output_dir_var.get()) / name

                settings = PipelineSettings(
                    INPUT_DIR=self.input_dir_var.get(),
                    OUTPUT_BASE=str(output_dir),
                    TARGET_SIZE=(width, height),
                    BACKGROUND_COLOR=(0, 0, 0)
                )

                # Create full registry
                registry = OperationRegistry(settings)

                # Filter to selected categories only - include ALL variations
                selected_operations = []
                for op in registry.operations:
                    # Check which category this operation belongs to
                    for category in selected_categories:
                        if category in self.operations_by_category:
                            category_ops = self.operations_by_category[category]
                            # Check if this operation is in the selected category
                            if any(cat_op.folder == op.folder and cat_op.name == op.name
                                   for cat_op in category_ops):
                                selected_operations.append(op)
                                break

                self.log(f"Running {len(selected_operations)} operation variations", "INFO")

                # Check for existing output (resume detection)
                completed_ops = []
                partial_ops = []
                for op in selected_operations:
                    check_folder = output_dir / op.folder
                    if check_folder.exists():
                        existing = list(check_folder.glob("*.png")) + list(check_folder.glob("*.jpg"))
                        if len(existing) > 0:
                            completed_ops.append(f"{op.folder} ({len(existing)} files)")

                if completed_ops:
                    self.log(f"RESUME MODE: Found existing output, will skip completed operations", "INFO")
                    self.log(f"Existing: {', '.join(completed_ops[:5])}" + (" ..." if len(completed_ops) > 5 else ""), "INFO")

                # Get input images
                input_path = Path(settings.INPUT_DIR)
                images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
                total_images = len(images)

                # Process each selected operation
                for op_idx, op in enumerate(selected_operations):
                    self.status_var.set(f"{name}: {op.folder}/{op.name}")

                    output_folder = output_dir / op.folder
                    output_folder.mkdir(parents=True, exist_ok=True)

                    # Check if operation already completed (resume functionality)
                    existing_files = list(output_folder.glob("*.png")) + list(output_folder.glob("*.jpg"))
                    if len(existing_files) >= total_images:
                        self.log(f"[{op_idx+1}/{len(selected_operations)}] ✓ SKIPPED: {op.folder}/{op.name} (already complete - {len(existing_files)} files)", "INFO")
                        # Update progress
                        overall = ((idx * len(selected_operations)) + (op_idx + 1)) / (total_res * len(selected_operations)) * 100
                        self.overall_progress_var.set(overall)
                        self.op_progress_var.set(100)
                        continue

                    self.log(f"[{op_idx+1}/{len(selected_operations)}] {op.folder}/{op.name} - {op.description}", "INFO")
                    if len(existing_files) > 0:
                        self.log(f"  → Resuming: {len(existing_files)}/{total_images} already processed", "INFO")

                    # Process images
                    processor = ImageProcessor(settings)
                    processed_count = 0
                    skipped_count = 0

                    for img_idx, img_path in enumerate(images):
                        # Check if output already exists (per-image resume)
                        output_name = f"{img_path.stem}_{op.name}.png"
                        output_file = output_folder / output_name

                        if output_file.exists():
                            skipped_count += 1
                            # Update operation progress
                            progress = ((img_idx + 1) / len(images)) * 100
                            self.op_progress_var.set(progress)
                            continue

                        # Load
                        img = processor.load_and_prepare_image(img_path)
                        if img is None:
                            continue

                        # Apply operation
                        result = op.pipeline(image=img)
                        augmented = result['image']

                        # Save
                        import cv2
                        cv2.imwrite(str(output_file), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                        processed_count += 1

                        # Update operation progress
                        progress = ((img_idx + 1) / len(images)) * 100
                        self.op_progress_var.set(progress)

                    # Log completion stats
                    if skipped_count > 0:
                        self.log(f"  → Completed: {processed_count} new, {skipped_count} skipped", "INFO")

                    # Update overall progress
                    overall = ((idx * total_ops) + (op_idx + 1)) / (total_res * total_ops) * 100
                    self.overall_progress_var.set(overall)

                self.log(f"Completed {name}!", "INFO")

            self.status_var.set("✓ All augmentations complete!")
            messagebox.showinfo("Success", f"Augmentation completed!\n\n{len(selected_categories)} types ({total_ops} variations) × {total_res} resolution(s)")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"ERROR: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            messagebox.showerror("Error", f"Augmentation failed:\n{e}")

        finally:
            self.running = False
            self.start_btn.config(state='normal')


def main():
    root = Tk()
    app = AugmentationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
