"""
Soil Fertility Dataset Augmentation - GUI Application
User-friendly interface with category-based operation selection
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
    """Main GUI application with category-based operation selection"""

    def __init__(self, root):
        self.root = root
        self.root.title("Soil Fertility Augmentation Tool")
        self.root.geometry("1100x800")

        # Set up proper window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variables
        self.input_dir_var = StringVar()
        self.output_dir_var = StringVar()
        self.status_var = StringVar(value="Ready")
        self.progress_var = DoubleVar(value=0)

        self.running = False

        # Operation selection variables
        self.operation_vars = {}  # {operation_name: BooleanVar}
        self.category_vars = {}   # {category_name: BooleanVar} for category checkboxes
        self.operations_by_category = {}  # {category_name: [operation_names]}

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
        try:
            all_ops = OperationRegistry.get_all_operations()

            # Group operations by category
            for op_name, op_info in all_ops.items():
                category = op_info['category']

                if category not in self.operations_by_category:
                    self.operations_by_category[category] = []

                self.operations_by_category[category].append(op_name)

            # Create checkbox variables for each operation
            for op_name in all_ops.keys():
                self.operation_vars[op_name] = BooleanVar(value=False)

            # Create category checkbox variables
            for category in self.operations_by_category.keys():
                self.category_vars[category] = BooleanVar(value=False)

        except Exception as e:
            self.log(f"Warning: Could not load operations: {e}", "WARNING")
            import traceback
            self.log(traceback.format_exc(), "ERROR")

    def build_ui(self):
        # Header
        header = Frame(self.root, bg="#2c3e50", height=70)
        header.pack(fill=X)
        header.pack_propagate(False)

        Label(header, text="🌱 Soil Fertility Augmentation",
              font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=(12,3))
        Label(header, text="Select augmentation operations for batch processing",
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

        # Operations List
        ops_frame = LabelFrame(left_col, text="Augmentation Operations", padx=10, pady=10)
        ops_frame.pack(fill=BOTH, expand=True)

        # Control buttons
        control_frame = Frame(ops_frame)
        control_frame.pack(fill=X, pady=(0,5))

        Button(control_frame, text="✓ Select All", command=self.select_all_ops,
               bg="#27ae60", fg="white", font=("Arial", 9)).pack(side=LEFT, padx=2)
        Button(control_frame, text="✗ Deselect All", command=self.deselect_all_ops,
               bg="#e74c3c", fg="white", font=("Arial", 9)).pack(side=LEFT, padx=2)
        Button(control_frame, text="Geometric Only", command=lambda: self.select_category_only(['Geometric']),
               font=("Arial", 9)).pack(side=LEFT, padx=2)
        Button(control_frame, text="Color Only", command=lambda: self.select_category_only(['Color']),
               font=("Arial", 9)).pack(side=LEFT, padx=2)

        # Scrollable operations list
        ops_canvas = Canvas(ops_frame, height=350)
        ops_scrollbar = Scrollbar(ops_frame, orient=VERTICAL, command=ops_canvas.yview)
        ops_scrollable = Frame(ops_canvas)

        ops_scrollable.bind("<Configure>", lambda e: ops_canvas.configure(scrollregion=ops_canvas.bbox("all")))
        ops_canvas.create_window((0, 0), window=ops_scrollable, anchor=NW)
        ops_canvas.configure(yscrollcommand=ops_scrollbar.set)

        # Populate operations grouped by category
        all_ops = OperationRegistry.get_all_operations()

        for category, op_names in self.operations_by_category.items():
            if op_names:
                # Category header with checkbox
                cat_frame = Frame(ops_scrollable, bg="#34495e", pady=5, padx=10)
                cat_frame.pack(fill=X, pady=(10, 0))

                cat_cb = Checkbutton(cat_frame,
                                   text=f"{category} ({len(op_names)} operations)",
                                   variable=self.category_vars[category],
                                   command=lambda c=category: self.toggle_category(c),
                                   font=("Arial", 11, "bold"),
                                   bg="#34495e",
                                   fg="white",
                                   selectcolor="#2c3e50",
                                   anchor=W)
                cat_cb.pack(fill=X)

                # Individual operations under category
                for op_name in op_names:
                    op_info = all_ops[op_name]
                    op_cb = Checkbutton(ops_scrollable,
                                       text=f"  • {op_info['name']}",
                                       variable=self.operation_vars[op_name],
                                       command=self.update_category_checkboxes,
                                       font=("Arial", 9),
                                       anchor=W)
                    op_cb.pack(fill=X, padx=20, pady=1)

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

        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=X, pady=2)

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

        self.start_btn = Button(btn_frame, text="▶ Run Augmentation",
                                command=self.start_aug, bg="#27ae60", fg="white",
                                font=("Arial", 12, "bold"), padx=25, pady=10)
        self.start_btn.pack(side=LEFT, padx=5)

        # Show selected count
        self.selected_count_label = Label(btn_frame, text="No operations selected",
                                          font=("Arial", 9), fg="#555")
        self.selected_count_label.pack(side=LEFT, padx=15)

        Button(btn_frame, text="Exit", command=self.on_closing,
              bg="#95a5a6", fg="white", font=("Arial", 10), padx=20, pady=10).pack(side=RIGHT, padx=5)

        # Update selected count periodically
        self.update_selected_count()

    def toggle_category(self, category):
        """Toggle all operations in a category"""
        state = self.category_vars[category].get()
        for op_name in self.operations_by_category[category]:
            self.operation_vars[op_name].set(state)

    def update_category_checkboxes(self):
        """Update category checkboxes based on individual operation selection"""
        for category, op_names in self.operations_by_category.items():
            selected = sum(1 for op in op_names if self.operation_vars[op].get())
            if selected == len(op_names):
                self.category_vars[category].set(True)
            elif selected == 0:
                self.category_vars[category].set(False)

    def select_all_ops(self):
        """Select all operations"""
        for var in self.operation_vars.values():
            var.set(True)
        for var in self.category_vars.values():
            var.set(True)
        self.log("Selected all operations", "INFO")

    def deselect_all_ops(self):
        """Deselect all operations"""
        for var in self.operation_vars.values():
            var.set(False)
        for var in self.category_vars.values():
            var.set(False)
        self.log("Deselected all operations", "INFO")

    def select_category_only(self, categories: List[str]):
        """Select only specific categories"""
        self.deselect_all_ops()
        for category in categories:
            if category in self.category_vars:
                self.category_vars[category].set(True)
                self.toggle_category(category)
        self.log(f"Selected: {', '.join(categories)}", "INFO")

    def update_selected_count(self):
        """Update the count of selected operations"""
        selected = sum(1 for var in self.operation_vars.values() if var.get())
        if selected == 0:
            self.selected_count_label.config(text="No operations selected")
        else:
            self.selected_count_label.config(
                text=f"{selected} operation{'s' if selected != 1 else ''} selected"
            )
        self.root.after(500, self.update_selected_count)

    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Directory")
        if folder:
            self.input_dir_var.set(folder)
            # Count images
            p = Path(folder)
            imgs = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
            self.log(f"Input: {folder} ({len(imgs)} images)", "INFO")

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

        # Check for images
        p = Path(self.input_dir_var.get())
        imgs = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
        if not imgs:
            messagebox.showerror("Error", "No images found in input directory")
            return False

        # Check selected operations
        selected = [op for op, var in self.operation_vars.items() if var.get()]
        if len(selected) == 0:
            messagebox.showerror("Error", "Select at least one operation to run")
            return False

        return True

    def start_aug(self):
        if self.running:
            messagebox.showwarning("Warning", "Already running")
            return

        if not self.validate():
            return

        # Get selected operations
        selected_ops = [op for op, var in self.operation_vars.items() if var.get()]

        # Confirm
        msg = f"Run augmentation with selected settings?\n\n"
        msg += f"Input: {self.input_dir_var.get()}\n"
        msg += f"Output: {self.output_dir_var.get()}\n"
        msg += f"Operations: {len(selected_ops)} selected\n\n"
        msg += f"This may take a while depending on the number of images and operations."

        if not messagebox.askyesno("Confirm", msg):
            return

        # Clear log
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, END)
        self.log_text.config(state='disabled')

        self.progress_var.set(0)

        self.running = True
        self.start_btn.config(state='disabled')

        # Start thread
        thread = threading.Thread(target=self.run_augmentation,
                                 args=(selected_ops,), daemon=True)
        thread.start()

    def run_augmentation(self, selected_ops):
        try:
            # Create settings
            settings = PipelineSettings(
                input_dir=self.input_dir_var.get(),
                output_dir=self.output_dir_var.get(),
                operations=selected_ops
            )

            self.log(f"Starting augmentation with {len(selected_ops)} operations", "INFO")

            # Create pipeline
            pipeline = OperationPipeline(settings)

            # Run with progress callback
            def progress_callback(current, total, filename):
                progress = (current / total) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Processing: {filename} ({current}/{total})")

            result = pipeline.run(progress_callback=progress_callback)

            # Show results
            self.status_var.set("✓ Augmentation complete!")
            self.progress_var.set(100)

            if result['success']:
                self.log(f"✓ Success: {result['processed']}/{result['total']} images processed", "INFO")
                if result['failed'] > 0:
                    self.log(f"⚠ {result['failed']} images failed", "WARNING")

                messagebox.showinfo("Success",
                    f"Augmentation completed!\n\n"
                    f"Processed: {result['processed']}/{result['total']} images\n"
                    f"Failed: {result['failed']} images")
            else:
                self.log(f"✗ Failed: {result['message']}", "ERROR")
                messagebox.showerror("Error", f"Augmentation failed:\n{result['message']}")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"ERROR: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            messagebox.showerror("Error", f"Augmentation failed:\n{e}")

        finally:
            self.running = False
            self.start_btn.config(state='normal')

    def on_closing(self):
        """Handle window close button - ensure clean shutdown"""
        if self.running:
            if messagebox.askyesno("Confirm Exit",
                                  "Augmentation is still running!\n\n"
                                  "Closing now will stop the process.\n"
                                  "Are you sure you want to exit?"):
                self.log("User requested exit during processing", "WARNING")
                self.running = False
                self.root.quit()
                self.root.destroy()
        else:
            self.root.quit()
            self.root.destroy()


def main():
    root = Tk()
    app = AugmentationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
