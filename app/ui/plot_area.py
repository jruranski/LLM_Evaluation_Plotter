"""
Plot area UI component.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

class PlotArea:
    """Plot area UI component."""
    
    def __init__(self, parent, file_controller, plot_controller):
        """
        Initialize the plot area.
        
        Args:
            parent: Parent frame
            file_controller: FileController instance
            plot_controller: PlotController instance
        """
        self.parent = parent
        self.file_controller = file_controller
        self.plot_controller = plot_controller
        
        self._create_plot_area()
    
    def _create_plot_area(self):
        """Create plot area UI."""
        # Main container for plot area with proper spacing
        plot_container = ttk.Frame(self.parent)
        plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas placeholder for plots
        self.plot_canvas_frame = ttk.Frame(plot_container)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Button container with fixed height
        button_frame = ttk.Frame(plot_container, height=30)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        button_frame.pack_propagate(False)  # Maintain fixed height
        
        # Add LaTeX export button
        latex_button = ttk.Button(button_frame, text="Export to LaTeX", command=self._export_to_latex)
        latex_button.pack(side=tk.RIGHT, padx=(0, 5))
        
        save_button = ttk.Button(button_frame, text="Save Plot", command=self._save_plot)
        save_button.pack(side=tk.RIGHT)
    
    def _save_plot(self):
        """Save the current plot to a file."""
        self.file_controller.save_plot(self.plot_controller.current_plot_fig)
    
    def _export_to_latex(self):
        """Export data to LaTeX table format."""
        # Check if there's any data to export
        if not hasattr(self, 'plot_params') or not self.plot_params:
            messagebox.showerror("Error", "No data available to export. Please generate a plot first.")
            return
        
        # Open dialog to configure export options
        export_window = tk.Toplevel(self.parent)
        export_window.title("Export to LaTeX")
        export_window.geometry("400x300")
        export_window.transient(self.parent)
        export_window.grab_set()
        export_window.resizable(False, False)
        
        # Frame for options
        options_frame = ttk.Frame(export_window, padding="10")
        options_frame.pack(fill=tk.BOTH, expand=True)
        
        # Data type selection
        ttk.Label(options_frame, text="Data Format:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        data_type_var = tk.StringVar(value="means")
        means_radio = ttk.Radiobutton(options_frame, text="Means only", variable=data_type_var, value="means")
        means_radio.grid(row=1, column=0, sticky="w")
        
        means_std_radio = ttk.Radiobutton(options_frame, text="Means with standard deviations", variable=data_type_var, value="means_with_std")
        means_std_radio.grid(row=2, column=0, sticky="w")
        
        test_case_radio = ttk.Radiobutton(options_frame, text="Per test case data", variable=data_type_var, value="per_test_case")
        test_case_radio.grid(row=3, column=0, sticky="w", pady=(0, 10))
        
        # Table caption and label
        ttk.Label(options_frame, text="Table Caption:").grid(row=4, column=0, sticky="w", pady=(0, 5))
        caption_entry = ttk.Entry(options_frame)
        caption_entry.insert(0, "Experimental Results")
        caption_entry.grid(row=5, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(options_frame, text="Table Label:").grid(row=6, column=0, sticky="w", pady=(0, 5))
        label_entry = ttk.Entry(options_frame)
        label_entry.insert(0, "tab:results")
        label_entry.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        
        # Buttons frame
        buttons_frame = ttk.Frame(options_frame)
        buttons_frame.grid(row=8, column=0, sticky="ew")
        
        # Define functions for the buttons
        def export_to_clipboard():
            export_params = {
                'data_type': data_type_var.get(),
                'selected_metrics': self.plot_params.get('selected_metrics', []),
                'display_metrics': self.plot_params.get('display_metrics', {}),
                'display_experiments': self.plot_params.get('display_experiments', {}),
                'caption': caption_entry.get(),
                'label': label_entry.get()
            }
            
            latex_code = self.plot_controller.export_to_latex(export_params)
            if latex_code:
                export_window.clipboard_clear()
                export_window.clipboard_append(latex_code)
                messagebox.showinfo("Success", "LaTeX table copied to clipboard.")
                export_window.destroy()
        
        def export_to_file():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".tex",
                filetypes=[("LaTeX files", "*.tex"), ("Text files", "*.txt"), ("All files", "*.*")],
                title="Save LaTeX Table As"
            )
            
            if not file_path:
                return
            
            export_params = {
                'data_type': data_type_var.get(),
                'selected_metrics': self.plot_params.get('selected_metrics', []),
                'display_metrics': self.plot_params.get('display_metrics', {}),
                'display_experiments': self.plot_params.get('display_experiments', {}),
                'caption': caption_entry.get(),
                'label': label_entry.get()
            }
            
            success = self.plot_controller.export_to_latex(export_params, file_path)
            if success:
                messagebox.showinfo("Success", f"LaTeX table saved to {file_path}")
                export_window.destroy()
        
        def cancel():
            export_window.destroy()
        
        # Add buttons
        clipboard_btn = ttk.Button(buttons_frame, text="Copy to Clipboard", command=export_to_clipboard)
        clipboard_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        file_btn = ttk.Button(buttons_frame, text="Save to File", command=export_to_file)
        file_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        cancel_btn = ttk.Button(buttons_frame, text="Cancel", command=cancel)
        cancel_btn.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
    
    def update_plot(self, plot_params):
        """
        Update the plot area with a new plot.
        
        Args:
            plot_params: Plot parameters dictionary
        """
        # Store the plot parameters for LaTeX export
        self.plot_params = plot_params
        
        # Generate the plot
        self.plot_controller.generate_plot(plot_params, self.plot_canvas_frame) 