"""
Control panel UI component.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from app.config import PLOT_TYPES, SIZE_PRESETS
from app.utils.ui_utils import validate_float

class ControlPanel:
    """Control panel UI component."""
    
    def __init__(self, parent, file_controller, plot_controller):
        """
        Initialize the control panel.
        
        Args:
            parent: Parent frame
            file_controller: FileController instance
            plot_controller: PlotController instance
        """
        self.parent = parent
        self.file_controller = file_controller
        self.plot_controller = plot_controller
        self.model = file_controller.model
        
        # UI variables
        self.metric_vars = {}
        self.metric_rename_entries = {}
        self.model_rename_entries = {}
        self.plot_type_var = tk.StringVar(value="Bar Plot")
        self.custom_y_range_var = tk.BooleanVar(value=False)
        self.show_outliers_var = tk.BooleanVar(value=True)
        
        # UI components
        self._create_control_panel()
    
    def _create_control_panel(self):
        """Create the control panel UI."""
        # Canvas and Scrollbar for Control Panel
        self.control_canvas = tk.Canvas(self.parent, borderwidth=0, background="#ffffff")
        self.control_frame = ttk.Frame(self.control_canvas, padding="10")  # Widgets go here
        self.scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_canvas_window = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        
        self.control_frame.bind("<Configure>", self._on_control_frame_configure)
        self.control_canvas.bind('<Enter>', self._bind_mousewheel)
        self.control_canvas.bind('<Leave>', self._unbind_mousewheel)
        
        self._create_control_widgets()
    
    def _on_control_frame_configure(self, event=None):
        """Update the scrollregion to encompass the inner frame."""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
    
    def _bind_mousewheel(self, event):
        """Bind mousewheel events to scroll the canvas."""
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # For Windows/Linux
        self.control_canvas.bind_all("<Button-4>", self._on_mousewheel)  # For Linux scroll up
        self.control_canvas.bind_all("<Button-5>", self._on_mousewheel)  # For Linux scroll down
    
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel events."""
        self.control_canvas.unbind_all("<MouseWheel>")
        self.control_canvas.unbind_all("<Button-4>")
        self.control_canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event):
        """Handle mousewheel events."""
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.control_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.control_canvas.yview_scroll(1, "units")
    
    def _create_control_widgets(self):
        """Create control panel widgets."""
        frame = self.control_frame  # Add widgets to this frame
        
        # 1. File Selection
        ttk.Label(frame, text="1. Select Evaluation Files", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.file_listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, height=7, width=45)
        self.file_listbox.pack(fill=tk.X, expand=True, pady=(0, 5))
        browse_button = ttk.Button(frame, text="Browse Files", command=self._browse_files)
        browse_button.pack(fill=tk.X, pady=(0, 5))
        load_button = ttk.Button(frame, text="Load & Process Selected Files", command=self._load_and_process_files)
        load_button.pack(fill=tk.X, pady=(0, 10))
        
        # 2. Metric Selection
        ttk.Label(frame, text="2. Select Metrics to Plot (Y-axis)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.metric_selection_frame = ttk.Frame(frame)  # Dynamic checkbuttons here
        self.metric_selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 3. Plot Type
        ttk.Label(frame, text="3. Select Plot Type", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        plot_type_combo = ttk.Combobox(frame, textvariable=self.plot_type_var, values=PLOT_TYPES, state="readonly")
        plot_type_combo.pack(fill=tk.X, pady=(0, 10))
        
        # 4. Y-Axis Range (Zooming)
        ttk.Label(frame, text="4. Y-Axis Range", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        ttk.Checkbutton(frame, text="Use Custom Y-Axis Range", variable=self.custom_y_range_var).pack(anchor="w")
        
        y_range_frame = ttk.Frame(frame)
        y_range_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(y_range_frame, text="Min:").pack(side=tk.LEFT, padx=(0, 5))
        vcmd = (self.parent.register(validate_float), '%P')
        self.y_min_entry = ttk.Entry(y_range_frame, width=8, validate="key", validatecommand=vcmd)
        self.y_min_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(y_range_frame, text="Max:").pack(side=tk.LEFT, padx=(0, 5))
        self.y_max_entry = ttk.Entry(y_range_frame, width=8, validate="key", validatecommand=vcmd)
        self.y_max_entry.pack(side=tk.LEFT)
        
        # 5. Plot Size
        ttk.Label(frame, text="5. Plot Size", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        size_frame = ttk.Frame(frame)
        size_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(size_frame, text="Width (inches):").pack(side=tk.LEFT, padx=(0, 5))
        self.plot_width_entry = ttk.Entry(size_frame, width=6, validate="key", validatecommand=vcmd)
        self.plot_width_entry.insert(0, "10")  # Default width
        self.plot_width_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(size_frame, text="Height (inches):").pack(side=tk.LEFT, padx=(0, 5))
        self.plot_height_entry = ttk.Entry(size_frame, width=6, validate="key", validatecommand=vcmd)
        self.plot_height_entry.insert(0, "6")  # Default height
        self.plot_height_entry.pack(side=tk.LEFT)
        
        # Size presets - use a scrollable frame to prevent overflow
        preset_label = ttk.Label(frame, text="Size Presets:")
        preset_label.pack(anchor="w", pady=(0, 5))
        
        presets_container = ttk.Frame(frame)
        presets_container.pack(fill=tk.X, pady=(0, 10))
        
        # Create two rows for presets to avoid horizontal overflow
        row1 = ttk.Frame(presets_container)
        row1.pack(fill=tk.X, pady=(0, 5))
        row2 = ttk.Frame(presets_container)
        row2.pack(fill=tk.X)
        
        preset_items = list(SIZE_PRESETS.items())
        for i, (preset_name, (w, h)) in enumerate(preset_items[:3]):
            preset_btn = ttk.Button(row1, text=preset_name, 
                                  command=lambda w=w, h=h: self._set_size_preset(w, h))
            preset_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            
        for i, (preset_name, (w, h)) in enumerate(preset_items[3:]):
            preset_btn = ttk.Button(row2, text=preset_name, 
                                  command=lambda w=w, h=h: self._set_size_preset(w, h))
            preset_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        # 6. Plot Title Settings
        ttk.Label(frame, text="6. Plot Title Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        title_frame = ttk.Frame(frame)
        title_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(title_frame, text="Main Title:").pack(side=tk.LEFT, padx=(0, 5))
        self.title_entry = ttk.Entry(title_frame)
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        subtitle_frame = ttk.Frame(frame)
        subtitle_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(subtitle_frame, text="Subtitle:").pack(side=tk.LEFT, padx=(0, 5))
        self.subtitle_entry = ttk.Entry(subtitle_frame)
        self.subtitle_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 7. Box Plot Settings
        ttk.Label(frame, text="7. Box Plot Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Checkbutton(frame, text="Show Outliers", variable=self.show_outliers_var).pack(anchor="w", pady=(0, 10))
        
        # 8. Customize Labels (Optional)
        ttk.Label(frame, text="8. Customize Labels (Optional)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        ttk.Label(frame, text="Metric Display Names:").pack(anchor="w")
        self.metric_rename_frame = ttk.Frame(frame)  # Dynamic entries here
        self.metric_rename_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(frame, text="Model/Experiment Display Names:").pack(anchor="w")
        self.model_rename_frame = ttk.Frame(frame)  # Dynamic entries here
        self.model_rename_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 9. Generate Plot
        generate_button = ttk.Button(frame, text="Generate Plot", command=self._generate_plot)
        generate_button.pack(fill=tk.X, pady=(10, 0))
    
    def _browse_files(self):
        """Browse and select files."""
        files = self.file_controller.browse_files()
        if files:
            self.file_listbox.delete(0, tk.END)
            for filename in self.file_controller.get_selected_filenames():
                self.file_listbox.insert(tk.END, filename)
    
    def _load_and_process_files(self):
        """Load and process selected files."""
        if self.file_controller.load_and_process_files():
            self._update_ui_after_load()
    
    def _update_ui_after_load(self):
        """Update UI components after loading data."""
        # Clear old metric selectors and rename entries
        for widget in self.metric_selection_frame.winfo_children(): widget.destroy()
        self.metric_vars.clear()
        for widget in self.metric_rename_frame.winfo_children(): widget.destroy()
        self.metric_rename_entries.clear()
        for widget in self.model_rename_frame.winfo_children(): widget.destroy()
        self.model_rename_entries.clear()
        
        # Populate metric selectors and rename fields
        metrics = self.model.get_metrics()
        if not metrics:
            ttk.Label(self.metric_selection_frame, text="No numeric metrics found.").pack()
        else:
            for metric in metrics:
                var = tk.BooleanVar(value=False)
                cb = ttk.Checkbutton(self.metric_selection_frame, text=metric, variable=var)
                cb.pack(anchor="w")
                self.metric_vars[metric] = var
                
                # Metric rename field
                row_frame = ttk.Frame(self.metric_rename_frame)
                row_frame.pack(fill=tk.X)
                ttk.Label(row_frame, text=f"{metric[:25]}:", width=28, anchor='w').pack(side=tk.LEFT)  # Truncate long original names
                entry = ttk.Entry(row_frame, width=25)
                entry.insert(0, metric)  # Default to original name
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.metric_rename_entries[metric] = entry
        
        # Populate model rename fields
        for exp_name in self.model.get_experiments():
            row_frame = ttk.Frame(self.model_rename_frame)
            row_frame.pack(fill=tk.X)
            ttk.Label(row_frame, text=f"{exp_name[:25]}:", width=28, anchor='w').pack(side=tk.LEFT)
            entry = ttk.Entry(row_frame, width=25)
            entry.insert(0, exp_name)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.model_rename_entries[exp_name] = entry
        
        self.control_frame.update_idletasks()
        self._on_control_frame_configure()
    
    def _set_size_preset(self, width, height):
        """Set the plot size entries to a preset value."""
        self.plot_width_entry.delete(0, tk.END)
        self.plot_width_entry.insert(0, str(width))
        self.plot_height_entry.delete(0, tk.END)
        self.plot_height_entry.insert(0, str(height))
    
    def _generate_plot(self):
        """Generate plot based on current settings."""
        if not self.model.experiment_data:
            messagebox.showerror("Error", "No data loaded. Please load and process files first.")
            return
        
        selected_metrics = [m for m, v in self.metric_vars.items() if v.get()]
        if not selected_metrics:
            messagebox.showerror("Error", "Please select at least one metric to plot.")
            return
        
        # Create a dict of display names for metrics
        display_metrics = {}
        for metric, entry in self.metric_rename_entries.items():
            display_name = entry.get().strip()
            if display_name:  # Only include non-empty names
                display_metrics[metric] = display_name
        
        # Create a dict of display names for experiments
        display_experiments = {}
        for exp_name, entry in self.model_rename_entries.items():
            display_name = entry.get().strip()
            if display_name:  # Only include non-empty names
                display_experiments[exp_name] = display_name
        
        # Get plot size from UI
        try:
            plot_width = float(self.plot_width_entry.get() or "10")
            plot_height = float(self.plot_height_entry.get() or "6")
            # Enforce minimum size to prevent tiny plots
            plot_width = max(4, plot_width)
            plot_height = max(3, plot_height)
        except ValueError:
            plot_width, plot_height = 10, 6  # Default if invalid input
        
        # Create plot parameters
        plot_params = {
            'plot_type': self.plot_type_var.get(),
            'selected_metrics': selected_metrics,
            'display_metrics': display_metrics,
            'display_experiments': display_experiments,
            'title': self.title_entry.get().strip(),
            'subtitle': self.subtitle_entry.get().strip(),
            'use_custom_y_range': self.custom_y_range_var.get(),
            'y_min': self.y_min_entry.get(),
            'y_max': self.y_max_entry.get(),
            'width': plot_width,
            'height': plot_height,
            'show_outliers': self.show_outliers_var.get()
        }
        
        # Emit plot generation event
        self.on_generate_plot(plot_params)
    
    def on_generate_plot(self, plot_params):
        """Callback function that will be overridden by the main application."""
        pass 