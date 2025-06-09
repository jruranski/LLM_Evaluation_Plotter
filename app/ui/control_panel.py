"""
Control panel UI component.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os

from app.config import PLOT_TYPES, SIZE_PRESETS, DEFAULT_FONT_SIZES, FONT_SIZE_PRESETS
from app.utils.ui_utils import validate_float

class ControlPanel:
    """Control panel UI component."""
    
    # Metric name translations
    METRIC_TRANSLATIONS = {
        "Contextual Metric Score": "Trafność kontekstu",
        "Contextual Precision Score": "Precyzja kontekstu",
        "Correctness Metric Score": "Poprawność odpowiedzi",
        "Faithfulness Metric Score": "Wierność odpowiedzi",
        "Metric Score": "Trafność odpowiedzi",
        "Test Case": "Test Case"
    }
    
    # LaTeX table types
    LATEX_TABLE_TYPES = [
        "Means only",
        "Means with standard deviations",
        "Per test case data"
    ]
    
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
        self.output_format_var = tk.StringVar(value="Plot")
        self.latex_table_type_var = tk.StringVar(value="Means only")
        self.show_titles_var = tk.BooleanVar(value=True)
        self.legend_position_var = tk.StringVar(value="best")
        self.legend_ncol_var = tk.StringVar(value="auto")
        self.legend_frameon_var = tk.BooleanVar(value=True)
        self.legend_alpha_var = tk.StringVar(value="0.9")
        
        # Font size variables
        self.font_preset_var = tk.StringVar(value="Large")
        self.font_size_vars = {}
        for key in DEFAULT_FONT_SIZES.keys():
            self.font_size_vars[key] = tk.StringVar(value=str(DEFAULT_FONT_SIZES[key]))
        
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
        
        # File operation buttons
        file_btn_frame = ttk.Frame(frame)
        file_btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        browse_button = ttk.Button(file_btn_frame, text="Add Files", command=self._browse_files)
        browse_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        remove_button = ttk.Button(file_btn_frame, text="Remove Selected", command=self._remove_selected_files)
        remove_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        clear_button = ttk.Button(file_btn_frame, text="Clear All", command=self._clear_files)
        clear_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        load_button = ttk.Button(frame, text="Load & Process Selected Files", command=self._load_and_process_files)
        load_button.pack(fill=tk.X, pady=(0, 10))
        
        # Configuration Save/Load Section
        ttk.Label(frame, text="Configuration Save/Load", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        config_btn_frame = ttk.Frame(frame)
        config_btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        save_config_button = ttk.Button(config_btn_frame, text="Save Configuration", command=self._save_config)
        save_config_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        load_config_button = ttk.Button(config_btn_frame, text="Load Configuration", command=self._load_config)
        load_config_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # 2. Metric Selection
        ttk.Label(frame, text="2. Select Metrics to Plot (Y-axis)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.metric_selection_frame = ttk.Frame(frame)  # Dynamic checkbuttons here
        self.metric_selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 3. Plot Type
        ttk.Label(frame, text="3. Select Plot Type", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        plot_type_combo = ttk.Combobox(frame, textvariable=self.plot_type_var, values=PLOT_TYPES, state="readonly")
        plot_type_combo.pack(fill=tk.X, pady=(0, 10))
        
        # 4. Font Size Settings
        ttk.Label(frame, text="4. Font Size Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Font size presets
        preset_frame = ttk.Frame(frame)
        preset_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(preset_frame, text="Font Size Preset:").pack(side=tk.LEFT, padx=(0, 5))
        font_preset_combo = ttk.Combobox(preset_frame, textvariable=self.font_preset_var, 
                                        values=list(FONT_SIZE_PRESETS.keys()), state="readonly")
        font_preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        font_preset_combo.bind('<<ComboboxSelected>>', self._on_font_preset_change)
        
        # Individual font size controls
        font_controls_frame = ttk.LabelFrame(frame, text="Individual Font Sizes", padding=5)
        font_controls_frame.pack(fill=tk.X, pady=(5, 10))
        
        # Create validation command for font size entries
        vcmd_int = (self.parent.register(self._validate_int), '%P')
        
        font_labels = {
            "title": "Plot Title:",
            "subtitle": "Subtitle:",
            "axis_title": "Axis Titles:",
            "axis_labels": "Axis Labels:",
            "legend": "Legend:",
            "tick_labels": "Tick Labels:",
            "annotations": "Annotations:"
        }
        
        # Create two columns for font controls
        left_col = ttk.Frame(font_controls_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        right_col = ttk.Frame(font_controls_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        font_keys = list(font_labels.keys())
        for i, key in enumerate(font_keys):
            parent_col = left_col if i < len(font_keys) // 2 + 1 else right_col
            
            row_frame = ttk.Frame(parent_col)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=font_labels[key], width=12).pack(side=tk.LEFT)
            entry = ttk.Entry(row_frame, textvariable=self.font_size_vars[key], width=6, 
                            validate="key", validatecommand=vcmd_int)
            entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # 5. Y-Axis Range (Zooming)
        ttk.Label(frame, text="5. Y-Axis Range", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
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
        
        # 6. Plot Size
        ttk.Label(frame, text="6. Plot Size", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
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
        
        # 7. Plot Title Settings
        ttk.Label(frame, text="7. Plot Title Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        title_frame = ttk.Frame(frame)
        title_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(title_frame, text="Main Title:").pack(side=tk.LEFT, padx=(0, 5))
        self.title_entry = ttk.Entry(title_frame)
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        subtitle_frame = ttk.Frame(frame)
        subtitle_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(subtitle_frame, text="Subtitle:").pack(side=tk.LEFT, padx=(0, 5))
        self.subtitle_entry = ttk.Entry(subtitle_frame)
        self.subtitle_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        y_axis_frame = ttk.Frame(frame)
        y_axis_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(y_axis_frame, text="Y-Axis Title:").pack(side=tk.LEFT, padx=(0, 5))
        self.y_axis_title_entry = ttk.Entry(y_axis_frame)
        self.y_axis_title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Show titles checkbox
        ttk.Checkbutton(frame, text="Show plot titles", variable=self.show_titles_var).pack(anchor="w", pady=(0, 10))
        
        # 8. Box Plot Settings
        ttk.Label(frame, text="8. Box Plot Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Checkbutton(frame, text="Show Outliers", variable=self.show_outliers_var).pack(anchor="w", pady=(0, 10))
        
        # 9. Legend Settings
        ttk.Label(frame, text="9. Legend Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Legend position
        legend_position_frame = ttk.Frame(frame)
        legend_position_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(legend_position_frame, text="Legend Position:").pack(side=tk.LEFT, padx=(0, 5))
        legend_position_combo = ttk.Combobox(legend_position_frame, textvariable=self.legend_position_var, 
                                           values=list(self._get_legend_positions().keys()), state="readonly", width=20)
        legend_position_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Advanced legend settings
        legend_settings_frame = ttk.LabelFrame(frame, text="Advanced Legend Settings", padding=5)
        legend_settings_frame.pack(fill=tk.X, pady=(5, 10))
        
        # Number of columns
        ncol_frame = ttk.Frame(legend_settings_frame)
        ncol_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ncol_frame, text="Columns:", width=12).pack(side=tk.LEFT)
        legend_ncol_combo = ttk.Combobox(ncol_frame, textvariable=self.legend_ncol_var, 
                                       values=["auto", "1", "2", "3", "4", "5"], state="readonly", width=8)
        legend_ncol_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Frame settings
        legend_frame_frame = ttk.Frame(legend_settings_frame)
        legend_frame_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(legend_frame_frame, text="Show Legend Frame", variable=self.legend_frameon_var).pack(side=tk.LEFT)
        
        # Transparency
        alpha_frame = ttk.Frame(legend_settings_frame)
        alpha_frame.pack(fill=tk.X, pady=2)
        ttk.Label(alpha_frame, text="Transparency:", width=12).pack(side=tk.LEFT)
        alpha_combo = ttk.Combobox(alpha_frame, textvariable=self.legend_alpha_var, 
                                 values=["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], state="readonly", width=8)
        alpha_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # 10. Customize Labels (Optional)
        ttk.Label(frame, text="10. Customize Labels (Optional)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        ttk.Label(frame, text="Metric Display Names:").pack(anchor="w")
        self.metric_rename_frame = ttk.Frame(frame)  # Dynamic entries here
        self.metric_rename_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(frame, text="Model/Experiment Display Names:").pack(anchor="w")
        self.model_rename_frame = ttk.Frame(frame)  # Dynamic entries here
        self.model_rename_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 11. Output Format
        ttk.Label(frame, text="11. Output Format", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        format_frame = ttk.Frame(frame)
        format_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Radiobutton(format_frame, text="Plot", variable=self.output_format_var, 
                      value="Plot", command=self._toggle_output_format).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(format_frame, text="LaTeX Table", variable=self.output_format_var, 
                      value="LaTeX", command=self._toggle_output_format).pack(side=tk.LEFT)
        
        # 12. LaTeX Table Settings (initially hidden)
        self.latex_settings_frame = ttk.LabelFrame(frame, text="LaTeX Table Settings")
        
        latex_inner_frame = ttk.Frame(self.latex_settings_frame, padding=5)
        latex_inner_frame.pack(fill=tk.X, expand=True)
        
        ttk.Label(latex_inner_frame, text="Table Type:").pack(anchor="w", pady=(0, 5))
        latex_type_combo = ttk.Combobox(latex_inner_frame, textvariable=self.latex_table_type_var, 
                                     values=self.LATEX_TABLE_TYPES, state="readonly")
        latex_type_combo.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(latex_inner_frame, text="Table Caption:").pack(anchor="w", pady=(0, 5))
        self.latex_caption_entry = ttk.Entry(latex_inner_frame)
        self.latex_caption_entry.insert(0, "Experimental Results")
        self.latex_caption_entry.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(latex_inner_frame, text="Table Label:").pack(anchor="w", pady=(0, 5))
        self.latex_label_entry = ttk.Entry(latex_inner_frame)
        self.latex_label_entry.insert(0, "tab:results")
        self.latex_label_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Initially hide the LaTeX settings since Plot is the default
        # Will be shown/hidden by _toggle_output_format
        
        # 13. Generate Output
        self.generate_button = ttk.Button(frame, text="Generate Plot", command=self._generate_plot)
        self.generate_button.pack(fill=tk.X, pady=(10, 0))
    
    def _validate_int(self, value):
        """Validate that input is a positive integer."""
        if value == "":
            return True
        try:
            int_val = int(value)
            return int_val > 0
        except ValueError:
            return False
    
    def _on_font_preset_change(self, event=None):
        """Handle font preset selection change."""
        preset_name = self.font_preset_var.get()
        if preset_name in FONT_SIZE_PRESETS:
            preset_sizes = FONT_SIZE_PRESETS[preset_name]
            for key, size in preset_sizes.items():
                if key in self.font_size_vars:
                    self.font_size_vars[key].set(str(size))

    def _browse_files(self):
        """Browse and add files to the list."""
        files = self.file_controller.browse_files()
        if files:
            self._update_file_listbox()
    
    def _clear_files(self):
        """Clear all files from the list."""
        self.file_controller.clear_files()
        self.file_listbox.delete(0, tk.END)
    
    def _remove_selected_files(self):
        """Remove selected files from the list."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Information", "No files selected to remove.")
            return
        
        # Get the full paths of selected files
        selected_files = []
        for idx in selected_indices:
            filename = self.file_listbox.get(idx)
            # Find the corresponding full path
            for full_path in self.file_controller.selected_files:
                if os.path.basename(full_path) == filename:
                    selected_files.append(full_path)
                    break
        
        # Remove each selected file
        for file_path in selected_files:
            self.file_controller.remove_file(file_path)
        
        # Update the listbox
        self._update_file_listbox()
    
    def _update_file_listbox(self):
        """Update the file listbox with current selected files."""
        self.file_listbox.delete(0, tk.END)
        for filename in self.file_controller.get_selected_filenames():
            self.file_listbox.insert(tk.END, filename)
    
    def _load_and_process_files(self):
        """Load and process selected files."""
        if self.file_controller.load_and_process_files():
            self._update_ui_after_load()
    
    def _save_config(self):
        """Save current configuration to a file."""
        try:
            self.file_controller.save_config(self)
        except Exception as e:
            messagebox.showerror("Save Configuration Error", f"Failed to save configuration:\n{str(e)}")
    
    def _load_config(self):
        """Load configuration from a file."""
        try:
            self.file_controller.load_config(self)
        except Exception as e:
            messagebox.showerror("Load Configuration Error", f"Failed to load configuration:\n{str(e)}")
    
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
                
                # Use translation if available, otherwise use original name
                display_name = self.METRIC_TRANSLATIONS.get(metric, metric)
                entry.insert(0, display_name)
                
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
    
    def _toggle_output_format(self):
        """Toggle between plot and LaTeX table output formats."""
        if self.output_format_var.get() == "Plot":
            # Hide LaTeX settings, show plot-specific settings
            if self.latex_settings_frame.winfo_ismapped():
                self.latex_settings_frame.pack_forget()
            
            # Update button text
            self.generate_button.config(text="Generate Plot")
        else:
            # Show LaTeX settings, hide plot-specific settings
            if not self.latex_settings_frame.winfo_ismapped():
                self.latex_settings_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Update button text
            self.generate_button.config(text="Generate LaTeX Table")
    
    def _get_font_sizes(self):
        """Get current font sizes from UI controls."""
        font_sizes = {}
        for key, var in self.font_size_vars.items():
            try:
                font_sizes[key] = int(var.get())
            except ValueError:
                font_sizes[key] = DEFAULT_FONT_SIZES[key]  # Use default if invalid
        return font_sizes
    
    def _generate_plot(self):
        """Generate plot or LaTeX table based on current settings."""
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
        
        # Apply automatic translations (user-defined names take precedence)
        for metric in selected_metrics:
            if metric in self.METRIC_TRANSLATIONS and metric not in display_metrics:
                display_metrics[metric] = self.METRIC_TRANSLATIONS[metric]
        
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
        
        # Create base parameters (common to both plot and LaTeX)
        base_params = {
            'selected_metrics': selected_metrics,
            'display_metrics': display_metrics,
            'display_experiments': display_experiments,
            'font_sizes': self._get_font_sizes(),  # Add font sizes to parameters
        }
        
        # Check output format
        if self.output_format_var.get() == "Plot":
            # Create plot-specific parameters
            plot_params = base_params.copy()
            plot_params.update({
                'plot_type': self.plot_type_var.get(),
                'title': self.title_entry.get().strip(),
                'subtitle': self.subtitle_entry.get().strip(),
                'y_axis_title': self.y_axis_title_entry.get().strip(),
                'use_custom_y_range': self.custom_y_range_var.get(),
                'y_min': self.y_min_entry.get(),
                'y_max': self.y_max_entry.get(),
                'width': plot_width,
                'height': plot_height,
                'show_outliers': self.show_outliers_var.get(),
                'show_titles': self.show_titles_var.get(),
                'legend_position': self.legend_position_var.get(),
                'legend_ncol': self.legend_ncol_var.get(),
                'legend_frameon': self.legend_frameon_var.get(),
                'legend_alpha': self.legend_alpha_var.get(),
            })
            
            # Call the callback
            self.on_generate_plot(plot_params)
        else:
            # LaTeX table generation
            latex_params = base_params.copy()
            latex_params.update({
                'table_type': self.latex_table_type_var.get(),
                'caption': self.latex_caption_entry.get().strip(),
                'label': self.latex_label_entry.get().strip(),
            })
            
            # Call the LaTeX generation method
            self.on_generate_latex(latex_params)
    
    def on_generate_plot(self, plot_params):
        """Callback method to be overridden by the parent."""
        pass
    
    def on_generate_latex(self, latex_params):
        """Callback method to be overridden by the parent."""
        pass

    def _get_legend_positions(self):
        """Get available legend positions with descriptions."""
        return {
            "best": "Best (automatic)",
            "upper right": "Upper Right",
            "upper left": "Upper Left", 
            "lower left": "Lower Left",
            "lower right": "Lower Right",
            "right": "Right (outside)",
            "center left": "Center Left",
            "center right": "Center Right",
            "lower center": "Lower Center",
            "upper center": "Upper Center",
            "center": "Center",
            "bottom": "Bottom (outside)",
            "top": "Top (outside)",
            "left": "Left (outside)",
            "none": "None (hide legend)"
        }
    
    def get_legend_settings(self):
        """Get the current legend settings."""
        ncol = self.legend_ncol_var.get()
        if ncol == "auto":
            ncol = None
        else:
            ncol = int(ncol)
            
        return {
            'position': self.legend_position_var.get(),
            'ncol': ncol,
            'frameon': self.legend_frameon_var.get(),
            'alpha': float(self.legend_alpha_var.get())
        } 