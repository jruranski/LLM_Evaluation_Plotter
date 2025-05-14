import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Explicitly set backend for Tkinter compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import re
from collections import defaultdict
import time # For simple loading simulation if needed, though UI updates are better

# --- Scientific Plotting Defaults ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('ggplot')
    except OSError:
        print("Warning: Neither 'seaborn-v0_8-whitegrid' nor 'ggplot' styles found. Using Matplotlib default.")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"], # Fallback fonts
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 100, # Default for display
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "axes.formatter.use_mathtext": True, # For math symbols
    "figure.constrained_layout.use": True, # Better layout management
    "figure.figsize": [10, 6],  # Default figure size
    "figure.autolayout": False,  # Let us control the layout manually
    "axes.titlelocation": "center",
    "axes.titlepad": 12,
    "axes.labelpad": 8,
})

class RagEvaluationPlotter:
    def __init__(self, master):
        self.master = master
        master.title("RAG Evaluation Plotter")
        master.geometry("1300x850")

        # --- Data Storage ---
        self.selected_files = []
        self.experiment_data = {} # Stores {exp_name: {mean: pd.Series, std: pd.Series, all_values: {metric: [vals]}}}
        self.available_metrics = [] # Numeric columns for plotting
        self.current_plot_fig = None # Store the current Matplotlib figure

        # --- UI Structure ---
        # Main PanedWindow for resizable sections
        paned_window = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control Panel (Left)
        control_panel_outer = ttk.Frame(paned_window, width=400) # Initial width
        paned_window.add(control_panel_outer, weight=1) # weight determines resize proportion

        # Canvas and Scrollbar for Control Panel
        self.control_canvas = tk.Canvas(control_panel_outer, borderwidth=0, background="#ffffff")
        self.control_frame = ttk.Frame(self.control_canvas, padding="10") # Widgets go here
        self.scrollbar = ttk.Scrollbar(control_panel_outer, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_canvas_window = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        self.control_frame.bind("<Configure>", self._on_control_frame_configure)
        self.control_canvas.bind('<Enter>', self._bind_mousewheel)
        self.control_canvas.bind('<Leave>', self._unbind_mousewheel)


        # Plot Area (Right)
        self.plot_area_frame = ttk.Frame(paned_window)
        paned_window.add(self.plot_area_frame, weight=3)

        self._create_control_widgets()
        self._create_plot_area_widgets()

    def _on_control_frame_configure(self, event=None):
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def _bind_mousewheel(self, event):
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel) # For Windows/Linux
        self.control_canvas.bind_all("<Button-4>", self._on_mousewheel) # For Linux scroll up
        self.control_canvas.bind_all("<Button-5>", self._on_mousewheel) # For Linux scroll down

    def _unbind_mousewheel(self, event):
        self.control_canvas.unbind_all("<MouseWheel>")
        self.control_canvas.unbind_all("<Button-4>")
        self.control_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0: # Scroll up
            self.control_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0: # Scroll down
            self.control_canvas.yview_scroll(1, "units")

    def _create_control_widgets(self):
        frame = self.control_frame # Add widgets to this frame

        # 1. File Selection
        ttk.Label(frame, text="1. Select Evaluation Files", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        self.file_listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, height=7, width=45)
        self.file_listbox.pack(fill=tk.X, expand=True, pady=(0,5))
        browse_button = ttk.Button(frame, text="Browse Files", command=self._browse_files)
        browse_button.pack(fill=tk.X, pady=(0,5))
        load_button = ttk.Button(frame, text="Load & Process Selected Files", command=self._load_and_process_files)
        load_button.pack(fill=tk.X, pady=(0,10))

        # 2. Metric Selection
        ttk.Label(frame, text="2. Select Metrics to Plot (Y-axis)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        self.metric_selection_frame = ttk.Frame(frame) # Dynamic checkbuttons here
        self.metric_selection_frame.pack(fill=tk.X, pady=(0,10))
        self.metric_vars = {}

        # 3. Plot Type
        ttk.Label(frame, text="3. Select Plot Type", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        self.plot_type_var = tk.StringVar(value="Bar Plot")
        plot_types = ["Bar Plot", "Grouped Bar Plot", "Line Plot", "Box Plot", "Per Test Case", "Per Test Case Line Plot", "Heatmap Per Model", 
                     "Correlation Matrix", "Radar Chart", "Violin Plot", "Statistical Significance"]
        plot_type_combo = ttk.Combobox(frame, textvariable=self.plot_type_var, values=plot_types, state="readonly")
        plot_type_combo.pack(fill=tk.X, pady=(0,10))

        # 4. Y-Axis Range (Zooming)
        ttk.Label(frame, text="4. Y-Axis Range", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        
        self.custom_y_range_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Use Custom Y-Axis Range", variable=self.custom_y_range_var).pack(anchor="w")
        
        y_range_frame = ttk.Frame(frame)
        y_range_frame.pack(fill=tk.X, pady=(5,10))
        
        ttk.Label(y_range_frame, text="Min:").pack(side=tk.LEFT, padx=(0,5))
        vcmd = (self.master.register(self._validate_float), '%P')
        self.y_min_entry = ttk.Entry(y_range_frame, width=8, validate="key", validatecommand=vcmd)
        self.y_min_entry.pack(side=tk.LEFT, padx=(0,10))
        
        ttk.Label(y_range_frame, text="Max:").pack(side=tk.LEFT, padx=(0,5))
        self.y_max_entry = ttk.Entry(y_range_frame, width=8, validate="key", validatecommand=vcmd)
        self.y_max_entry.pack(side=tk.LEFT)
        
        # 5. Plot Size
        ttk.Label(frame, text="5. Plot Size", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        
        size_frame = ttk.Frame(frame)
        size_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(size_frame, text="Width (inches):").pack(side=tk.LEFT, padx=(0,5))
        self.plot_width_entry = ttk.Entry(size_frame, width=6, validate="key", validatecommand=vcmd)
        self.plot_width_entry.insert(0, "10")  # Default width
        self.plot_width_entry.pack(side=tk.LEFT, padx=(0,10))
        
        ttk.Label(size_frame, text="Height (inches):").pack(side=tk.LEFT, padx=(0,5))
        self.plot_height_entry = ttk.Entry(size_frame, width=6, validate="key", validatecommand=vcmd)
        self.plot_height_entry.insert(0, "6")  # Default height
        self.plot_height_entry.pack(side=tk.LEFT)
        
        # Size presets - use a scrollable frame to prevent overflow
        preset_label = ttk.Label(frame, text="Size Presets:")
        preset_label.pack(anchor="w", pady=(0,5))
        
        presets_container = ttk.Frame(frame)
        presets_container.pack(fill=tk.X, pady=(0,10))
        
        size_presets = {
            "Standard (10×6)": (10, 6),
            "Wide (16×9)": (16, 9),
            "Square (8×8)": (8, 8),
            "Poster (12×15)": (12, 15),
            "Small (6×4)": (6, 4)
        }
        
        # Create two rows for presets to avoid horizontal overflow
        row1 = ttk.Frame(presets_container)
        row1.pack(fill=tk.X, pady=(0,5))
        row2 = ttk.Frame(presets_container)
        row2.pack(fill=tk.X)
        
        preset_items = list(size_presets.items())
        for i, (preset_name, (w, h)) in enumerate(preset_items[:3]):
            preset_btn = ttk.Button(row1, text=preset_name, 
                                  command=lambda w=w, h=h: self._set_size_preset(w, h))
            preset_btn.pack(side=tk.LEFT, padx=(0,5), fill=tk.X, expand=True)
            
        for i, (preset_name, (w, h)) in enumerate(preset_items[3:]):
            preset_btn = ttk.Button(row2, text=preset_name, 
                                  command=lambda w=w, h=h: self._set_size_preset(w, h))
            preset_btn.pack(side=tk.LEFT, padx=(0,5), fill=tk.X, expand=True)

        # 6. Plot Title Settings
        ttk.Label(frame, text="6. Plot Title Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        
        title_frame = ttk.Frame(frame)
        title_frame.pack(fill=tk.X, pady=(0,5))
        ttk.Label(title_frame, text="Main Title:").pack(side=tk.LEFT, padx=(0,5))
        self.title_entry = ttk.Entry(title_frame)
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        subtitle_frame = ttk.Frame(frame)
        subtitle_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Label(subtitle_frame, text="Subtitle:").pack(side=tk.LEFT, padx=(0,5))
        self.subtitle_entry = ttk.Entry(subtitle_frame)
        self.subtitle_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 7. Box Plot Settings
        ttk.Label(frame, text="7. Box Plot Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        self.show_outliers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Show Outliers", variable=self.show_outliers_var).pack(anchor="w", pady=(0,10))

        # 8. Customize Labels (Optional)
        ttk.Label(frame, text="8. Customize Labels (Optional)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        
        ttk.Label(frame, text="Metric Display Names:").pack(anchor="w")
        self.metric_rename_frame = ttk.Frame(frame) # Dynamic entries here
        self.metric_rename_frame.pack(fill=tk.X, pady=(0,5))
        self.metric_rename_entries = {}

        ttk.Label(frame, text="Model/Experiment Display Names:").pack(anchor="w")
        self.model_rename_frame = ttk.Frame(frame) # Dynamic entries here
        self.model_rename_frame.pack(fill=tk.X, pady=(0,10))
        self.model_rename_entries = {}

        # 9. Generate Plot
        generate_button = ttk.Button(frame, text="Generate Plot", command=self._generate_plot_with_indicator)
        generate_button.pack(fill=tk.X, pady=(10,0))

    def _create_plot_area_widgets(self):
        # Main container for plot area with proper spacing
        plot_container = ttk.Frame(self.plot_area_frame)
        plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas placeholder for plots
        self.plot_canvas_frame = ttk.Frame(plot_container)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        
        # Button container with fixed height
        button_frame = ttk.Frame(plot_container, height=30)
        button_frame.pack(fill=tk.X, pady=(5,0))
        button_frame.pack_propagate(False)  # Maintain fixed height
        
        save_button = ttk.Button(button_frame, text="Save Plot", command=self._save_plot)
        save_button.pack(side=tk.RIGHT)
        
        # No initial canvas - we'll create it when needed
        self.plot_canvas = None

    def _browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select CSV Evaluation Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if files:
            self.selected_files = list(files) # Replace, don't append
            self.file_listbox.delete(0, tk.END)
            for f_path in self.selected_files:
                self.file_listbox.insert(tk.END, os.path.basename(f_path))

    def _get_experiment_base_name(self, file_path):
        filename = os.path.basename(file_path)
        # Remove _run_N.csv suffix
        base_name = re.sub(r'_run_\d+\.csv$', '', filename, flags=re.IGNORECASE)
        # Remove .csv if it's not a run file (e.g., experiment_name.csv)
        base_name = base_name.replace('.csv', '')
        return base_name

    def _load_and_process_files(self):
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected.")
            return

        # Show loading indicator
        loading_win = self._show_loading_window("Processing files...")
        self.master.update_idletasks()

        grouped_files = defaultdict(list)
        for f_path in self.selected_files:
            base_name = self._get_experiment_base_name(f_path)
            grouped_files[base_name].append(f_path)

        self.experiment_data.clear()
        self.available_metrics.clear()
        all_numeric_cols = set()
        
        # Store per-test case data for the "Per Test Case" plot
        self.test_case_data = {}

        try:
            for exp_name, file_paths in grouped_files.items():
                run_summaries = [] # List of pd.Series, one for each run's summary
                test_case_values = {} # Dict to store values for each test case and metric
                
                for f_path in file_paths:
                    try:
                        df_run = pd.read_csv(f_path)
                        if df_run.empty:
                            print(f"Warning: File {f_path} is empty. Skipping.")
                            continue

                        numeric_cols = df_run.select_dtypes(include=np.number).columns
                        if not numeric_cols.empty:
                            # If multiple rows, average numeric metrics for this run
                            summary_series = df_run[numeric_cols].mean()
                            run_summaries.append(summary_series)
                            all_numeric_cols.update(summary_series.index)
                            
                            # Process per-test case data if "Test Case" column exists
                            if "Test Case" in df_run.columns and any(col.endswith("Score") for col in df_run.columns):
                                # Initialize structure for test case data if needed
                                if exp_name not in self.test_case_data:
                                    self.test_case_data[exp_name] = {}
                                
                                # Extract score columns
                                score_cols = [col for col in df_run.columns if col.endswith("Score") and df_run[col].dtype in [np.float64, np.int64]]
                                
                                for score_col in score_cols:
                                    if score_col not in self.test_case_data[exp_name]:
                                        self.test_case_data[exp_name][score_col] = {}
                                        
                                    for _, row in df_run.iterrows():
                                        if pd.notna(row["Test Case"]) and pd.notna(row[score_col]):
                                            test_case = int(row["Test Case"]) if isinstance(row["Test Case"], (int, float)) else str(row["Test Case"])
                                            
                                            if test_case not in self.test_case_data[exp_name][score_col]:
                                                self.test_case_data[exp_name][score_col][test_case] = []
                                                
                                            self.test_case_data[exp_name][score_col][test_case].append(float(row[score_col]))
                    except Exception as e:
                        print(f"Error reading or processing file {f_path}: {e}")
                        messagebox.showwarning("File Error", f"Could not process {os.path.basename(f_path)}:\n{e}")
                
                if not run_summaries:
                    print(f"Warning: No valid data found for experiment {exp_name}. Skipping.")
                    continue

                # Combine summaries from all runs of this experiment
                exp_df = pd.DataFrame(run_summaries)
                
                mean_stats = exp_df.mean()
                std_stats = exp_df.std()
                
                all_values_for_boxplot = {}
                for col in exp_df.columns:
                    all_values_for_boxplot[col] = exp_df[col].dropna().tolist()

                self.experiment_data[exp_name] = {
                    'mean': mean_stats,
                    'std': std_stats,
                    'all_values': all_values_for_boxplot
                }
            
            self.available_metrics = sorted(list(all_numeric_cols))
            self._update_ui_after_load()
            messagebox.showinfo("Success", f"Processed {len(self.experiment_data)} experiments.")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during data processing: {e}")
            print(f"Processing error: {e}")
        finally:
            if loading_win: loading_win.destroy()
            self._on_control_frame_configure() # Recalculate scroll region


    def _update_ui_after_load(self):
        # Clear old metric selectors and rename entries
        for widget in self.metric_selection_frame.winfo_children(): widget.destroy()
        self.metric_vars.clear()
        for widget in self.metric_rename_frame.winfo_children(): widget.destroy()
        self.metric_rename_entries.clear()
        for widget in self.model_rename_frame.winfo_children(): widget.destroy()
        self.model_rename_entries.clear()

        # Populate metric selectors and rename fields
        if not self.available_metrics:
            ttk.Label(self.metric_selection_frame, text="No numeric metrics found.").pack()
        else:
            for metric in self.available_metrics:
                var = tk.BooleanVar(value=False)
                cb = ttk.Checkbutton(self.metric_selection_frame, text=metric, variable=var)
                cb.pack(anchor="w")
                self.metric_vars[metric] = var

                # Metric rename field
                row_frame = ttk.Frame(self.metric_rename_frame)
                row_frame.pack(fill=tk.X)
                ttk.Label(row_frame, text=f"{metric[:25]}:", width=28, anchor='w').pack(side=tk.LEFT) # Truncate long original names
                entry = ttk.Entry(row_frame, width=25)
                entry.insert(0, metric) # Default to original name
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.metric_rename_entries[metric] = entry
        
        # Populate model rename fields
        for exp_name in self.experiment_data.keys():
            row_frame = ttk.Frame(self.model_rename_frame)
            row_frame.pack(fill=tk.X)
            ttk.Label(row_frame, text=f"{exp_name[:25]}:", width=28, anchor='w').pack(side=tk.LEFT)
            entry = ttk.Entry(row_frame, width=25)
            entry.insert(0, exp_name)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.model_rename_entries[exp_name] = entry
        
        self.control_frame.update_idletasks()
        self._on_control_frame_configure()

    def _validate_float(self, value):
        """Validate if input string is a valid float or empty string"""
        if value == "" or value == "-":
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _set_size_preset(self, width, height):
        """Set the plot size entries to a preset value"""
        self.plot_width_entry.delete(0, tk.END)
        self.plot_width_entry.insert(0, str(width))
        self.plot_height_entry.delete(0, tk.END)
        self.plot_height_entry.insert(0, str(height))

    def _get_display_name(self, original_name, rename_map_entries):
        entry = rename_map_entries.get(original_name)
        if entry:
            name = entry.get().strip()
            return name if name else original_name
        return original_name

    def _generate_plot_with_indicator(self):
        if not self.experiment_data:
            messagebox.showerror("Error", "No data loaded. Please load and process files first.")
            return
        
        selected_orig_metrics = [m for m, v in self.metric_vars.items() if v.get()]
        if not selected_orig_metrics:
            messagebox.showerror("Error", "Please select at least one metric to plot.")
            return
        
        plot_type = self.plot_type_var.get()
        if plot_type in ["Box Plot", "Per Test Case"] and len(selected_orig_metrics) > 1:
            messagebox.showwarning(f"{plot_type}", f"For {plot_type}, typically one metric is selected for clarity. Plotting the first selected metric.")
            selected_orig_metrics = [selected_orig_metrics[0]]
        
        loading_win = self._show_loading_window("Generating plot...")
        self.master.update_idletasks()

        try:
            # Clean up any existing plot to avoid memory leaks
            self._clear_current_plot()
            
            # Get plot size from UI
            try:
                plot_width = float(self.plot_width_entry.get() or "10")
                plot_height = float(self.plot_height_entry.get() or "6")
                # Enforce minimum size to prevent tiny plots
                plot_width = max(4, plot_width)
                plot_height = max(3, plot_height)
            except ValueError:
                plot_width, plot_height = 10, 6  # Default if invalid input
            
            # Prepare data for plotting
            plot_exp_names = list(self.experiment_data.keys())
            display_exp_names = [self._get_display_name(name, self.model_rename_entries) for name in plot_exp_names]
            display_metric_names = [self._get_display_name(name, self.metric_rename_entries) for name in selected_orig_metrics]

            # Get custom title and subtitle
            custom_title = self.title_entry.get().strip()
            custom_subtitle = self.subtitle_entry.get().strip()
            
            # Check if the plot type uses colorbars to determine the appropriate layout engine
            colorbar_plots = ["Heatmap Per Model", "Correlation Matrix", "Violin Plot", "Radar Chart", "Statistical Significance"]
            use_constrained_layout = plot_type not in colorbar_plots
            
            # Create new figure with specified size and appropriate layout settings
            self.current_plot_fig = plt.figure(figsize=(plot_width, plot_height), 
                                              constrained_layout=use_constrained_layout)
            
            # Only set manual adjustments if not using constrained layout
            if not use_constrained_layout:
                self.current_plot_fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
            
            # Only create ax for non-heatmap plots, as heatmap will create its own subplots
            if plot_type != "Heatmap Per Model":
                ax = self.current_plot_fig.add_subplot(111)
            else:
                ax = None

            num_exps = len(plot_exp_names)
            num_metrics = len(selected_orig_metrics)

            # Colors
            color_palette_name = "colorblind" if num_metrics > 1 or plot_type in ["Box Plot", "Per Test Case", "Per Test Case Line Plot"] else "viridis" 
            try:
                colors = sns.color_palette(color_palette_name, max(num_exps, num_metrics, 3))
            except:
                colors = sns.color_palette("deep", max(num_exps, num_metrics, 3))

            if plot_type == "Bar Plot" and num_metrics == 1:
                metric = selected_orig_metrics[0]
                display_metric = display_metric_names[0]
                means = [self.experiment_data[exp]['mean'].get(metric, np.nan) for exp in plot_exp_names]
                stds = [self.experiment_data[exp]['std'].get(metric, np.nan) for exp in plot_exp_names]
                
                bars = ax.bar(display_exp_names, means, yerr=stds, capsize=5, color=colors[0], alpha=0.8)
                ax.set_ylabel(display_metric)
                
                # Use custom title if provided, otherwise use default
                title = custom_title if custom_title else f"Comparison for {display_metric}"
                ax.set_title(title, fontsize=14, pad=15)
                
                # Add subtitle if provided
                if custom_subtitle:
                    ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                           fontsize=11, ha='center', va='top', style='italic')
                
                # Add text labels on bars
                for i, bar_obj in enumerate(bars):
                    yval = bar_obj.get_height()
                    std_val = stds[i]
                    text = f"{yval:.3f}"
                    if pd.notna(std_val) and std_val > 1e-6: # Don't show for NaN or very small std
                        text += f"\n(±{std_val:.3f})"
                    
                    # Position text slightly above the bar or error bar
                    text_y_pos = yval
                    if pd.notna(std_val) and yval >= 0 : text_y_pos += std_val
                    elif pd.notna(std_val) and yval < 0 : text_y_pos -= std_val # Below for negative bars
                    
                    # Small offset from the top of the bar/error bar
                    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
                    text_y_pos += offset if yval >=0 else -offset

                    ax.text(bar_obj.get_x() + bar_obj.get_width()/2.0, text_y_pos, text,
                            ha='center', va='bottom' if yval >=0 else 'top', fontsize=8, linespacing=0.9)


            elif plot_type == "Grouped Bar Plot" or (plot_type == "Bar Plot" and num_metrics > 1):
                # Construct DataFrame for grouped bar plot
                data_for_plot = {}
                for i, metric in enumerate(selected_orig_metrics):
                    data_for_plot[display_metric_names[i]] = [self.experiment_data[exp]['mean'].get(metric, np.nan) for exp in plot_exp_names]
                plot_df = pd.DataFrame(data_for_plot, index=display_exp_names)

                # Prepare yerr data if std is available
                yerr_data = None
                if all(self.experiment_data[exp]['std'] is not None for exp in plot_exp_names):
                    yerr_values = {}
                    for i, metric in enumerate(selected_orig_metrics):
                         yerr_values[display_metric_names[i]] = [self.experiment_data[exp]['std'].get(metric, 0) for exp in plot_exp_names] # fill 0 for missing std
                    yerr_data = pd.DataFrame(yerr_values, index=display_exp_names)
                
                plot_df.plot(kind='bar', ax=ax, yerr=yerr_data, capsize=4, color=colors[:num_metrics], alpha=0.8, width=0.8)
                ax.set_ylabel("Metric Value")
                ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Use custom title if provided, otherwise use default
                title = custom_title if custom_title else "Comparison Across Metrics"
                ax.set_title(title, fontsize=14, pad=15)
                
                # Add subtitle if provided
                if custom_subtitle:
                    ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                           fontsize=11, ha='center', va='top', style='italic')

            elif plot_type == "Line Plot":
                for i, metric in enumerate(selected_orig_metrics):
                    display_metric = display_metric_names[i]
                    means = [self.experiment_data[exp]['mean'].get(metric, np.nan) for exp in plot_exp_names]
                    stds = [self.experiment_data[exp]['std'].get(metric, np.nan) for exp in plot_exp_names]
                    ax.errorbar(display_exp_names, means, yerr=stds, label=display_metric, marker='o', capsize=5, color=colors[i])
                ax.set_ylabel("Metric Value")
                if num_metrics > 1:
                    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Use custom title if provided, otherwise use default
                title = custom_title if custom_title else "Metric Trends"
                ax.set_title(title, fontsize=14, pad=15)
                
                # Add subtitle if provided
                if custom_subtitle:
                    ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                           fontsize=11, ha='center', va='top', style='italic')

            elif plot_type == "Box Plot": # Assumes one metric was selected
                metric = selected_orig_metrics[0]
                display_metric = display_metric_names[0]
                data_to_plot = [self.experiment_data[exp]['all_values'].get(metric, []) for exp in plot_exp_names]
                
                # Filter out experiments with no data for this metric
                valid_data_indices = [i for i, d in enumerate(data_to_plot) if d]
                filtered_data = [data_to_plot[i] for i in valid_data_indices]
                filtered_labels = [display_exp_names[i] for i in valid_data_indices]

                show_outliers = self.show_outliers_var.get()

                if filtered_data:
                    bp = ax.boxplot(filtered_data, labels=filtered_labels, patch_artist=True,
                                   showfliers=show_outliers,  # Show outliers based on user preference
                                   medianprops=dict(color="black", linewidth=1.5))
                    for i, patch in enumerate(bp['boxes']):
                        patch.set_facecolor(colors[i % len(colors)])
                        patch.set_alpha(0.7)
                    ax.set_ylabel(display_metric)
                    
                    # Use custom title if provided, otherwise use default
                    title = custom_title if custom_title else f"Distribution of {display_metric}"
                    ax.set_title(title, fontsize=14, pad=15)
                    
                    # Add subtitle if provided
                    if custom_subtitle:
                        ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                               fontsize=11, ha='center', va='top', style='italic')
                    
                    # Add legend for outliers if they're shown
                    if show_outliers and bp.get('fliers', []):
                        outlier_label = plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor='black', markersize=6, label='Outliers')
                        ax.legend(handles=[outlier_label], loc='upper right')
                        
                else:
                    ax.text(0.5, 0.5, f"No data available for {display_metric} to create box plot.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                            
            elif plot_type == "Per Test Case": # New plot type
                metric = selected_orig_metrics[0]
                display_metric = display_metric_names[0]
                
                if not hasattr(self, 'test_case_data') or not self.test_case_data:
                    ax.text(0.5, 0.5, "No per-test case data available. Please ensure your files have 'Test Case' column and metric scores.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                else:
                    # Check if any experiment has data for this metric
                    has_data = False
                    for exp_name in plot_exp_names:
                        if exp_name in self.test_case_data and metric in self.test_case_data[exp_name]:
                            has_data = True
                            break
                    
                    if not has_data:
                        ax.text(0.5, 0.5, f"No test case data available for metric '{display_metric}'.",
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else:
                        # Find all unique test cases across all experiments
                        all_test_cases = set()
                        for exp_name in plot_exp_names:
                            if exp_name in self.test_case_data and metric in self.test_case_data[exp_name]:
                                all_test_cases.update(self.test_case_data[exp_name][metric].keys())
                        
                        # Sort test cases (numeric if possible)
                        try:
                            test_cases = sorted([int(tc) for tc in all_test_cases if str(tc).isdigit()])
                            test_cases.extend(sorted([tc for tc in all_test_cases if not str(tc).isdigit()]))
                        except:
                            test_cases = sorted(all_test_cases)
                        
                        # Convert to strings for x-axis
                        test_case_labels = [str(tc) for tc in test_cases]
                        
                        # Calculate bar positions
                        bar_width = 0.8 / num_exps
                        offsets = np.arange(-(num_exps-1)/2, (num_exps-1)/2 + 0.1, 1) * bar_width
                        
                        # Plot bars for each experiment
                        for i, exp_name in enumerate(plot_exp_names):
                            display_exp_name = display_exp_names[i]
                            means = []
                            stds = []
                            
                            for tc in test_cases:
                                if (exp_name in self.test_case_data and 
                                    metric in self.test_case_data[exp_name] and 
                                    tc in self.test_case_data[exp_name][metric]):
                                    
                                    values = self.test_case_data[exp_name][metric][tc]
                                    means.append(np.mean(values) if values else np.nan)
                                    stds.append(np.std(values) if len(values) > 1 else 0)
                                else:
                                    means.append(np.nan)
                                    stds.append(0)
                            
                            # Calculate x positions for bars
                            x_positions = np.arange(len(test_case_labels)) + offsets[i]
                            
                            # Plot bars
                            bars = ax.bar(x_positions, means, width=bar_width, yerr=stds, 
                                          label=display_exp_name, color=colors[i], alpha=0.8, capsize=3)
                        
                        # Set x-axis labels
                        ax.set_xticks(np.arange(len(test_case_labels)))
                        ax.set_xticklabels(test_case_labels)
                        ax.set_xlabel("Test Case")
                        ax.set_ylabel(display_metric)
                        
                        # Add legend
                        if num_exps > 1:
                            ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
                        
                        # Use custom title if provided, otherwise use default
                        title = custom_title if custom_title else f"{display_metric} per Test Case"
                        ax.set_title(title, fontsize=14, pad=15)
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                                   fontsize=11, ha='center', va='top', style='italic')

            elif plot_type == "Per Test Case Line Plot": # New line plot for test cases
                if not hasattr(self, 'test_case_data') or not self.test_case_data:
                    ax.text(0.5, 0.5, "No per-test case data available. Please ensure your files have 'Test Case' column and metric scores.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                else:
                    # Find all unique test cases across all experiments and metrics
                    all_test_cases = set()
                    has_data = False
                    
                    for exp_name in plot_exp_names:
                        if exp_name in self.test_case_data:
                            for metric in selected_orig_metrics:
                                if metric in self.test_case_data[exp_name]:
                                    all_test_cases.update(self.test_case_data[exp_name][metric].keys())
                                    has_data = True
                    
                    if not has_data:
                        ax.text(0.5, 0.5, "No test case data available for the selected metrics.",
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else:
                        # Sort test cases (numeric if possible)
                        try:
                            test_cases = sorted([int(tc) for tc in all_test_cases if str(tc).isdigit()])
                            test_cases.extend(sorted([tc for tc in all_test_cases if not str(tc).isdigit()]))
                        except:
                            test_cases = sorted(all_test_cases)
                        
                        # Convert to strings for x-axis
                        test_case_labels = [str(tc) for tc in test_cases]
                        
                        # Define marker styles for different models
                        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
                        if num_exps > len(markers):
                            # Repeat markers if needed
                            markers = markers * (num_exps // len(markers) + 1)
                        
                        # Create line plots for each model-metric combination
                        for i, exp_name in enumerate(plot_exp_names):
                            display_exp_name = display_exp_names[i]
                            
                            for j, metric in enumerate(selected_orig_metrics):
                                display_metric = display_metric_names[j]
                                
                                if (exp_name in self.test_case_data and 
                                    metric in self.test_case_data[exp_name]):
                                    
                                    means = []
                                    stds = []
                                    
                                    for tc in test_cases:
                                        if tc in self.test_case_data[exp_name][metric]:
                                            values = self.test_case_data[exp_name][metric][tc]
                                            means.append(np.mean(values) if values else np.nan)
                                            stds.append(np.std(values) if len(values) > 1 else 0)
                                        else:
                                            means.append(np.nan)
                                            stds.append(0)
                                    
                                    # Get x positions
                                    x_positions = np.arange(len(test_case_labels))
                                    
                                    # Remove NaN values for plotting
                                    valid_indices = ~np.isnan(means)
                                    if not any(valid_indices):
                                        continue  # Skip if no valid data
                                    
                                    x_valid = x_positions[valid_indices]
                                    y_valid = np.array(means)[valid_indices]
                                    yerr_valid = np.array(stds)[valid_indices]
                                    
                                    # Create label that includes both model and metric
                                    if num_metrics > 1:
                                        label = f"{display_exp_name} - {display_metric}"
                                    else:
                                        label = display_exp_name
                                    
                                    # Plot with different marker for each model
                                    line = ax.errorbar(
                                        x_valid, y_valid, yerr=yerr_valid,
                                        label=label,
                                        marker=markers[i % len(markers)],
                                        markersize=8,
                                        color=colors[j % len(colors)] if num_metrics > 1 else colors[i % len(colors)],
                                        linestyle='-' if num_metrics == 1 else ['-', '--', ':', '-.'][j % 4],
                                        capsize=3
                                    )
                        
                        # Set x-axis labels
                        ax.set_xticks(np.arange(len(test_case_labels)))
                        ax.set_xticklabels(test_case_labels)
                        ax.set_xlabel("Test Case")
                        
                        if num_metrics == 1:
                            ax.set_ylabel(display_metric_names[0])
                        else:
                            ax.set_ylabel("Score Value")
                        
                        # Add legend (with more space if many items)
                        if (num_exps > 1 or num_metrics > 1):
                            legend_items = num_exps * (1 if num_metrics == 1 else num_metrics)
                            if legend_items > 6:
                                # Two-column legend for many items
                                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
                            else:
                                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        
                        # Use custom title if provided, otherwise use default
                        if num_metrics == 1:
                            title = custom_title if custom_title else f"{display_metric_names[0]} per Test Case"
                        else:
                            title = custom_title if custom_title else "Metrics per Test Case"
                            
                        ax.set_title(title, fontsize=14, pad=15)
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                                   fontsize=11, ha='center', va='top', style='italic')
                        
                        # Make sure all test cases are visible
                        if len(test_case_labels) > 10:
                            # For many test cases, show fewer labels
                            step = max(1, len(test_case_labels) // 10)
                            visible_ticks = np.arange(0, len(test_case_labels), step)
                            ax.set_xticks(visible_ticks)
                            ax.set_xticklabels([test_case_labels[i] for i in visible_ticks])

            elif plot_type == "Heatmap Per Model":
                if not hasattr(self, 'test_case_data') or not self.test_case_data:
                    # Create a single subplot for the error message
                    ax = self.current_plot_fig.add_subplot(111)
                    ax.text(0.5, 0.5, "No per-test case data available. Please ensure your files have 'Test Case' column and metric scores.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.axis('off')  # Hide axes
                else:
                    # Check if any experiment has data for the selected metrics
                    valid_models = []
                    for exp_name in plot_exp_names:
                        if exp_name in self.test_case_data:
                            has_metrics = False
                            for metric in selected_orig_metrics:
                                if metric in self.test_case_data[exp_name] and self.test_case_data[exp_name][metric]:
                                    has_metrics = True
                                    break
                            if has_metrics:
                                valid_models.append(exp_name)
                    
                    if not valid_models:
                        # Create a single subplot for the error message
                        ax = self.current_plot_fig.add_subplot(111)
                        ax.text(0.5, 0.5, "No test case data available for the selected metrics.",
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                        ax.axis('off')  # Hide axes
                    else:
                        # Find all unique test cases across all experiments and metrics
                        all_test_cases = set()
                        for exp_name in valid_models:
                            for metric in selected_orig_metrics:
                                if metric in self.test_case_data[exp_name]:
                                    all_test_cases.update(self.test_case_data[exp_name][metric].keys())
                        
                        # Sort test cases (numeric if possible)
                        try:
                            test_cases = sorted([int(tc) for tc in all_test_cases if str(tc).isdigit()])
                            test_cases.extend(sorted([tc for tc in all_test_cases if not str(tc).isdigit()]))
                        except:
                            test_cases = sorted(all_test_cases)
                        
                        # Convert to strings for axis labels
                        test_case_labels = [str(tc) for tc in test_cases]
                        
                        # Determine grid layout based on number of models
                        n_models = len(valid_models)
                        if n_models == 1:
                            n_rows, n_cols = 1, 1
                        elif n_models == 2:
                            n_rows, n_cols = 1, 2
                        elif n_models <= 4:
                            n_rows, n_cols = 2, 2
                        elif n_models <= 6:
                            n_rows, n_cols = 2, 3
                        else:
                            # For more models, use a more rectangular layout
                            n_cols = min(4, n_models)
                            n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
                        
                        # Create a tight layout to maximize heatmap size
                        self.current_plot_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
                        
                        # Create a standardized color scale across all heatmaps
                        vmin, vmax = 0, 1
                        for exp_name in valid_models:
                            for metric in selected_orig_metrics:
                                if metric in self.test_case_data[exp_name]:
                                    for tc in self.test_case_data[exp_name][metric]:
                                        values = self.test_case_data[exp_name][metric][tc]
                                        if values:
                                            mean_val = np.mean(values)
                                            if pd.notna(mean_val):
                                                vmin = min(vmin, mean_val)
                                                vmax = max(vmax, mean_val)
                        
                        # Add a bit of padding to the range
                        vmin = max(0, vmin - 0.05)
                        vmax = min(1, vmax + 0.05)
                        
                        # Use a sequential colormap
                        cmap = sns.color_palette("rocket", as_cmap=True)
                        
                        # Create heatmaps for each model
                        for i, exp_name in enumerate(valid_models):
                            display_exp_name = display_exp_names[plot_exp_names.index(exp_name)]
                            
                            # Create the subplot
                            ax = self.current_plot_fig.add_subplot(n_rows, n_cols, i+1)
                            
                            # Prepare data for the heatmap
                            heatmap_data = np.zeros((len(selected_orig_metrics), len(test_cases)))
                            heatmap_data.fill(np.nan)  # Fill with NaN for missing values
                            
                            # Fill in the data
                            for j, metric in enumerate(selected_orig_metrics):
                                if metric in self.test_case_data[exp_name]:
                                    for k, tc in enumerate(test_cases):
                                        if tc in self.test_case_data[exp_name][metric]:
                                            values = self.test_case_data[exp_name][metric][tc]
                                            if values:
                                                heatmap_data[j, k] = np.mean(values)
                            
                            # Create the heatmap
                            sns.heatmap(
                                heatmap_data,
                                ax=ax,
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax,
                                annot=True,  # Show values in cells
                                fmt=".2f",  # Format for annotations
                                linewidths=0.5,
                                cbar=True,
                                cbar_kws={"shrink": 0.8}
                            )
                            
                            # Set title and labels
                            ax.set_title(display_exp_name, fontsize=11)
                            ax.set_xlabel("Test Case", fontsize=9)
                            ax.set_ylabel("Metric", fontsize=9)
                            
                            # Set y-axis labels to metric names
                            ax.set_yticklabels(display_metric_names, rotation=0, fontsize=8)
                            
                            # Set x-axis labels to test case numbers
                            # If there are many test cases, show fewer labels
                            if len(test_case_labels) > 10:
                                step = max(1, len(test_case_labels) // 10)
                                visible_tick_indices = list(range(0, len(test_case_labels), step))
                                visible_ticks = [test_case_labels[i] for i in visible_tick_indices]
                                
                                # Create new tick positions and labels
                                ax.set_xticks([i * step for i in range(len(visible_ticks))])
                                ax.set_xticklabels(visible_ticks, rotation=45, fontsize=8)
                            else:
                                ax.set_xticklabels(test_case_labels, rotation=45, fontsize=8)
                        
                        # Add an overall title if provided
                        if custom_title:
                            self.current_plot_fig.suptitle(custom_title, fontsize=14, y=0.98)
                        else:
                            self.current_plot_fig.suptitle("Performance Heatmap by Model", fontsize=14, y=0.98)
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            self.current_plot_fig.text(0.5, 0.94, custom_subtitle, 
                                                 ha='center', va='top', fontsize=11, style='italic')
                        
                        # Adjust figure layout to make room for all subplots
                        # Removed tight_layout call to avoid colorbar conflicts
                
                # Skip common adjustments at the end as we've handled them here
                common_adjustments_needed = False
            elif plot_type == "Correlation Matrix":
                # Require at least 2 metrics for correlation
                if len(selected_orig_metrics) < 2:
                    ax.text(0.5, 0.5, "Please select at least two metrics for correlation analysis.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    # Create a DataFrame with all metrics data for each experiment
                    corr_data_by_exp = {}
                    
                    for exp_name in plot_exp_names:
                        # Get metric values for this experiment
                        metric_values = {}
                        
                        # Check if we have test case data available
                        if hasattr(self, 'test_case_data') and self.test_case_data and exp_name in self.test_case_data:
                            # Collect all metric values across test cases
                            for metric in selected_orig_metrics:
                                if metric in self.test_case_data[exp_name]:
                                    all_values = []
                                    for tc, values in self.test_case_data[exp_name][metric].items():
                                        all_values.extend(values)
                                    
                                    if all_values:
                                        metric_values[metric] = all_values
                        
                        # If we don't have test case data or it's incomplete, use mean values
                        missing_metrics = [m for m in selected_orig_metrics if m not in metric_values]
                        if missing_metrics:
                            for metric in missing_metrics:
                                if metric in self.experiment_data[exp_name]['mean']:
                                    # Just use the mean as a single value
                                    metric_values[metric] = [self.experiment_data[exp_name]['mean'][metric]]
                        
                        # Only include experiments with values for all selected metrics
                        if all(metric in metric_values for metric in selected_orig_metrics):
                            # Make sure all metrics have the same number of values
                            max_length = max(len(values) for values in metric_values.values())
                            for metric, values in metric_values.items():
                                if len(values) < max_length:
                                    # Repeat the value to match the max length
                                    metric_values[metric] = values * (max_length // len(values) + 1)
                                    metric_values[metric] = metric_values[metric][:max_length]
                            
                            # Create dataframe with metrics as columns
                            df = pd.DataFrame({display_metric_names[selected_orig_metrics.index(metric)]: values 
                                              for metric, values in metric_values.items()})
                            corr_data_by_exp[display_exp_names[plot_exp_names.index(exp_name)]] = df
                    
                    if not corr_data_by_exp:
                        ax.text(0.5, 0.5, "No complete data available for correlation analysis.",
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                        ax.axis('off')
                    else:
                        # Determine layout based on number of experiments
                        n_exps = len(corr_data_by_exp)
                        if n_exps == 1:
                            n_rows, n_cols = 1, 1
                        elif n_exps == 2:
                            n_rows, n_cols = 1, 2
                        elif n_exps <= 4:
                            n_rows, n_cols = 2, 2
                        elif n_exps <= 6:
                            n_rows, n_cols = 2, 3
                        else:
                            n_rows, n_cols = (n_exps + 2) // 3, 3
                        
                        # Clear the existing axis to make room for subplots
                        self.current_plot_fig.clear()
                        
                        # Disable constrained_layout to avoid colorbar conflicts
                        self.current_plot_fig.set_constrained_layout(False)
                        
                        # Set manual subplots adjustment
                        self.current_plot_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
                        
                        # Add overall title
                        if custom_title:
                            self.current_plot_fig.suptitle(custom_title, fontsize=14, y=0.98)
                        else:
                            self.current_plot_fig.suptitle("Correlation Matrix between Metrics", fontsize=14, y=0.98)
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            self.current_plot_fig.text(0.5, 0.94, custom_subtitle, 
                                                 ha='center', va='top', fontsize=11, style='italic')
                        
                        # Create correlation heatmaps for each experiment
                        for i, (exp_name, df) in enumerate(corr_data_by_exp.items()):
                            # Calculate correlation matrix
                            corr_matrix = df.corr()
                            
                            # Create subplot
                            ax = self.current_plot_fig.add_subplot(n_rows, n_cols, i+1)
                            
                            # Create heatmap
                            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
                            cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Red-blue color palette
                            
                            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                                       square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True,
                                       fmt=".2f", ax=ax)
                            
                            # Set title
                            ax.set_title(exp_name, fontsize=11)
                        
                        # We've handled the layout ourselves
                        common_adjustments_needed = False
            
            elif plot_type == "Radar Chart":
                # Need at least 3 metrics for a meaningful radar chart
                if len(selected_orig_metrics) < 3:
                    ax.text(0.5, 0.5, "Please select at least three metrics for radar chart.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    # Clear the existing axis
                    self.current_plot_fig.clear()
                    
                    # Add overall title
                    if custom_title:
                        self.current_plot_fig.suptitle(custom_title, fontsize=14, y=0.98)
                    else:
                        self.current_plot_fig.suptitle("Radar Chart of Metrics by Model", fontsize=14, y=0.98)
                    
                    # Add subtitle if provided
                    if custom_subtitle:
                        self.current_plot_fig.text(0.5, 0.94, custom_subtitle, 
                                             ha='center', va='top', fontsize=11, style='italic')
                    
                    # Create the radar chart
                    ax = self.current_plot_fig.add_subplot(111, polar=True)
                    
                    # Set number of angles (metrics)
                    angles = np.linspace(0, 2*np.pi, len(selected_orig_metrics), endpoint=False).tolist()
                    
                    # Close the polygon
                    angles += angles[:1]
                    
                    # Set labels
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(display_metric_names)
                    
                    # Draw axis lines for each angle and label
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    
                    # Draw the y-tick labels (no explicit ranges, automatically scaled)
                    ax.set_rlabel_position(0)
                    
                    # Plot each experiment
                    for i, exp_name in enumerate(plot_exp_names):
                        display_exp_name = display_exp_names[i]
                        
                        # Get values for each metric
                        values = []
                        for metric in selected_orig_metrics:
                            if metric in self.experiment_data[exp_name]['mean']:
                                values.append(self.experiment_data[exp_name]['mean'][metric])
                            else:
                                # Use 0 if metric not available
                                values.append(0)
                        
                        # Close the polygon by appending the first value
                        values += values[:1]
                        
                        # Plot values
                        ax.plot(angles, values, linewidth=1, linestyle='solid', label=display_exp_name, color=colors[i])
                        ax.fill(angles, values, alpha=0.1, color=colors[i])
                    
                    # Add legend
                    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    
                    # We've handled the layout ourselves
                    common_adjustments_needed = False
                    
            elif plot_type == "Violin Plot":
                # Similar to box plot but with distribution visualization
                metric = selected_orig_metrics[0]
                display_metric = display_metric_names[0]
                
                # We need individual values, not just mean/std
                if not hasattr(self, 'test_case_data') or not self.test_case_data:
                    # Try to use boxplot data if test case data not available
                    data_to_plot = [self.experiment_data[exp]['all_values'].get(metric, []) for exp in plot_exp_names]
                    
                    # Filter out experiments with no data for this metric
                    valid_data_indices = [i for i, d in enumerate(data_to_plot) if d]
                    filtered_data = [data_to_plot[i] for i in valid_data_indices]
                    filtered_labels = [display_exp_names[i] for i in valid_data_indices]
                    
                    if not filtered_data or all(len(d) == 0 for d in filtered_data):
                        ax.text(0.5, 0.5, "No distribution data available for violin plot. Please ensure you have test case data.",
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else:
                        # Create a DataFrame for seaborn
                        all_values = []
                        all_models = []
                        
                        for i, values in enumerate(filtered_data):
                            all_values.extend(values)
                            all_models.extend([filtered_labels[i]] * len(values))
                        
                        df = pd.DataFrame({
                            'Model': all_models,
                            'Value': all_values
                        })
                        
                        # Clear existing axes and create seaborn plot
                        self.current_plot_fig.clear()
                        ax = self.current_plot_fig.add_subplot(111)
                        
                        # Create the violin plot
                        sns.violinplot(x='Model', y='Value', data=df, 
                                     palette=colors[:len(filtered_labels)],
                                     inner='quartile', # Show quartiles inside
                                     ax=ax)
                        
                        # Add individual observations as points
                        sns.stripplot(x='Model', y='Value', data=df, 
                                    size=4, color='.3', alpha=0.7,
                                    ax=ax)
                        
                        # Add labels
                        ax.set_ylabel(display_metric)
                        if custom_title:
                            ax.set_title(custom_title, fontsize=14, pad=15)
                        else:
                            ax.set_title(f"Distribution of {display_metric}", fontsize=14, pad=15)
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                                   fontsize=11, ha='center', va='top', style='italic')
                else:
                    # Use test case data for more detailed violin plots
                    all_values = []
                    all_models = []
                    
                    for i, exp_name in enumerate(plot_exp_names):
                        display_exp_name = display_exp_names[i]
                        
                        if exp_name in self.test_case_data and metric in self.test_case_data[exp_name]:
                            # Collect all values across test cases
                            exp_values = []
                            for tc, values in self.test_case_data[exp_name][metric].items():
                                exp_values.extend(values)
                            
                            if exp_values:
                                all_values.extend(exp_values)
                                all_models.extend([display_exp_name] * len(exp_values))
                    
                    if not all_values:
                        ax.text(0.5, 0.5, f"No test case data available for metric '{display_metric}'.",
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else:
                        # Create a DataFrame for seaborn
                        df = pd.DataFrame({
                            'Model': all_models,
                            'Value': all_values
                        })
                        
                        # Clear existing axes and create seaborn plot
                        self.current_plot_fig.clear()
                        ax = self.current_plot_fig.add_subplot(111)
                        
                        # Create the violin plot
                        sns.violinplot(x='Model', y='Value', data=df, 
                                     palette=colors[:len(plot_exp_names)],
                                     inner='quartile', # Show quartiles inside
                                     ax=ax)
                        
                        # Add individual observations as points
                        sns.stripplot(x='Model', y='Value', data=df, 
                                    size=4, color='.3', alpha=0.7,
                                    ax=ax)
                        
                        # Add labels
                        ax.set_ylabel(display_metric)
                        if custom_title:
                            ax.set_title(custom_title, fontsize=14, pad=15)
                        else:
                            ax.set_title(f"Distribution of {display_metric}", fontsize=14, pad=15)
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                                   fontsize=11, ha='center', va='top', style='italic')
            
            elif plot_type == "Statistical Significance":
                if len(selected_orig_metrics) != 1:
                    ax.text(0.5, 0.5, "Please select exactly one metric for statistical significance visualization.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.axis('off')
                elif len(plot_exp_names) < 2:
                    ax.text(0.5, 0.5, "Please include at least two models for comparison.",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    metric = selected_orig_metrics[0]
                    display_metric = display_metric_names[0]
                    
                    # Get model values for this metric
                    model_values = []
                    
                    for exp_name in plot_exp_names:
                        if hasattr(self, 'test_case_data') and self.test_case_data and exp_name in self.test_case_data and metric in self.test_case_data[exp_name]:
                            # Get all values across test cases
                            values = []
                            for tc, vals in self.test_case_data[exp_name][metric].items():
                                values.extend(vals)
                            
                            if values:
                                model_values.append(values)
                            else:
                                # If no test case values, try to use mean
                                if metric in self.experiment_data[exp_name]['mean']:
                                    model_values.append([self.experiment_data[exp_name]['mean'][metric]])
                                else:
                                    model_values.append([])
                        else:
                            # Try to use mean if available
                            if metric in self.experiment_data[exp_name]['mean']:
                                model_values.append([self.experiment_data[exp_name]['mean'][metric]])
                            else:
                                model_values.append([])
                    
                    # Remove empty models
                    valid_indices = [i for i, values in enumerate(model_values) if values]
                    if not valid_indices:
                        ax.text(0.5, 0.5, f"No data available for metric '{display_metric}'.",
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else:
                        # Filter to valid models only
                        filtered_values = [model_values[i] for i in valid_indices]
                        filtered_names = [display_exp_names[i] for i in valid_indices]
                        
                        # Calculate means and confidence intervals
                        means = [np.mean(values) for values in filtered_values]
                        
                        # Calculate 95% confidence intervals
                        confidence_intervals = []
                        for values in filtered_values:
                            if len(values) > 1:
                                # Calculate standard error
                                stderr = np.std(values, ddof=1) / np.sqrt(len(values))
                                # 95% confidence interval (approximately 1.96 standard errors)
                                ci = 1.96 * stderr
                                confidence_intervals.append(ci)
                            else:
                                # Can't calculate CI with one value
                                confidence_intervals.append(0)
                        
                        # Clear existing axes
                        self.current_plot_fig.clear()
                        ax = self.current_plot_fig.add_subplot(111)
                        
                        # Plot bar chart with error bars
                        bars = ax.bar(filtered_names, means, yerr=confidence_intervals, 
                                     capsize=10, color=colors[:len(filtered_names)], alpha=0.7)
                        
                        # Add pairwise significance markers
                        if len(filtered_values) > 1:
                            # Calculate all pairwise p-values
                            p_values = np.zeros((len(filtered_values), len(filtered_values)))
                            
                            for i in range(len(filtered_values)):
                                for j in range(i+1, len(filtered_values)):
                                    # Skip if either has only one value
                                    if len(filtered_values[i]) <= 1 or len(filtered_values[j]) <= 1:
                                        p_values[i, j] = 1.0  # Not significant
                                        continue
                                    
                                    # Perform t-test for independent samples
                                    from scipy import stats
                                    _, p_value = stats.ttest_ind(filtered_values[i], filtered_values[j], 
                                                               equal_var=False)  # Welch's t-test
                                    p_values[i, j] = p_value
                                    p_values[j, i] = p_value  # Symmetric
                            
                            # Add significance markers
                            bar_width = bars[0].get_width()
                            bar_centers = [bar.get_x() + bar_width/2 for bar in bars]
                            y_max = max(means) + max(confidence_intervals) * 1.2
                            
                            significance_levels = [0.05, 0.01, 0.001]
                            significance_markers = ['*', '**', '***']
                            
                            # Add a legend for significance
                            ax.text(0.02, 0.98, '* p<0.05  ** p<0.01  *** p<0.001', 
                                   transform=ax.transAxes, fontsize=10, ha='left', va='top')
                            
                            for i in range(len(filtered_values)):
                                for j in range(i+1, len(filtered_values)):
                                    p = p_values[i, j]
                                    
                                    # Determine significance level
                                    marker = ''
                                    for k, level in enumerate(significance_levels):
                                        if p < level:
                                            marker = significance_markers[k]
                                    
                                    if marker:  # If significant
                                        # Draw a line connecting the bars
                                        height = y_max + 0.05 * (j - i)
                                        line_height = [height, height]
                                        line_x = [bar_centers[i], bar_centers[j]]
                                        
                                        ax.plot(line_x, line_height, '-k', linewidth=0.75)
                                        
                                        # Add the significance marker
                                        ax.text((bar_centers[i] + bar_centers[j]) / 2, height,
                                              marker, ha='center', va='bottom', fontsize=12)
                        
                        # Add labels and title
                        ax.set_ylabel(display_metric)
                        if custom_title:
                            ax.set_title(custom_title, fontsize=14, pad=15)
                        else:
                            ax.set_title(f"Statistical Comparison of {display_metric}", fontsize=14, pad=15)
                        
                        # Add subtitle if provided
                        if custom_subtitle:
                            ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                                   fontsize=11, ha='center', va='top', style='italic')
                        
                        # Adjust y-axis to include significance markers
                        if len(filtered_values) > 1:
                            ax.set_ylim(top=y_max * 1.2)
                        
                        # Add means as text on bars
                        for i, bar in enumerate(bars):
                            height = means[i]
                            ci = confidence_intervals[i]
                            ax.text(bar.get_x() + bar.get_width()/2, height + (ci if ci > 0 else 0) + 0.01,
                                  f'{height:.3f}' + (f'\n±{ci:.3f}' if ci > 0 else ''),
                                  ha='center', va='bottom', fontsize=9, linespacing=0.9)

            # Common adjustments (only for non-heatmap plots)
            if plot_type != "Heatmap Per Model":
                if ax.get_xticklabels(): # Check if x-tick labels exist
                    ax.tick_params(axis='x', rotation=45, labelsize=9)
                    # Adjust tick label properties directly for better control with rotation
                    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
                
                # Apply custom y-axis range if enabled
                if self.custom_y_range_var.get():
                    try:
                        y_min = float(self.y_min_entry.get()) if self.y_min_entry.get() else None
                        y_max = float(self.y_max_entry.get()) if self.y_max_entry.get() else None
                        
                        if y_min is not None and y_max is not None:
                            if y_min >= y_max:
                                messagebox.showwarning("Invalid Range", "Y-axis minimum must be less than maximum. Using auto-scaling.")
                            else:
                                ax.set_ylim(y_min, y_max)
                        elif y_min is not None:
                            ax.set_ylim(bottom=y_min)
                        elif y_max is not None:
                            ax.set_ylim(top=y_max)
                    except ValueError:
                        # If conversion fails, just use auto-scaling
                        pass
                
                # Add more space for rotated labels
                # Only apply tight_layout to plots without colorbars to avoid layout engine conflicts
                colorbar_plots = ["Heatmap Per Model", "Correlation Matrix", "Violin Plot", "Radar Chart", "Statistical Significance"]
                common_adjustments_needed = False
                if plot_type not in colorbar_plots and common_adjustments_needed:
                    try:
                        self.current_plot_fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
                    except Exception as e:
                        print(f"Warning: Could not apply tight_layout: {e}")

            # Create a new canvas if none exists, or update the existing one
            if self.plot_canvas is None:
                self.plot_canvas = FigureCanvasTkAgg(self.current_plot_fig, master=self.plot_canvas_frame)
                self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
                self.plot_canvas_widget.pack(fill=tk.BOTH, expand=True)
            else:
                # Update existing canvas with new figure
                self.plot_canvas.figure = self.current_plot_fig
            
            # Refresh canvas
            self.plot_canvas.draw()

        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred while generating the plot: {e}")
            print(f"Plotting error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if loading_win: loading_win.destroy()

    def _clear_current_plot(self):
        """Clear the current plot to free memory"""
        if hasattr(self, 'current_plot_fig') and self.current_plot_fig:
            plt.close(self.current_plot_fig)
            self.current_plot_fig = None
            
        # Clear any existing canvas widgets to avoid stacking
        if hasattr(self, 'plot_canvas_frame'):
            for widget in self.plot_canvas_frame.winfo_children():
                widget.destroy()
            self.plot_canvas = None

    def _save_plot(self):
        if not self.current_plot_fig:
            messagebox.showerror("Error", "No plot generated to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")],
            title="Save Plot As"
        )
        if not filepath:
            return

        try:
            # Use bbox_inches='tight' for better layout in saved file, especially with rotated labels
            self.current_plot_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save plot: {e}")

    def __del__(self):
        """Destructor to ensure all plots are closed when the app exits"""
        self._clear_current_plot()

    def _show_loading_window(self, message="Loading..."):
        loading_win = tk.Toplevel(self.master)
        loading_win.transient(self.master) # Show above master
        loading_win.grab_set() # Modal
        loading_win.title("")
        loading_win.resizable(False, False)
        
        # Remove title bar (more like a popup)
        # This might be OS-dependent or look odd on some systems.
        # try: loading_win.overrideredirect(True) # Risky, can make window unmanageable
        # except: pass

        ttk.Label(loading_win, text=message, font=("Arial", 11)).pack(padx=30, pady=20)
        
        # Center it
        self.master.update_idletasks() # Ensure master window dimensions are current
        master_x = self.master.winfo_x()
        master_y = self.master.winfo_y()
        master_w = self.master.winfo_width()
        master_h = self.master.winfo_height()
        win_w = loading_win.winfo_reqwidth() + 60 # Add padding
        win_h = loading_win.winfo_reqheight() + 40
        x_pos = master_x + (master_w // 2) - (win_w // 2)
        y_pos = master_y + (master_h // 2) - (win_h // 2)
        loading_win.geometry(f'{win_w}x{win_h}+{x_pos}+{y_pos}')
        return loading_win


if __name__ == '__main__':
    root = tk.Tk()
    # The NSOpenPanel warning on macOS is usually harmless, related to Tk file dialogs.
    app = RagEvaluationPlotter(root)
    root.mainloop()