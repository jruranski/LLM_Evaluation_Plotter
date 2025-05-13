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
        plot_types = ["Bar Plot", "Grouped Bar Plot", "Line Plot", "Box Plot"]
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

        try:
            for exp_name, file_paths in grouped_files.items():
                run_summaries = [] # List of pd.Series, one for each run's summary
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
        if plot_type == "Box Plot" and len(selected_orig_metrics) > 1:
            messagebox.showwarning("Box Plot", "For Box Plot, typically one metric is selected for clarity. Plotting the first selected metric.")
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
            
            # Create new figure with specified size
            self.current_plot_fig = plt.figure(figsize=(plot_width, plot_height))
            self.current_plot_fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
            
            ax = self.current_plot_fig.add_subplot(111)

            num_exps = len(plot_exp_names)
            num_metrics = len(selected_orig_metrics)

            # Colors
            color_palette_name = "colorblind" if num_metrics > 1 or plot_type=="Box Plot" else "viridis" 
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


            # Common adjustments
            # ax.set_xlabel("Model / Experiment")
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
            self.current_plot_fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])

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