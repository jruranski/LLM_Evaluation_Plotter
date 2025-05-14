"""
Controller for plotting operations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox

from app.utils.plot_utils import (
    get_plot_color_palette,
    create_figure,
    apply_y_axis_range,
    add_title_and_subtitle,
    format_axis_ticks
)

class PlotController:
    """Controller for generating and managing plots."""
    
    def __init__(self, model):
        """
        Initialize the plot controller.
        
        Args:
            model: ExperimentData model instance
        """
        self.model = model
        self.current_plot_fig = None
        self.plot_canvas = None
    
    def clear_plot(self):
        """Clear the current plot to free memory."""
        if self.current_plot_fig:
            plt.close(self.current_plot_fig)
            self.current_plot_fig = None
    
    def update_canvas(self, canvas_frame):
        """
        Update or create the plot canvas.
        
        Args:
            canvas_frame: Frame to place the canvas in
        """
        # Clear any existing canvas widgets to avoid stacking
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        
        # Create a new canvas
        self.plot_canvas = FigureCanvasTkAgg(self.current_plot_fig, master=canvas_frame)
        self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
        self.plot_canvas_widget.pack(fill="both", expand=True)
        
        # Refresh canvas
        self.plot_canvas.draw()
    
    def generate_plot(self, plot_params, canvas_frame):
        """
        Generate a plot based on the given parameters.
        
        Args:
            plot_params: Dictionary containing plot parameters
            canvas_frame: Frame to place the canvas in
            
        Returns:
            True if plot generated successfully, False otherwise
        """
        try:
            # Extract parameters
            plot_type = plot_params.get('plot_type')
            selected_metrics = plot_params.get('selected_metrics', [])
            display_metrics = plot_params.get('display_metrics', {})
            display_experiments = plot_params.get('display_experiments', {})
            custom_title = plot_params.get('title')
            custom_subtitle = plot_params.get('subtitle')
            custom_y_range = plot_params.get('use_custom_y_range', False)
            y_min = plot_params.get('y_min')
            y_max = plot_params.get('y_max')
            plot_width = plot_params.get('width', 10)
            plot_height = plot_params.get('height', 6)
            show_outliers = plot_params.get('show_outliers', True)
            
            # Validate metrics
            if not selected_metrics:
                messagebox.showerror("Error", "Please select at least one metric to plot.")
                return False
            
            # Handle specific plot types that need only one metric
            single_metric_plots = ["Box Plot", "Per Test Case", "Statistical Significance"]
            if plot_type in single_metric_plots and len(selected_metrics) > 1:
                messagebox.showwarning(
                    f"{plot_type}", 
                    f"For {plot_type}, typically one metric is selected for clarity. Plotting the first selected metric."
                )
                selected_metrics = [selected_metrics[0]]
            
            # Get experiment data
            experiment_names = self.model.get_experiments()
            
            # Cleanup existing plot
            self.clear_plot()
            
            # Create new figure
            self.current_plot_fig = create_figure(plot_type, plot_width, plot_height)
            
            # Generate the appropriate plot
            if plot_type == "Bar Plot" and len(selected_metrics) == 1:
                success = self._generate_bar_plot(
                    experiment_names, selected_metrics[0], 
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max
                )
            elif plot_type == "Grouped Bar Plot" or (plot_type == "Bar Plot" and len(selected_metrics) > 1):
                success = self._generate_grouped_bar_plot(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max
                )
            elif plot_type == "Line Plot":
                success = self._generate_line_plot(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max
                )
            elif plot_type == "Box Plot":
                success = self._generate_box_plot(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max,
                    show_outliers
                )
            elif plot_type == "Per Test Case":
                success = self._generate_per_test_case_plot(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max
                )
            elif plot_type == "Per Test Case Line Plot":
                success = self._generate_per_test_case_line_plot(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max
                )
            elif plot_type == "Heatmap Per Model":
                success = self._generate_heatmap_per_model(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle
                )
            elif plot_type == "Correlation Matrix":
                success = self._generate_correlation_matrix(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle
                )
            elif plot_type == "Radar Chart":
                success = self._generate_radar_chart(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle
                )
            elif plot_type == "Violin Plot":
                success = self._generate_violin_plot(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max
                )
            elif plot_type == "Statistical Significance":
                success = self._generate_statistical_significance(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle,
                    custom_y_range, y_min, y_max
                )
            else:
                messagebox.showerror("Error", f"Plot type '{plot_type}' not implemented.")
                return False
            
            if success:
                # Update canvas with the new plot
                self.update_canvas(canvas_frame)
                return True
            return False
            
        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred while generating the plot: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_bar_plot(self, experiment_names, metric, 
                          display_metrics, display_experiments,
                          custom_title, custom_subtitle,
                          custom_y_range, y_min, y_max):
        """Generate a simple bar plot for a single metric."""
        display_metric = display_metrics.get(metric, metric)
        
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        # Get data
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        means = [self.model.get_experiment_mean(exp, metric) for exp in experiment_names]
        stds = [self.model.get_experiment_std(exp, metric) for exp in experiment_names]
        
        # Create plot
        colors = get_plot_color_palette(1, "viridis")
        bars = ax.bar(display_exp_names, means, yerr=stds, capsize=5, color=colors[0], alpha=0.8)
        ax.set_ylabel(display_metric)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"Comparison for {display_metric}"
        )
        
        # Add text labels on bars
        for i, bar_obj in enumerate(bars):
            yval = bar_obj.get_height()
            std_val = stds[i]
            text = f"{yval:.3f}"
            if pd.notna(std_val) and std_val > 1e-6:  # Don't show for NaN or very small std
                text += f"\n(±{std_val:.3f})"
            
            # Position text slightly above the bar or error bar
            text_y_pos = yval
            if pd.notna(std_val) and yval >= 0: text_y_pos += std_val
            elif pd.notna(std_val) and yval < 0: text_y_pos -= std_val  # Below for negative bars
            
            # Small offset from the top of the bar/error bar
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
            text_y_pos += offset if yval >= 0 else -offset
            
            ax.text(bar_obj.get_x() + bar_obj.get_width()/2.0, text_y_pos, text,
                   ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8, linespacing=0.9)
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        
        return True
    
    def _generate_grouped_bar_plot(self, experiment_names, selected_metrics,
                                  display_metrics, display_experiments, 
                                  custom_title, custom_subtitle,
                                  custom_y_range, y_min, y_max):
        """Generate a grouped bar plot for multiple metrics."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        # Get display names
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        # Construct DataFrame for grouped bar plot
        data_for_plot = {}
        for i, metric in enumerate(selected_metrics):
            data_for_plot[display_metric_names[i]] = [
                self.model.get_experiment_mean(exp, metric) for exp in experiment_names
            ]
        plot_df = pd.DataFrame(data_for_plot, index=display_exp_names)
        
        # Prepare yerr data if std is available
        yerr_data = None
        if all(self.model.get_experiment_std(exp, metric) is not None 
              for exp in experiment_names for metric in selected_metrics):
            yerr_values = {}
            for i, metric in enumerate(selected_metrics):
                yerr_values[display_metric_names[i]] = [
                    self.model.get_experiment_std(exp, metric) for exp in experiment_names
                ]
            yerr_data = pd.DataFrame(yerr_values, index=display_exp_names)
        
        # Get colors
        colors = get_plot_color_palette(len(selected_metrics))
        
        # Create plot
        plot_df.plot(kind='bar', ax=ax, yerr=yerr_data, capsize=4, 
                   color=colors[:len(selected_metrics)], alpha=0.8, width=0.8)
        ax.set_ylabel("Metric Value")
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            "Comparison Across Metrics"
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        
        return True
    
    def _generate_line_plot(self, experiment_names, selected_metrics,
                           display_metrics, display_experiments,
                           custom_title, custom_subtitle, 
                           custom_y_range, y_min, y_max):
        """Generate a line plot for one or more metrics."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        # Get display names
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        # Get colors
        colors = get_plot_color_palette(len(selected_metrics))
        
        # Create plot
        for i, metric in enumerate(selected_metrics):
            display_metric = display_metric_names[i]
            means = [self.model.get_experiment_mean(exp, metric) for exp in experiment_names]
            stds = [self.model.get_experiment_std(exp, metric) for exp in experiment_names]
            ax.errorbar(display_exp_names, means, yerr=stds, label=display_metric, 
                      marker='o', capsize=5, color=colors[i])
        
        ax.set_ylabel("Metric Value")
        if len(selected_metrics) > 1:
            ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            "Metric Trends"
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        
        return True
    
    def _generate_box_plot(self, experiment_names, metric,
                          display_metrics, display_experiments,
                          custom_title, custom_subtitle,
                          custom_y_range, y_min, y_max,
                          show_outliers):
        """Generate a box plot for a single metric."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric = display_metrics.get(metric, metric)
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        
        # Get data
        data_to_plot = [self.model.get_experiment_values(exp, metric) for exp in experiment_names]
        
        # Filter out experiments with no data for this metric
        valid_data_indices = [i for i, d in enumerate(data_to_plot) if d]
        filtered_data = [data_to_plot[i] for i in valid_data_indices]
        filtered_labels = [display_exp_names[i] for i in valid_data_indices]
        
        if not filtered_data:
            ax.text(0.5, 0.5, f"No data available for {display_metric} to create box plot.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Get colors
        colors = get_plot_color_palette(len(filtered_data))
        
        # Create plot
        bp = ax.boxplot(filtered_data, labels=filtered_labels, patch_artist=True,
                      showfliers=show_outliers,  # Show outliers based on user preference
                      medianprops=dict(color="black", linewidth=1.5))
        
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        
        ax.set_ylabel(display_metric)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"Distribution of {display_metric}"
        )
        
        # Add legend for outliers if they're shown
        if show_outliers and bp.get('fliers', []):
            outlier_label = plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='black', markersize=6, label='Outliers')
            ax.legend(handles=[outlier_label], loc='upper right')
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        
        return True
    
    def _generate_per_test_case_plot(self, experiment_names, metric,
                                    display_metrics, display_experiments,
                                    custom_title, custom_subtitle,
                                    custom_y_range, y_min, y_max):
        """Generate a per-test case bar plot."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric = display_metrics.get(metric, metric)
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        
        if not self.model.has_test_case_data():
            ax.text(0.5, 0.5, "No per-test case data available. Please ensure your files have 'Test Case' column and metric scores.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Check if any experiment has data for this metric
        has_data = False
        for exp_name in experiment_names:
            if exp_name in self.model.test_case_data and metric in self.model.test_case_data[exp_name]:
                has_data = True
                break
        
        if not has_data:
            ax.text(0.5, 0.5, f"No test case data available for metric '{display_metric}'.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Get all test cases
        test_cases = self.model.get_all_test_cases([metric], experiment_names)
        test_case_labels = [str(tc) for tc in test_cases]
        
        # Get colors
        colors = get_plot_color_palette(len(experiment_names))
        
        # Calculate bar positions
        num_exps = len(experiment_names)
        bar_width = 0.8 / num_exps
        offsets = np.arange(-(num_exps-1)/2, (num_exps-1)/2 + 0.1, 1) * bar_width
        
        # Plot bars for each experiment
        for i, exp_name in enumerate(experiment_names):
            display_exp_name = display_exp_names[i]
            means = []
            stds = []
            
            for tc in test_cases:
                values = self.model.get_test_case_data(exp_name, metric, tc)
                means.append(np.mean(values) if values else np.nan)
                stds.append(np.std(values) if len(values) > 1 else 0)
            
            # Calculate x positions for bars
            x_positions = np.arange(len(test_case_labels)) + offsets[i]
            
            # Plot bars
            ax.bar(x_positions, means, width=bar_width, yerr=stds,
                  label=display_exp_name, color=colors[i], alpha=0.8, capsize=3)
        
        # Set x-axis labels
        ax.set_xticks(np.arange(len(test_case_labels)))
        ax.set_xticklabels(test_case_labels)
        ax.set_xlabel("Test Case")
        ax.set_ylabel(display_metric)
        
        # Add legend
        if num_exps > 1:
            ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"{display_metric} per Test Case"
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        
        return True
    
    # Add other plotting methods as needed...
    
    def _generate_per_test_case_line_plot(self, experiment_names, selected_metrics,
                                         display_metrics, display_experiments,
                                         custom_title, custom_subtitle,
                                         custom_y_range, y_min, y_max):
        """Generate a line plot for test cases."""
        # Implement this method similar to _generate_per_test_case_plot but with line plot
        # For brevity, we'll focus on implementing just a few plot types in this example
        ax = self.current_plot_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Per Test Case Line Plot not implemented in this example.",
               horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return True
    
    def _generate_heatmap_per_model(self, experiment_names, selected_metrics,
                                   display_metrics, display_experiments,
                                   custom_title, custom_subtitle):
        """Generate a heatmap per model."""
        # Implement this method
        ax = self.current_plot_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Heatmap Per Model not implemented in this example.",
               horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return True
    
    def _generate_correlation_matrix(self, experiment_names, selected_metrics,
                                    display_metrics, display_experiments,
                                    custom_title, custom_subtitle):
        """Generate a correlation matrix."""
        # Implement this method
        ax = self.current_plot_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Correlation Matrix not implemented in this example.",
               horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return True
    
    def _generate_radar_chart(self, experiment_names, selected_metrics,
                             display_metrics, display_experiments,
                             custom_title, custom_subtitle):
        """Generate a radar chart."""
        # Implement this method
        ax = self.current_plot_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Radar Chart not implemented in this example.",
               horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return True
    
    def _generate_violin_plot(self, experiment_names, metric,
                             display_metrics, display_experiments,
                             custom_title, custom_subtitle,
                             custom_y_range, y_min, y_max):
        """Generate a violin plot."""
        # Implement this method
        ax = self.current_plot_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Violin Plot not implemented in this example.",
               horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return True
    
    def _generate_statistical_significance(self, experiment_names, metric,
                                         display_metrics, display_experiments,
                                         custom_title, custom_subtitle,
                                         custom_y_range, y_min, y_max):
        """Generate a statistical significance plot."""
        # Implement this method
        ax = self.current_plot_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Statistical Significance not implemented in this example.",
               horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return True 