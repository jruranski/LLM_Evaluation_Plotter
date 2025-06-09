"""
Controller for plotting operations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
import os

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
        # ax.set_ylabel("Metric Value")
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
        
        # ax.set_ylabel("Metric Value")
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
    
    def _generate_per_test_case_line_plot(self, experiment_names, selected_metrics,
                                         display_metrics, display_experiments,
                                         custom_title, custom_subtitle,
                                         custom_y_range, y_min, y_max):
        """Generate a line plot for test cases."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        if not self.model.has_test_case_data():
            ax.text(0.5, 0.5, "No per-test case data available. Please ensure your files have 'Test Case' column and metric scores.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Check if any experiment has data for any of the selected metrics
        has_data = False
        for exp_name in experiment_names:
            for metric in selected_metrics:
                if (exp_name in self.model.test_case_data and 
                    metric in self.model.test_case_data[exp_name] and 
                    self.model.test_case_data[exp_name][metric]):
                    has_data = True
                    break
            if has_data:
                break
        
        if not has_data:
            metrics_str = ", ".join([f"'{display_metrics.get(m, m)}'" for m in selected_metrics])
            ax.text(0.5, 0.5, f"No test case data available for metrics: {metrics_str}.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Get all test cases
        test_cases = self.model.get_all_test_cases(selected_metrics, experiment_names)
        if not test_cases:
            ax.text(0.5, 0.5, "No test cases available for the selected metrics and experiments.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        test_case_labels = [str(tc) for tc in test_cases]
        
        # Choose line style and marker based on the number of combinations
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        
        # Get colors - one color per experiment
        exp_colors = get_plot_color_palette(len(experiment_names))
        
        # Line plot for each experiment and metric
        legend_handles = []
        
        for i, exp_name in enumerate(experiment_names):
            display_exp_name = display_exp_names[i]
            exp_color = exp_colors[i]
            
            for j, metric in enumerate(selected_metrics):
                display_metric = display_metric_names[j]
                line_style = line_styles[j % len(line_styles)]
                marker = markers[(i*len(selected_metrics) + j) % len(markers)]
                
                # Collect data points for this experiment and metric
                x_vals = []
                y_vals = []
                
                for tc_idx, tc in enumerate(test_cases):
                    values = self.model.get_test_case_data(exp_name, metric, tc)
                    if values:
                        x_vals.append(tc_idx)
                        y_vals.append(np.mean(values))
                
                if x_vals:  # Only plot if we have data
                    line = ax.plot(
                        [test_case_labels[idx] for idx in x_vals], 
                        y_vals,
                        label=f"{display_exp_name} - {display_metric}",
                        color=exp_color,
                        linestyle=line_style,
                        marker=marker,
                        markersize=7,
                        linewidth=2,
                        alpha=0.8
                    )
                    legend_handles.append(line[0])
        
        # Set labels and titles
        ax.set_xlabel("Test Case")
        ax.set_ylabel("Average Value")
        
        # Add legend if we have multiple lines
        if len(legend_handles) > 1:
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        elif len(legend_handles) == 1:
            ax.legend(handles=legend_handles)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            "Test Case Comparison"
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        
        return True
    
    def _generate_heatmap_per_model(self, experiment_names, selected_metrics,
                                   display_metrics, display_experiments,
                                   custom_title, custom_subtitle):
        """Generate a heatmap per model."""
        # Create heatmap with one row per model and one column per metric
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        # Create a dataframe for the heatmap
        data = np.zeros((len(experiment_names), len(selected_metrics)))
        
        # Fill the dataframe with mean values
        for i, exp_name in enumerate(experiment_names):
            for j, metric in enumerate(selected_metrics):
                mean_val = self.model.get_experiment_mean(exp_name, metric)
                data[i, j] = mean_val if not pd.isna(mean_val) else 0
        
        # Create DataFrame for seaborn
        heatmap_df = pd.DataFrame(data, index=display_exp_names, columns=display_metric_names)
        
        # Set up colormap
        cmap = plt.cm.viridis  # or choose another: RdYlBu, coolwarm, YlGnBu, etc.
        
        # Create heatmap
        sns.heatmap(heatmap_df, ax=ax, cmap=cmap, annot=True, fmt=".3f", 
                  linewidths=1, cbar_kws={"shrink": 0.8}, vmin=None, vmax=None)
        
        # Adjust labels
        ax.set_ylabel("Models")
        ax.set_xlabel("Metrics")
        
        # Add title and subtitle
        fig_title = custom_title if custom_title else "Performance Heatmap by Model"
        plt.title(fig_title, fontsize=14, pad=20)
        
        if custom_subtitle:
            plt.text(0.5, 1.05, custom_subtitle, horizontalalignment='center',
                    fontsize=11, transform=ax.transAxes, style='italic')
        
        # Format axis ticks - we need special handling for heatmaps
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust the figure for the colorbar
        self.current_plot_fig.tight_layout()
        
        return True
    
    def _generate_correlation_matrix(self, experiment_names, selected_metrics,
                                    display_metrics, display_experiments,
                                    custom_title, custom_subtitle):
        """Generate a correlation matrix."""
        # Create a correlation matrix between metrics
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        if len(selected_metrics) < 2:
            ax.text(0.5, 0.5, "Correlation matrix requires at least 2 metrics.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Create a dataframe with all data points for correlation
        # We'll collect all values for each metric across all experiments
        all_data = {}
        
        for metric in selected_metrics:
            metric_values = []
            for exp_name in experiment_names:
                values = self.model.get_experiment_values(exp_name, metric)
                metric_values.extend(values)
            
            all_data[metric] = metric_values
        
        # Create pandas dataframe and ensure all columns have the same length
        # We need to pad with NaN as needed
        max_len = max(len(values) for values in all_data.values())
        for metric, values in all_data.items():
            if len(values) < max_len:
                all_data[metric] = values + [np.nan] * (max_len - len(values))
        
        # Create dataframe
        data_df = pd.DataFrame(all_data)
        
        # Rename columns to display names
        display_map = {metric: display_metrics.get(metric, metric) for metric in selected_metrics}
        data_df.rename(columns=display_map, inplace=True)
        
        # Compute correlation
        corr_matrix = data_df.corr(method='pearson')
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for the upper triangle
        cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Red-Blue diverging colormap
        
        sns.heatmap(corr_matrix, mask=mask, ax=ax, cmap=cmap, vmin=-1, vmax=1,
                   annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8})
        
        # Add title and subtitle
        fig_title = custom_title if custom_title else "Metric Correlation Matrix"
        plt.title(fig_title, fontsize=14, pad=20)
        
        if custom_subtitle:
            plt.text(0.5, 1.05, custom_subtitle, horizontalalignment='center',
                    fontsize=11, transform=ax.transAxes, style='italic')
        
        # Format axis ticks - we need special handling for heatmaps
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        # Adjust layout for the colorbar
        self.current_plot_fig.tight_layout()
        
        return True
    
    def _generate_radar_chart(self, experiment_names, selected_metrics,
                             display_metrics, display_experiments,
                             custom_title, custom_subtitle):
        """Generate a radar chart."""
        if len(selected_metrics) < 3:
            ax = self.current_plot_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Radar chart requires at least 3 metrics. Please select more metrics.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        # Number of variables
        N = len(selected_metrics)
        
        # Calculate angles for each metric (divide the plot into equal parts)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Get colors
        colors = get_plot_color_palette(len(experiment_names))
        
        # Create subplot with polar projection
        ax = self.current_plot_fig.add_subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], display_metric_names, size=9)
        
        # Calculate data values
        for i, exp_name in enumerate(experiment_names):
            display_exp_name = display_exp_names[i]
            values = []
            
            # Get values and normalize them to a common scale (0-1)
            all_values = []
            for metric in selected_metrics:
                value = self.model.get_experiment_mean(exp_name, metric)
                all_values.append(value if not pd.isna(value) else 0)
            
            # Normalize values between 0 and 1
            min_val = min(all_values)
            max_val = max(all_values)
            range_val = max_val - min_val if max_val > min_val else 1  # Avoid division by zero
            
            values = [(val - min_val) / range_val for val in all_values]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=display_exp_name)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title and subtitle
        fig_title = custom_title if custom_title else "Radar Chart Comparison"
        plt.title(fig_title, fontsize=14, y=1.1)
        
        if custom_subtitle:
            plt.text(0.5, 1.15, custom_subtitle, horizontalalignment='center',
                    fontsize=11, transform=ax.transAxes, style='italic')
        
        # Adjust layout for polar chart
        self.current_plot_fig.tight_layout()
        
        return True
    
    def _generate_violin_plot(self, experiment_names, metric,
                             display_metrics, display_experiments,
                             custom_title, custom_subtitle,
                             custom_y_range, y_min, y_max):
        """Generate a violin plot."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric = display_metrics.get(metric, metric)
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        
        # Get data for violin plot
        data_to_plot = []
        valid_exp_names = []
        
        for i, exp_name in enumerate(experiment_names):
            values = self.model.get_experiment_values(exp_name, metric)
            if values:  # Only include experiments with data
                data_to_plot.append(values)
                valid_exp_names.append(display_exp_names[i])
        
        if not data_to_plot:
            ax.text(0.5, 0.5, f"No data available for {display_metric} to create violin plot.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Get colors for the plot
        colors = get_plot_color_palette(len(data_to_plot))
        
        # Create violin plot
        vp = ax.violinplot(
            data_to_plot,
            showmeans=True,
            showmedians=True,
            showextrema=True
        )
        
        # Customize violin plot parts
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        # Customize other parts
        vp['cmeans'].set_color('black')
        vp['cmedians'].set_color('red')
        vp['cbars'].set_color('black')
        vp['cmins'].set_color('black')
        vp['cmaxes'].set_color('black')
        
        # Set axis labels
        ax.set_ylabel(display_metric)
        ax.set_xticks(np.arange(1, len(valid_exp_names) + 1))
        ax.set_xticklabels(valid_exp_names)
        
        # Add legend for mean and median
        mean_line = plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Mean')
        median_line = plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='Median')
        ax.legend(handles=[mean_line, median_line], loc='upper right')
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"Distribution of {display_metric}"
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        
        return True
    
    def _generate_statistical_significance(self, experiment_names, metric,
                                         display_metrics, display_experiments,
                                         custom_title, custom_subtitle,
                                         custom_y_range, y_min, y_max):
        """Generate a statistical significance plot."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric = display_metrics.get(metric, metric)
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        
        if len(experiment_names) < 2:
            ax.text(0.5, 0.5, "Statistical significance test requires at least 2 models to compare.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Get data for each experiment
        exp_data = {}
        for exp_name in experiment_names:
            values = self.model.get_experiment_values(exp_name, metric)
            if values:
                exp_data[exp_name] = values
        
        if len(exp_data) < 2:
            ax.text(0.5, 0.5, f"Not enough data for metric '{display_metric}' to perform statistical tests.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return True
        
        # Create a matrix for p-values
        n_exps = len(exp_data)
        p_values = np.ones((n_exps, n_exps))  # Default to 1.0 (no significance)
        
        # Calculate p-values for each pair using t-test
        from scipy import stats
        
        exp_list = list(exp_data.keys())
        for i in range(n_exps):
            for j in range(i+1, n_exps):
                data1 = exp_data[exp_list[i]]
                data2 = exp_data[exp_list[j]]
                
                # Check if we have enough data
                if len(data1) < 2 or len(data2) < 2:
                    p_values[i, j] = 1.0
                    p_values[j, i] = 1.0
                    continue
                
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)  # Welch's t-test
                
                p_values[i, j] = p_val
                p_values[j, i] = p_val
        
        # Create DataFrame for the heatmap
        display_names = [display_experiments.get(name, name) for name in exp_list]
        p_value_df = pd.DataFrame(p_values, index=display_names, columns=display_names)
        
        # Create heatmap
        # Define a custom colormap for p-values: red = significant, blue = not significant
        cmap = plt.cm.RdYlBu_r
        
        # Create mask to highlight the upper triangle only
        mask = np.triu(np.ones_like(p_values), k=0)
        
        # Plot the heatmap
        sns.heatmap(p_value_df, mask=mask, cmap=cmap, ax=ax, vmin=0, vmax=0.1, 
                   annot=True, fmt=".3f", linewidths=1, cbar_kws={"label": "p-value"})
        
        # Add significance stars
        for i in range(n_exps):
            for j in range(i+1, n_exps):
                p_val = p_values[i, j]
                stars = ""
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                
                if stars:
                    ax.text(j + 0.5, i + 0.85, stars, ha='center', va='center', color='black', fontweight='bold')
        
        # Add title and subtitle
        fig_title = custom_title if custom_title else f"Statistical Significance: {display_metric}"
        plt.title(fig_title, fontsize=14, pad=20)
        
        if custom_subtitle:
            plt.text(0.5, 1.05, custom_subtitle, horizontalalignment='center',
                    fontsize=11, transform=ax.transAxes, style='italic')
        
        # Add legend for significance levels
        legend_text = "Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001"
        ax.text(0.5, -0.15, legend_text, ha='center', va='center', 
               transform=ax.transAxes, fontsize=9, style='italic')
        
        # Format ticks
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Tight layout for better appearance with colorbar
        self.current_plot_fig.tight_layout()
        
        return True
    
    def export_to_latex(self, export_params, output_path=None):
        """
        Export data to LaTeX table format.
        
        Args:
            export_params: Dictionary containing export parameters
                - data_type: Type of data to export (e.g., 'means', 'raw', 'per_test_case')
                - selected_metrics: List of metrics to include
                - display_metrics: Dictionary mapping metric names to display names
                - display_experiments: Dictionary mapping experiment names to display names
                - caption: Optional caption for the table
                - label: Optional label for the table
            output_path: Path to save the LaTeX table. If None, returns the LaTeX code
            
        Returns:
            If output_path is None, returns the LaTeX code as a string.
            Otherwise, writes to file and returns True if successful.
        """
        try:
            # Extract parameters
            data_type = export_params.get('data_type', 'means')
            selected_metrics = export_params.get('selected_metrics', [])
            display_metrics = export_params.get('display_metrics', {})
            display_experiments = export_params.get('display_experiments', {})
            caption = export_params.get('caption', 'Experimental Results')
            label = export_params.get('label', 'tab:results')
            
            # Validate metrics
            if not selected_metrics:
                messagebox.showerror("Error", "Please select at least one metric to export.")
                return False
            
            # Get experiment data
            experiment_names = self.model.get_experiments()
            
            # Get display names
            display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
            display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
            
            # Generate LaTeX table based on data_type
            if data_type == 'means':
                latex_code = self._generate_means_table(
                    experiment_names, selected_metrics,
                    display_exp_names, display_metric_names,
                    caption, label
                )
            elif data_type == 'means_with_std':
                latex_code = self._generate_means_with_std_table(
                    experiment_names, selected_metrics,
                    display_exp_names, display_metric_names,
                    caption, label
                )
            elif data_type == 'per_test_case':
                latex_code = self._generate_per_test_case_table(
                    experiment_names, selected_metrics,
                    display_exp_names, display_metric_names,
                    caption, label
                )
            else:
                messagebox.showerror("Error", f"Unsupported data type: {data_type}")
                return False
            
            # Either return the LaTeX code or write to file
            if output_path:
                try:
                    with open(output_path, 'w') as f:
                        f.write(latex_code)
                    return True
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to write to file: {e}")
                    return False
            else:
                return latex_code
            
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting to LaTeX: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_means_table(self, experiment_names, metrics, 
                              display_exp_names, display_metric_names,
                              caption, label):
        """Generate a LaTeX table with mean values."""
        # Create header row with metrics
        header = " & ".join(["Model"] + display_metric_names) + " \\\\"
        
        # Create rows for each experiment
        rows = []
        for i, exp_name in enumerate(experiment_names):
            exp_display = display_exp_names[i]
            values = []
            
            for metric in metrics:
                mean_val = self.model.get_experiment_mean(exp_name, metric)
                # Format value with 3 decimal places
                if pd.notna(mean_val):
                    values.append(f"{mean_val:.3f}")
                else:
                    values.append("N/A")
            
            row = f"{exp_display} & " + " & ".join(values) + " \\\\"
            rows.append(row)
        
        # Combine into LaTeX table
        latex_code = "\\begin{table}[htbp]\n\\centering\n"
        latex_code += f"\\caption{{{caption}}}\n"
        latex_code += f"\\label{{{label}}}\n"
        latex_code += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex_code += "\\hline\n"
        latex_code += header + "\n"
        latex_code += "\\hline\n"
        latex_code += "\n".join(rows) + "\n"
        latex_code += "\\hline\n"
        latex_code += "\\end{tabular}\n"
        latex_code += "\\end{table}"
        
        return latex_code
    
    def _generate_means_with_std_table(self, experiment_names, metrics,
                                       display_exp_names, display_metric_names,
                                       caption, label):
        """Generate a LaTeX table with mean values and standard deviations."""
        # Create header row with metrics
        header = " & ".join(["Model"] + display_metric_names) + " \\\\"
        
        # Create rows for each experiment
        rows = []
        for i, exp_name in enumerate(experiment_names):
            exp_display = display_exp_names[i]
            values = []
            
            for metric in metrics:
                mean_val = self.model.get_experiment_mean(exp_name, metric)
                std_val = self.model.get_experiment_std(exp_name, metric)
                
                # Format value with mean and std
                if pd.notna(mean_val) and pd.notna(std_val):
                    values.append(f"{mean_val:.3f} $\\pm$ {std_val:.3f}")
                elif pd.notna(mean_val):
                    values.append(f"{mean_val:.3f}")
                else:
                    values.append("N/A")
            
            row = f"{exp_display} & " + " & ".join(values) + " \\\\"
            rows.append(row)
        
        # Combine into LaTeX table
        latex_code = "\\begin{table}[htbp]\n\\centering\n"
        latex_code += f"\\caption{{{caption}}}\n"
        latex_code += f"\\label{{{label}}}\n"
        latex_code += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex_code += "\\hline\n"
        latex_code += header + "\n"
        latex_code += "\\hline\n"
        latex_code += "\n".join(rows) + "\n"
        latex_code += "\\hline\n"
        latex_code += "\\end{tabular}\n"
        latex_code += "\\end{table}"
        
        return latex_code
    
    def _generate_per_test_case_table(self, experiment_names, metrics,
                                     display_exp_names, display_metric_names,
                                     caption, label):
        """Generate a LaTeX table with per-test case data."""
        # Check if test case data is available
        if not self.model.has_test_case_data():
            return "% No test case data available"
        
        # Get all test cases
        test_cases = self.model.get_all_test_cases(metrics, experiment_names)
        if not test_cases:
            return "% No test cases available for the selected metrics and experiments"
        
        # We'll organize data by test case and metric
        # Create header row with experiments
        header = "Test Case & Metric & " + " & ".join(display_exp_names) + " \\\\"
        
        # Create rows for each test case and metric
        rows = []
        for tc in test_cases:
            for j, metric in enumerate(metrics):
                display_metric = display_metric_names[j]
                values = []
                
                for exp_name in experiment_names:
                    tc_values = self.model.get_test_case_data(exp_name, metric, tc)
                    if tc_values:
                        mean_val = np.mean(tc_values)
                        values.append(f"{mean_val:.3f}")
                    else:
                        values.append("N/A")
                
                # Add a row for this test case and metric
                row = f"{tc} & {display_metric} & " + " & ".join(values) + " \\\\"
                rows.append(row)
            
            # Add a horizontal line between test cases
            rows.append("\\hline")
        
        # Combine into LaTeX table
        latex_code = "\\begin{table}[htbp]\n\\centering\n"
        latex_code += f"\\caption{{{caption}}}\n"
        latex_code += f"\\label{{{label}}}\n"
        latex_code += "\\begin{tabular}{llc" + "c" * len(experiment_names) + "}\n"
        latex_code += "\\hline\n"
        latex_code += header + "\n"
        latex_code += "\\hline\n"
        latex_code += "\n".join(rows) + "\n"
        latex_code += "\\end{tabular}\n"
        latex_code += "\\end{table}"
        
        return latex_code 