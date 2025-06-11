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
    format_axis_ticks,
    apply_font_sizes_to_axis,
    get_font_sizes_from_params,
    configure_legend,
    get_legend_settings_from_params
)

class PlotController:
    """Controller for generating and managing plots."""
    
    # Polish translations for all plot text
    POLISH_TRANSLATIONS = {
        # Axis labels
        "Models": "Modele",
        "Metric Value": "Wartość metryki",
        "Average Value": "Średnia Wartość",
        "Test Case": "Przypadek Testowy",
        
        # Legend titles
        "Metrics": "Metryki",
        "Outliers": "Wartości Odstające",
        "Mean": "Średnia",
        "Median": "Mediana",
        
        # Default plot titles
        "Comparison for": "Porównanie dla",
        "Comparison Across Metrics": "Porównanie Różnych Metryk",
        "Metric Trends": "Trendy Metryk",
        "Distribution of": "Rozkład",
        "per Test Case": "na Przypadek Testowy",
        "Test Case Comparison": "Porównanie Przypadków Testowych",
        "Performance Heatmap by Model": "Mapa Cieplna Wydajności według Modelu",
        "Metric Correlation Matrix": "Macierz Korelacji Metryk",
        "Radar Chart Comparison": "Porównanie na Wykresie Radarowym",
        "Statistical Significance": "Istotność Statystyczna",
        
        # Error messages
        "No data available for": "Brak danych dla",
        "to create box plot.": "do utworzenia wykresu pudełkowego.",
        "to create violin plot.": "do utworzenia wykresu skrzypcowego.",
        "No per-test case data available. Please ensure your files have 'Test Case' column and metric scores.": 
            "Brak danych dla przypadków testowych. Upewnij się, że pliki zawierają kolumnę 'Test Case' i wyniki metryk.",
        "No test case data available for metric": "Brak danych przypadków testowych dla metryki",
        "No test case data available for metrics": "Brak danych przypadków testowych dla metryk",
        "No test cases available for the selected metrics and experiments.": 
            "Brak przypadków testowych dla wybranych metryk i eksperymentów.",
        "Correlation matrix requires at least 2 metrics.": "Macierz korelacji wymaga co najmniej 2 metryk.",
        "Radar chart requires at least 3 metrics. Please select more metrics.": 
            "Wykres radarowy wymaga co najmniej 3 metryk. Proszę wybrać więcej metryk.",
        "Statistical significance test requires at least 2 models to compare.": 
            "Test istotności statystycznej wymaga co najmniej 2 modeli do porównania.",
        "Not enough data for metric": "Niewystarczająco danych dla metryki",
        "to perform statistical tests.": "do przeprowadzenia testów statystycznych.",
        
        # Statistical significance
        "Significance levels": "Poziomy istotności",
        "p < 0.05": "p < 0,05",
        "p < 0.01": "p < 0,01", 
        "p < 0.001": "p < 0,001",
        "not significant": "nieistotne",
    }
    
    def _translate(self, text):
        """Translate text to Polish if translation exists."""
        return self.POLISH_TRANSLATIONS.get(text, text)
    
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
            custom_y_label = plot_params.get('y_axis_title')
            show_titles = plot_params.get('show_titles', True)
            custom_y_range = plot_params.get('use_custom_y_range', False)
            y_min = plot_params.get('y_min')
            y_max = plot_params.get('y_max')
            plot_width = plot_params.get('width', 10)
            plot_height = plot_params.get('height', 6)
            show_outliers = plot_params.get('show_outliers', True)
            
            # Get font sizes from parameters
            font_sizes = get_font_sizes_from_params(plot_params)
            
            # Get legend settings from parameters
            legend_settings = get_legend_settings_from_params(plot_params)
            
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
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max, font_sizes, legend_settings
                )
            elif plot_type == "Grouped Bar Plot" or (plot_type == "Bar Plot" and len(selected_metrics) > 1):
                success = self._generate_grouped_bar_plot(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max, font_sizes, legend_settings
                )
            elif plot_type == "Line Plot":
                success = self._generate_line_plot(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max, font_sizes, legend_settings
                )
            elif plot_type == "Box Plot":
                success = self._generate_box_plot(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max,
                    show_outliers, font_sizes, legend_settings
                )
            elif plot_type == "Per Test Case":
                success = self._generate_per_test_case_plot(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max, font_sizes, legend_settings
                )
            elif plot_type == "Per Test Case Line Plot":
                success = self._generate_per_test_case_line_plot(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max, font_sizes, legend_settings
                )
            elif plot_type == "Heatmap Per Model":
                success = self._generate_heatmap_per_model(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, show_titles, font_sizes, legend_settings
                )
            elif plot_type == "Correlation Matrix":
                success = self._generate_correlation_matrix(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, show_titles, font_sizes, legend_settings
                )
            elif plot_type == "Radar Chart":
                success = self._generate_radar_chart(
                    experiment_names, selected_metrics,
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, show_titles, font_sizes, legend_settings
                )
            elif plot_type == "Violin Plot":
                success = self._generate_violin_plot(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max, font_sizes, legend_settings
                )
            elif plot_type == "Statistical Significance":
                success = self._generate_statistical_significance(
                    experiment_names, selected_metrics[0],
                    display_metrics, display_experiments,
                    custom_title, custom_subtitle, custom_y_label, show_titles,
                    custom_y_range, y_min, y_max, font_sizes, legend_settings
                )
            else:
                messagebox.showerror("Error", f"Unknown plot type: {plot_type}")
                return False
            
            if success:
                self.update_canvas(canvas_frame)
                return True
            else:
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while generating the plot: {str(e)}")
            return False
    
    def _generate_bar_plot(self, experiment_names, metric, 
                          display_metrics, display_experiments,
                          custom_title, custom_subtitle, custom_y_label, show_titles,
                          custom_y_range, y_min, y_max, font_sizes, legend_settings):
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
        
        # Set axis labels
        y_label = custom_y_label if custom_y_label else display_metric
        ax.set_ylabel(y_label, fontsize=font_sizes['axis_title'])
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"{self._translate('Comparison for')} {display_metric}",
            show_titles, font_sizes
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
                   ha='center', va='bottom' if yval >= 0 else 'top', fontsize=font_sizes['annotations'], linespacing=0.9)
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks and apply font sizes
        format_axis_ticks(ax, font_sizes)
        apply_font_sizes_to_axis(ax, font_sizes)
        
        return True
    
    def _generate_grouped_bar_plot(self, experiment_names, selected_metrics,
                                  display_metrics, display_experiments, 
                                  custom_title, custom_subtitle, custom_y_label, show_titles,
                                  custom_y_range, y_min, y_max, font_sizes, legend_settings):
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
        
        # Set axis labels
        y_label = custom_y_label if custom_y_label else self._translate("Metric Value")
        ax.set_ylabel(y_label, fontsize=font_sizes['axis_title'])
        # Configure legend
        configure_legend(ax, legend_settings, font_sizes)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            self._translate("Comparison Across Metrics"),
            show_titles, font_sizes
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks and apply font sizes
        format_axis_ticks(ax, font_sizes)
        apply_font_sizes_to_axis(ax, font_sizes)
        
        return True
    
    def _generate_line_plot(self, experiment_names, selected_metrics,
                           display_metrics, display_experiments,
                           custom_title, custom_subtitle, custom_y_label, show_titles,
                           custom_y_range, y_min, y_max, font_sizes, legend_settings):
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
                      marker='o', capsize=5, color=colors[i], markersize=8, linewidth=2)
        
        # Set axis labels
        y_label = custom_y_label if custom_y_label else self._translate("Metric Value")
        ax.set_ylabel(y_label, fontsize=font_sizes['axis_title'])
        
        # Configure legend for multiple metrics
        if len(selected_metrics) > 1:
            configure_legend(ax, legend_settings, font_sizes)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            self._translate("Metric Trends"),
            show_titles, font_sizes
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks and apply font sizes
        format_axis_ticks(ax, font_sizes)
        apply_font_sizes_to_axis(ax, font_sizes)
        
        return True
    
    def _generate_box_plot(self, experiment_names, metric,
                          display_metrics, display_experiments,
                          custom_title, custom_subtitle, custom_y_label, show_titles,
                          custom_y_range, y_min, y_max,
                          show_outliers, font_sizes, legend_settings):
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
            ax.text(0.5, 0.5, f"{self._translate('No data available for')} {display_metric} {self._translate('to create box plot.')}",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes['annotations'])
            return True
        
        # Get colors
        colors = get_plot_color_palette(len(filtered_data))
        
        # Create plot
        bp = ax.boxplot(filtered_data, labels=filtered_labels, patch_artist=True,
                      showfliers=show_outliers,  # Show outliers based on user preference
                      medianprops=dict(color="black", linewidth=2))
        
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        
        # Set axis labels
        y_label = custom_y_label if custom_y_label else display_metric
        ax.set_ylabel(y_label, fontsize=font_sizes['axis_title'])
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"{self._translate('Distribution of')} {display_metric}",
            show_titles, font_sizes
        )
        
        # Add legend for outliers if they're shown
        if show_outliers and bp.get('fliers', []):
            outlier_label = plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='black', markersize=8, label=self._translate('Outliers'))
            ax.legend(handles=[outlier_label])
            configure_legend(ax, legend_settings, font_sizes)
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks and apply font sizes
        format_axis_ticks(ax, font_sizes)
        apply_font_sizes_to_axis(ax, font_sizes)
        
        return True
    
    def _generate_per_test_case_plot(self, experiment_names, metric,
                                    display_metrics, display_experiments,
                                    custom_title, custom_subtitle, custom_y_label, show_titles,
                                    custom_y_range, y_min, y_max, font_sizes, legend_settings):
        """Generate a per-test case bar plot."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric = display_metrics.get(metric, metric)
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        
        if not self.model.has_test_case_data():
            ax.text(0.5, 0.5, self._translate("No per-test case data available. Please ensure your files have 'Test Case' column and metric scores."),
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes['annotations'])
            return True
        
        # Check if any experiment has data for this metric
        has_data = False
        for exp_name in experiment_names:
            if exp_name in self.model.test_case_data and metric in self.model.test_case_data[exp_name]:
                has_data = True
                break
        
        if not has_data:
            ax.text(0.5, 0.5, f"{self._translate('No test case data available for metric')} '{display_metric}'.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes['annotations'])
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
        ax.set_xticklabels(test_case_labels, fontsize=font_sizes["tick_labels"], rotation=45, ha='right')
        ax.set_xlabel(self._translate("Test Case"), fontsize=font_sizes["axis_title"])
        
        # Set y-axis label
        y_label = custom_y_label if custom_y_label else display_metric
        ax.set_ylabel(y_label, fontsize=font_sizes["axis_title"])
        
        # Add legend
        if num_exps > 1:
            configure_legend(ax, legend_settings, font_sizes)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"{display_metric} {self._translate('per Test Case')}",
            show_titles
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        ax.tick_params(axis='y', which='major', labelsize=font_sizes["tick_labels"])
        
        return True
    
    def _generate_per_test_case_line_plot(self, experiment_names, selected_metrics,
                                         display_metrics, display_experiments,
                                         custom_title, custom_subtitle, custom_y_label, show_titles,
                                         custom_y_range, y_min, y_max, font_sizes, legend_settings):
        """Generate a line plot for test cases."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        if not self.model.has_test_case_data():
            ax.text(0.5, 0.5, self._translate("No per-test case data available. Please ensure your files have 'Test Case' column and metric scores."),
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
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
            ax.text(0.5, 0.5, f"{self._translate('No test case data available for metrics')}: {metrics_str}.",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
            return True
        
        # Get all test cases
        test_cases = self.model.get_all_test_cases(selected_metrics, experiment_names)
        if not test_cases:
            ax.text(0.5, 0.5, self._translate("No test cases available for the selected metrics and experiments."),
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
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
                        markersize=8,
                        linewidth=2,
                        alpha=0.8
                    )
                    legend_handles.append(line[0])
        
        # Set labels and titles
        ax.set_xlabel(self._translate("Test Case"), fontsize=font_sizes["axis_title"])
        
        # Set y-axis label
        y_label = custom_y_label if custom_y_label else self._translate("Average Value")
        ax.set_ylabel(y_label, fontsize=font_sizes["axis_title"])
        
        # Add legend if we have lines
        if legend_handles:
            configure_legend(ax, legend_settings, font_sizes)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            self._translate("Test Case Comparison"),
            show_titles, font_sizes
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        ax.tick_params(axis='both', which='major', labelsize=font_sizes["tick_labels"])
        
        return True
    
    def _generate_heatmap_per_model(self, experiment_names, selected_metrics,
                                   display_metrics, display_experiments,
                                   custom_title, custom_subtitle, show_titles, font_sizes, legend_settings):
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
                  linewidths=1, cbar_kws={"shrink": 0.8}, vmin=None, vmax=None,
                  annot_kws={"fontsize": font_sizes["annotations"]})
        
        # Adjust labels
        ax.set_xlabel(self._translate("Metrics"), fontsize=font_sizes["axis_title"])
        
        # Add title and subtitle
        fig_title = custom_title if custom_title else self._translate("Performance Heatmap by Model")
        if show_titles:
            plt.title(fig_title, fontsize=font_sizes["title"], pad=20)
        
        if custom_subtitle and show_titles:
            plt.text(0.5, 1.05, custom_subtitle, horizontalalignment='center',
                    fontsize=font_sizes["subtitle"], transform=ax.transAxes, style='italic')
        
        # Format axis ticks - we need special handling for heatmaps
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=font_sizes["tick_labels"])
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_sizes["tick_labels"])
        
        # Adjust the figure for the colorbar
        self.current_plot_fig.tight_layout()
        
        return True
    
    def _generate_correlation_matrix(self, experiment_names, selected_metrics,
                                    display_metrics, display_experiments,
                                    custom_title, custom_subtitle, show_titles, font_sizes, legend_settings):
        """Generate a correlation matrix."""
        # Create a correlation matrix between metrics
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        if len(selected_metrics) < 2:
            ax.text(0.5, 0.5, self._translate("Correlation matrix requires at least 2 metrics."),
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
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
                   annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8},
                   annot_kws={"fontsize": font_sizes["annotations"]})
        
        # Add title and subtitle
        fig_title = custom_title if custom_title else self._translate("Metric Correlation Matrix")
        if show_titles:
            plt.title(fig_title, fontsize=font_sizes["title"], pad=20)
        
        if custom_subtitle and show_titles:
            plt.text(0.5, 1.05, custom_subtitle, horizontalalignment='center',
                    fontsize=font_sizes["subtitle"], transform=ax.transAxes, style='italic')
        
        # Format axis ticks - we need special handling for heatmaps
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=font_sizes["tick_labels"])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_sizes["tick_labels"])
        
        # Adjust layout for the colorbar
        self.current_plot_fig.tight_layout()
        
        return True
    
    def _generate_radar_chart(self, experiment_names, selected_metrics,
                             display_metrics, display_experiments,
                             custom_title, custom_subtitle, show_titles, font_sizes, legend_settings):
        """Generate a radar chart comparing metrics across experiments."""
        # Check we have enough metrics for a radar chart
        if len(selected_metrics) < 3:
            ax = self.current_plot_fig.add_subplot(111)
            ax.text(0.5, 0.5, self._translate("Radar chart requires at least 3 metrics. Please select more metrics."),
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
            return True
            
        # Get display names
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        display_metric_names = [display_metrics.get(name, name) for name in selected_metrics]
        
        # Create the subplot with polar projection
        ax = self.current_plot_fig.add_subplot(111, polar=True)
        
        # Number of metrics (variables)
        N = len(selected_metrics)
        
        # Compute angle for each metric (in radians)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        # Complete the loop for plotting (IMPORTANT: must be a list for later use)
        theta_closed = np.concatenate((theta, [theta[0]]))
        
        # Get all data values for each metric to determine scale
        metric_min_max = {}
        for metric in selected_metrics:
            values = []
            for exp_name in experiment_names:
                val = self.model.get_experiment_mean(exp_name, metric)
                if pd.notna(val):
                    values.append(val)
            
            if values:
                min_val = min(values)
                max_val = max(values)
                # Ensure we don't have zero range
                if min_val == max_val:
                    if max_val == 0:
                        max_val = 1.0  # If all zeros, use 0-1 range
                    else:
                        max_val = min_val * 1.1  # Add 10% to create a range
                metric_min_max[metric] = (min_val, max_val)
            else:
                metric_min_max[metric] = (0, 1)  # Default if no data
        
        # Get color palette
        colors = get_plot_color_palette(len(experiment_names))
        
        # Plot each experiment
        for i, exp_name in enumerate(experiment_names):
            # Get data for this experiment
            raw_values = []
            missing_data = False
            
            for metric in selected_metrics:
                val = self.model.get_experiment_mean(exp_name, metric)
                
                # Handle missing values by recording we have missing data
                if pd.isna(val):
                    missing_data = True
                    raw_values.append(0)
                else:
                    raw_values.append(val)
            
            # Normalize values
            values = []
            for j, val in enumerate(raw_values):
                metric = selected_metrics[j]
                min_val, max_val = metric_min_max[metric]
                
                # Normalize to 0-1 range
                if max_val > min_val:
                    norm_val = (val - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5  # Default when all values are the same
                
                # Ensure we don't get negative values or values > 1
                norm_val = max(0, min(norm_val, 1))
                values.append(norm_val)
            
            # Make sure we have at least some data to plot
            if all(v == 0 for v in values) and missing_data:
                # If everything is 0 and we had missing data, use a small default
                # to ensure the polygon is visible
                values = [0.1] * len(values)
            
            # Convert to numpy array and close the loop
            values_array = np.array(values)
            values_closed = np.concatenate((values_array, [values_array[0]]))
            
            # Plot the polygon
            ax.plot(theta_closed, values_closed, 'o-', linewidth=3, markersize=8, 
                   label=display_exp_names[i], color=colors[i])
                   
            # Fill the polygon - make sure it's a proper polygon by using the same arrays
            ax.fill(theta_closed, values_closed, alpha=0.1, color=colors[i])
        
        # Customize the chart
        ax.set_xticks(theta)
        ax.set_xticklabels(display_metric_names, fontsize=font_sizes["tick_labels"])
        
        # Set y-ticks (circles)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='gray', fontsize=font_sizes["tick_labels"])
        ax.set_rlim(0, 1)
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        configure_legend(ax, legend_settings, font_sizes)
        
        # Add title and subtitle
        if show_titles:
            if custom_title:
                self.current_plot_fig.suptitle(custom_title, fontsize=font_sizes["title"], y=0.98)
            else:
                self.current_plot_fig.suptitle(self._translate("Radar Chart Comparison"), fontsize=font_sizes["title"], y=0.98)
                
            if custom_subtitle:
                self.current_plot_fig.text(0.5, 0.92, custom_subtitle, 
                                          ha='center', fontsize=font_sizes["subtitle"], style='italic')
        
        # Adjust layout to accommodate the legend
        self.current_plot_fig.tight_layout()
        self.current_plot_fig.subplots_adjust(bottom=0.25, right=0.85)
        
        return True
    
    def _generate_violin_plot(self, experiment_names, metric,
                             display_metrics, display_experiments,
                             custom_title, custom_subtitle, custom_y_label, show_titles,
                             custom_y_range, y_min, y_max, font_sizes, legend_settings):
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
            ax.text(0.5, 0.5, f"{self._translate('No data available for')} {display_metric} {self._translate('to create violin plot.')}",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
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
        vp['cmeans'].set_linewidth(2)
        vp['cmedians'].set_color('red')
        vp['cmedians'].set_linewidth(2)
        vp['cbars'].set_color('black')
        vp['cbars'].set_linewidth(2)
        vp['cmins'].set_color('black')
        vp['cmins'].set_linewidth(2)
        vp['cmaxes'].set_color('black')
        vp['cmaxes'].set_linewidth(2)
        
        # Set axis labels
        y_label = custom_y_label if custom_y_label else display_metric
        ax.set_ylabel(y_label, fontsize=font_sizes["axis_title"])
        ax.set_xlabel(self._translate("Models"), fontsize=font_sizes["axis_title"])
        ax.set_xticks(np.arange(1, len(valid_exp_names) + 1))
        ax.set_xticklabels(valid_exp_names, fontsize=font_sizes["tick_labels"])
        
        # Add legend for mean and median
        mean_line = plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label=self._translate('Mean'))
        median_line = plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2, label=self._translate('Median'))
        ax.legend(handles=[mean_line, median_line])
        configure_legend(ax, legend_settings, font_sizes)
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"{self._translate('Distribution of')} {display_metric}",
            show_titles
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks
        format_axis_ticks(ax)
        ax.tick_params(axis='y', which='major', labelsize=font_sizes["tick_labels"])
        
        return True
    
    def _generate_statistical_significance(self, experiment_names, metric,
                                         display_metrics, display_experiments,
                                         custom_title, custom_subtitle, custom_y_label, show_titles,
                                         custom_y_range, y_min, y_max, font_sizes, legend_settings):
        """Generate a statistical significance plot with bar plot."""
        # Add subplot
        ax = self.current_plot_fig.add_subplot(111)
        
        display_metric = display_metrics.get(metric, metric)
        display_exp_names = [display_experiments.get(name, name) for name in experiment_names]
        
        if len(experiment_names) < 2:
            ax.text(0.5, 0.5, self._translate("Statistical significance test requires at least 2 models to compare."),
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
            return True
        
        # Get data for each experiment
        exp_data = {}
        means = []
        stds = []
        
        for exp_name in experiment_names:
            values = self.model.get_experiment_values(exp_name, metric)
            if values and len(values) > 1:
                exp_data[exp_name] = values
                means.append(np.mean(values))
                stds.append(np.std(values) / np.sqrt(len(values)))  # Standard error
            else:
                means.append(0)
                stds.append(0)
        
        if len(exp_data) < 2:
            ax.text(0.5, 0.5, f"{self._translate('Not enough data for metric')} '{display_metric}' {self._translate('to perform statistical tests.')}",
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=font_sizes["axis_title"])
            return True
        
        # Import statistical test
        from scipy import stats
        
        # Get colors
        colors = get_plot_color_palette(len(experiment_names))
        
        # Set up bar positions
        x_pos = np.arange(len(experiment_names))
        
        # Create bar plot
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
        
        # Add labels and title
        # Removed x-axis label for models
        
        # Set y-axis label
        y_label = custom_y_label if custom_y_label else display_metric
        ax.set_ylabel(y_label, fontsize=font_sizes["axis_title"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_exp_names, rotation=45, ha='right', fontsize=font_sizes["tick_labels"])
        
        # Add text labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if pd.notna(height) and height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.02*max(means),
                       f'{height:.3f}', ha='center', va='bottom', fontsize=font_sizes["annotations"])
        
        # Add significance annotations
        def get_significance_marker(p_value):
            if p_value < 0.001:
                return "***"
            elif p_value < 0.01:
                return "**"
            elif p_value < 0.05:
                return "*"
            else:
                return "ns"
        
        # Compute significance between pairs of experiments and add brackets
        y_max_plot = max(means) + max(stds) * 2
        y_offset = (max(means) - min([m for m in means if m > 0])) * 0.1
        
        # Dictionary to track comparison brackets
        significance_brackets = []
        
        for i in range(len(experiment_names)):
            for j in range(i+1, len(experiment_names)):
                # Skip if either experiment doesn't have data
                if experiment_names[i] not in exp_data or experiment_names[j] not in exp_data:
                    continue
                
                # Calculate p-value for this pair
                data1 = exp_data[experiment_names[i]]
                data2 = exp_data[experiment_names[j]]
                _, p_val = stats.ttest_ind(data1, data2, equal_var=False)  # Welch's t-test
                
                # Get significance marker
                marker = get_significance_marker(p_val)
                
                if marker != "ns":  # Only add significant comparisons
                    # Calculate bracket height
                    height1 = means[i] + stds[i]
                    height2 = means[j] + stds[j]
                    bracket_height = max(height1, height2) + y_offset
                    
                    # Keep track of this bracket to avoid overlaps
                    significance_brackets.append((i, j, bracket_height, marker))
        
        # Sort brackets by span to draw wider ones higher
        significance_brackets.sort(key=lambda x: abs(x[1] - x[0]), reverse=True)
        
        # Draw brackets
        for idx, (i, j, height, marker) in enumerate(significance_brackets):
            # Adjust height to avoid overlaps
            adjusted_height = height + idx * y_offset
            
            # Draw the bracket
            x1, x2 = x_pos[i], x_pos[j]
            bar_width = bars[0].get_width()
            
            # horizontal lines
            ax.plot([x1, x1, x2, x2], 
                   [adjusted_height - 0.02*y_offset, adjusted_height, adjusted_height, adjusted_height - 0.02*y_offset],
                   color='black', linewidth=2)
            
            # Add marker
            ax.text((x1 + x2) / 2, adjusted_height, marker, 
                   ha='center', va='bottom', fontsize=font_sizes["legend"], fontweight='bold')
        
        # Add title and subtitle
        add_title_and_subtitle(
            self.current_plot_fig, ax, 
            custom_title, custom_subtitle, 
            f"{self._translate('Statistical Significance')}: {display_metric}",
            show_titles, font_sizes
        )
        
        # Apply custom y-axis range if requested
        apply_y_axis_range(ax, custom_y_range, y_min, y_max)
        
        # Format axis ticks and apply font sizes
        format_axis_ticks(ax, font_sizes)
        apply_font_sizes_to_axis(ax, font_sizes)
        
        # Adjust layout first to get proper spacing
        self.current_plot_fig.tight_layout()
        
        # Add legend for significance levels after layout adjustment
        # Position it lower to avoid overlap with rotated labels
        legend_text = f"{self._translate('Significance levels')}: * {self._translate('p < 0.05')}, ** {self._translate('p < 0.01')}, *** {self._translate('p < 0.001')}, ns: {self._translate('not significant')}"
        
        # Calculate appropriate position based on figure size and font size
        # Use figure coordinates instead of axes coordinates for better control
        legend_y_pos = 0.02  # Small margin from bottom of figure
        
        self.current_plot_fig.text(0.5, legend_y_pos, legend_text, ha='center', va='bottom', 
                                  fontsize=font_sizes["annotations"], style='italic',
                                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                                  transform=self.current_plot_fig.transFigure)
        
        # Adjust subplot to make room for the legend
        self.current_plot_fig.subplots_adjust(bottom=0.25)
        
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