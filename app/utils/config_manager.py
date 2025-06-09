"""
Configuration manager for saving and loading plot configurations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Dict, List, Optional, Any

from app.config import DEFAULT_FONT_SIZES, FONT_SIZE_PRESETS

class ConfigManager:
    """Manages saving and loading of plot configurations."""
    
    CONFIG_VERSION = "1.0"
    CONFIG_EXTENSION = ".plotconfig"
    
    def __init__(self):
        """Initialize the configuration manager."""
        pass
    
    def save_config(self, config_data: Dict[str, Any], file_path: str) -> bool:
        """
        Save configuration to a JSON file.
        
        Args:
            config_data: Dictionary containing all configuration data
            file_path: Path where to save the configuration file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Add metadata to the configuration
            config_with_meta = {
                "version": self.CONFIG_VERSION,
                "created_at": datetime.now().isoformat(),
                "config": config_data
            }
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file with proper formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration:\n{str(e)}")
            return False
    
    def load_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Configuration dictionary if loaded successfully, None otherwise
        """
        try:
            if not os.path.exists(file_path):
                messagebox.showerror("Load Error", f"Configuration file not found:\n{file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_with_meta = json.load(f)
            
            # Validate configuration structure
            if not self._validate_config_structure(config_with_meta):
                return None
            
            # Extract the actual configuration data
            config_data = config_with_meta.get("config", {})
            
            # Validate and fix any missing or invalid values
            config_data = self._validate_and_fix_config_data(config_data)
            
            return config_data
            
        except json.JSONDecodeError as e:
            messagebox.showerror("Load Error", f"Invalid configuration file format:\n{str(e)}")
            return None
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration:\n{str(e)}")
            return None
    
    def collect_config_from_ui(self, control_panel, file_controller) -> Dict[str, Any]:
        """
        Collect current configuration from UI components.
        
        Args:
            control_panel: ControlPanel instance
            file_controller: FileController instance
            
        Returns:
            Dictionary containing all current settings
        """
        config = {
            # File information
            "files": {
                "selected_files": file_controller.selected_files.copy(),
                "file_count": len(file_controller.selected_files)
            },
            
            # Plot settings
            "plot_settings": {
                "plot_type": control_panel.plot_type_var.get(),
                "show_titles": control_panel.show_titles_var.get(),
                "show_outliers": control_panel.show_outliers_var.get(),
                "output_format": control_panel.output_format_var.get()
            },
            
            # Font settings
            "font_settings": {
                "font_preset": control_panel.font_preset_var.get(),
                "individual_sizes": {
                    key: var.get() for key, var in control_panel.font_size_vars.items()
                }
            },
            
            # Y-axis settings
            "y_axis_settings": {
                "use_custom_range": control_panel.custom_y_range_var.get(),
                "y_min": control_panel.y_min_entry.get(),
                "y_max": control_panel.y_max_entry.get()
            },
            
            # Plot size settings
            "plot_size": {
                "width": control_panel.plot_width_entry.get(),
                "height": control_panel.plot_height_entry.get()
            },
            
            # Title settings
            "title_settings": {
                "main_title": control_panel.title_entry.get(),
                "subtitle": control_panel.subtitle_entry.get(),
                "y_axis_title": control_panel.y_axis_title_entry.get()
            },
            
            # LaTeX settings
            "latex_settings": {
                "table_type": control_panel.latex_table_type_var.get(),
                "caption": control_panel.latex_caption_entry.get(),
                "label": control_panel.latex_label_entry.get()
            },
            
            # Selected metrics
            "selected_metrics": [
                metric for metric, var in control_panel.metric_vars.items() if var.get()
            ],
            
            # Label customizations
            "label_customizations": {
                "metric_renames": {
                    metric: entry.get() for metric, entry in control_panel.metric_rename_entries.items()
                },
                "experiment_renames": {
                    exp: entry.get() for exp, entry in control_panel.model_rename_entries.items()
                }
            },
            
            # Metadata
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "app_version": "1.0"
            }
        }
        
        return config
    
    def apply_config_to_ui(self, config: Dict[str, Any], control_panel, file_controller) -> bool:
        """
        Apply loaded configuration to UI components.
        
        Args:
            config: Configuration dictionary
            control_panel: ControlPanel instance
            file_controller: FileController instance
            
        Returns:
            True if applied successfully, False otherwise
        """
        try:
            # Load files
            files_config = config.get("files", {})
            selected_files = files_config.get("selected_files", [])
            
            if selected_files:
                # Validate file paths and handle missing files
                existing_files, missing_files = self._validate_file_paths(selected_files)
                
                if missing_files:
                    missing_list = "\n".join(missing_files)
                    response = messagebox.askyesno(
                        "Missing Files",
                        f"The following files from the configuration are missing:\n\n{missing_list}\n\n"
                        f"Do you want to continue loading the configuration with the available files?"
                    )
                    if not response:
                        return False
                
                if existing_files:
                    # Clear current files and load new ones
                    file_controller.clear_files()
                    for file_path in existing_files:
                        file_controller.selected_files.append(file_path)
                    
                    # Update UI and load data
                    control_panel._update_file_listbox()
                    if file_controller.load_and_process_files():
                        control_panel._update_ui_after_load()
                else:
                    messagebox.showwarning("No Files", "No valid files found in the configuration.")
                    return False
            
            # Apply plot settings
            plot_settings = config.get("plot_settings", {})
            control_panel.plot_type_var.set(plot_settings.get("plot_type", "Bar Plot"))
            control_panel.show_titles_var.set(plot_settings.get("show_titles", True))
            control_panel.show_outliers_var.set(plot_settings.get("show_outliers", True))
            control_panel.output_format_var.set(plot_settings.get("output_format", "Plot"))
            
            # Apply font settings
            font_settings = config.get("font_settings", {})
            control_panel.font_preset_var.set(font_settings.get("font_preset", "Large"))
            
            individual_sizes = font_settings.get("individual_sizes", {})
            for key, var in control_panel.font_size_vars.items():
                if key in individual_sizes:
                    var.set(individual_sizes[key])
            
            # Apply Y-axis settings
            y_axis_settings = config.get("y_axis_settings", {})
            control_panel.custom_y_range_var.set(y_axis_settings.get("use_custom_range", False))
            control_panel.y_min_entry.delete(0, 'end')
            control_panel.y_min_entry.insert(0, y_axis_settings.get("y_min", ""))
            control_panel.y_max_entry.delete(0, 'end')
            control_panel.y_max_entry.insert(0, y_axis_settings.get("y_max", ""))
            
            # Apply plot size settings
            plot_size = config.get("plot_size", {})
            control_panel.plot_width_entry.delete(0, 'end')
            control_panel.plot_width_entry.insert(0, plot_size.get("width", "10"))
            control_panel.plot_height_entry.delete(0, 'end')
            control_panel.plot_height_entry.insert(0, plot_size.get("height", "6"))
            
            # Apply title settings
            title_settings = config.get("title_settings", {})
            control_panel.title_entry.delete(0, 'end')
            control_panel.title_entry.insert(0, title_settings.get("main_title", ""))
            control_panel.subtitle_entry.delete(0, 'end')
            control_panel.subtitle_entry.insert(0, title_settings.get("subtitle", ""))
            control_panel.y_axis_title_entry.delete(0, 'end')
            control_panel.y_axis_title_entry.insert(0, title_settings.get("y_axis_title", ""))
            
            # Apply LaTeX settings
            latex_settings = config.get("latex_settings", {})
            control_panel.latex_table_type_var.set(latex_settings.get("table_type", "Means only"))
            control_panel.latex_caption_entry.delete(0, 'end')
            control_panel.latex_caption_entry.insert(0, latex_settings.get("caption", "Experimental Results"))
            control_panel.latex_label_entry.delete(0, 'end')
            control_panel.latex_label_entry.insert(0, latex_settings.get("label", "tab:results"))
            
            # Apply selected metrics
            selected_metrics = config.get("selected_metrics", [])
            for metric, var in control_panel.metric_vars.items():
                var.set(metric in selected_metrics)
            
            # Apply label customizations
            label_customizations = config.get("label_customizations", {})
            
            # Apply metric renames
            metric_renames = label_customizations.get("metric_renames", {})
            for metric, entry in control_panel.metric_rename_entries.items():
                if metric in metric_renames:
                    entry.delete(0, 'end')
                    entry.insert(0, metric_renames[metric])
            
            # Apply experiment renames
            experiment_renames = label_customizations.get("experiment_renames", {})
            for exp, entry in control_panel.model_rename_entries.items():
                if exp in experiment_renames:
                    entry.delete(0, 'end')
                    entry.insert(0, experiment_renames[exp])
            
            # Update output format display
            control_panel._toggle_output_format()
            
            return True
            
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to apply configuration:\n{str(e)}")
            return False
    
    def _validate_config_structure(self, config_with_meta: Dict[str, Any]) -> bool:
        """
        Validate the basic structure of a configuration file.
        
        Args:
            config_with_meta: Configuration data with metadata
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            if not isinstance(config_with_meta, dict):
                messagebox.showerror("Load Error", "Invalid configuration file: not a valid JSON object")
                return False
            
            if "config" not in config_with_meta:
                messagebox.showerror("Load Error", "Invalid configuration file: missing 'config' section")
                return False
            
            version = config_with_meta.get("version", "unknown")
            if version != self.CONFIG_VERSION:
                response = messagebox.askyesno(
                    "Version Mismatch",
                    f"Configuration file version ({version}) differs from current version ({self.CONFIG_VERSION}).\n"
                    f"Do you want to try loading it anyway? Some features may not work correctly."
                )
                if not response:
                    return False
            
            return True
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Error validating configuration structure:\n{str(e)}")
            return False
    
    def _validate_and_fix_config_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix configuration data with sensible defaults.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            Validated and fixed configuration data
        """
        # Ensure all required sections exist with defaults
        defaults = {
            "files": {"selected_files": [], "file_count": 0},
            "plot_settings": {
                "plot_type": "Bar Plot",
                "show_titles": True,
                "show_outliers": True,
                "output_format": "Plot"
            },
            "font_settings": {
                "font_preset": "Large",
                "individual_sizes": DEFAULT_FONT_SIZES.copy()
            },
            "y_axis_settings": {
                "use_custom_range": False,
                "y_min": "",
                "y_max": ""
            },
            "plot_size": {"width": "10", "height": "6"},
            "title_settings": {"main_title": "", "subtitle": "", "y_axis_title": ""},
            "latex_settings": {
                "table_type": "Means only",
                "caption": "Experimental Results",
                "label": "tab:results"
            },
            "selected_metrics": [],
            "label_customizations": {
                "metric_renames": {},
                "experiment_renames": {}
            }
        }
        
        # Merge with defaults
        for section, default_values in defaults.items():
            if section not in config_data:
                config_data[section] = default_values
            elif isinstance(default_values, dict):
                for key, default_value in default_values.items():
                    if key not in config_data[section]:
                        config_data[section][key] = default_value
        
        # Validate font preset
        font_preset = config_data["font_settings"]["font_preset"]
        if font_preset not in FONT_SIZE_PRESETS:
            config_data["font_settings"]["font_preset"] = "Large"
        
        # Validate individual font sizes
        individual_sizes = config_data["font_settings"]["individual_sizes"]
        for key in DEFAULT_FONT_SIZES:
            if key not in individual_sizes:
                individual_sizes[key] = str(DEFAULT_FONT_SIZES[key])
            else:
                # Ensure it's a string representation of a positive integer
                try:
                    size = int(individual_sizes[key])
                    if size <= 0:
                        individual_sizes[key] = str(DEFAULT_FONT_SIZES[key])
                    else:
                        individual_sizes[key] = str(size)
                except (ValueError, TypeError):
                    individual_sizes[key] = str(DEFAULT_FONT_SIZES[key])
        
        return config_data
    
    def _validate_file_paths(self, file_paths: List[str]) -> tuple[List[str], List[str]]:
        """
        Validate that file paths exist and separate existing from missing files.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Tuple of (existing_files, missing_files)
        """
        existing_files = []
        missing_files = []
        
        for file_path in file_paths:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        return existing_files, missing_files
    
    def get_default_config_name(self) -> str:
        """
        Generate a default configuration file name with timestamp.
        
        Returns:
            Default configuration file name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"plot_config_{timestamp}{self.CONFIG_EXTENSION}"
    
    def get_config_filter(self) -> str:
        """
        Get file dialog filter for configuration files.
        
        Returns:
            File dialog filter string
        """
        return f"Plot Configuration files (*{self.CONFIG_EXTENSION})" 