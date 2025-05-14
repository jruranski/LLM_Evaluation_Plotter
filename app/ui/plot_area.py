"""
Plot area UI component.
"""

import tkinter as tk
from tkinter import ttk

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
        
        save_button = ttk.Button(button_frame, text="Save Plot", command=self._save_plot)
        save_button.pack(side=tk.RIGHT)
    
    def _save_plot(self):
        """Save the current plot to a file."""
        self.file_controller.save_plot(self.plot_controller.current_plot_fig)
    
    def update_plot(self, plot_params):
        """
        Update the plot area with a new plot.
        
        Args:
            plot_params: Plot parameters dictionary
        """
        # Generate the plot
        self.plot_controller.generate_plot(plot_params, self.plot_canvas_frame) 