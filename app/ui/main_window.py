"""
Main application window.
"""

import tkinter as tk
from tkinter import ttk
from app.config import DEFAULT_WINDOW_SIZE
from app.ui.control_panel import ControlPanel
from app.ui.plot_area import PlotArea
from app.utils.ui_utils import create_loading_window

class MainWindow:
    """Main application window."""
    
    def __init__(self, title, file_controller, plot_controller):
        """
        Initialize the main window.
        
        Args:
            title: Window title
            file_controller: FileController instance
            plot_controller: PlotController instance
        """
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(DEFAULT_WINDOW_SIZE)
        
        self.file_controller = file_controller
        self.plot_controller = plot_controller
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the main UI elements."""
        # Main PanedWindow for resizable sections
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control Panel (Left)
        control_panel_outer = ttk.Frame(paned_window, width=400)  # Initial width
        paned_window.add(control_panel_outer, weight=1)  # weight determines resize proportion
        
        # Plot Area (Right)
        plot_area_frame = ttk.Frame(paned_window)
        paned_window.add(plot_area_frame, weight=3)
        
        # Create Control Panel component
        self.control_panel = ControlPanel(control_panel_outer, self.file_controller, self.plot_controller)
        
        # Create Plot Area component
        self.plot_area = PlotArea(plot_area_frame, self.file_controller, self.plot_controller)
        
        # Connect the components
        self.control_panel.on_generate_plot = self._generate_plot_with_indicator
    
    def _generate_plot_with_indicator(self, plot_params):
        """
        Generate a plot with a loading indicator.
        
        Args:
            plot_params: Plot parameters dictionary
        """
        loading_win = create_loading_window(self.root, "Generating plot...")
        self.root.update_idletasks()
        
        try:
            self.plot_area.update_plot(plot_params)
        finally:
            if loading_win:
                loading_win.destroy()
    
    def run(self):
        """Run the application main loop."""
        self.root.mainloop()
    
    def __del__(self):
        """Clean up resources when the application exits."""
        # Ensure plot resources are properly released
        if self.plot_controller:
            self.plot_controller.clear_plot() 