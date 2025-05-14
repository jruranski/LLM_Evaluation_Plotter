"""
LLM Evaluation Plotter - Main Application

A tool for visualizing and analyzing Large Language Model evaluation results.
"""

import matplotlib
matplotlib.use('TkAgg')  # Explicitly set backend for Tkinter compatibility

from app.config import configure_matplotlib
from app.models.experiment_data import ExperimentData
from app.controllers.file_controller import FileController
from app.controllers.plot_controller import PlotController
from app.ui.main_window import MainWindow

def main():
    """Main application entry point."""
    # Configure matplotlib
    configure_matplotlib()
    
    # Create model
    model = ExperimentData()
    
    # Create controllers
    file_controller = FileController(model)
    plot_controller = PlotController(model)
    
    # Create main window
    app = MainWindow("RAG Evaluation Plotter", file_controller, plot_controller)
    
    # Run the application
    app.run()

if __name__ == "__main__":
    main() 