"""
Configuration settings for the LLM Evaluation Plotter.
"""

import matplotlib.pyplot as plt

# Default window size
DEFAULT_WINDOW_SIZE = "1300x850"

# Default plot sizes
DEFAULT_PLOT_WIDTH = 10
DEFAULT_PLOT_HEIGHT = 6

# Size presets
SIZE_PRESETS = {
    "Standard (10×6)": (10, 6),
    "Wide (16×9)": (16, 9),
    "Square (8×8)": (8, 8),
    "Poster (12×15)": (12, 15),
    "Small (6×4)": (6, 4)
}

# Plot types
PLOT_TYPES = [
    "Bar Plot", 
    "Grouped Bar Plot", 
    "Line Plot", 
    "Box Plot", 
    "Per Test Case", 
    "Per Test Case Line Plot", 
    "Heatmap Per Model", 
    "Correlation Matrix", 
    "Radar Chart", 
    "Violin Plot", 
    "Statistical Significance"
]

# The plot types that use colorbars, which conflict with constrained_layout
COLORBAR_PLOTS = [
    "Heatmap Per Model", 
    "Correlation Matrix", 
    "Violin Plot", 
    "Radar Chart", 
    "Statistical Significance"
]

# Setup matplotlib defaults
def configure_matplotlib():
    """Configure matplotlib defaults for this application."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('ggplot')
        except OSError:
            print("Warning: Neither 'seaborn-v0_8-whitegrid' nor 'ggplot' styles found. Using Matplotlib default.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],  # Fallback fonts
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 100,  # Default for display
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.formatter.use_mathtext": True,  # For math symbols
        "figure.constrained_layout.use": True,  # Better layout management
        "figure.figsize": [DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT],  # Default figure size
        "figure.autolayout": False,  # Let us control the layout manually
        "axes.titlelocation": "center",
        "axes.titlepad": 12,
        "axes.labelpad": 8,
    }) 