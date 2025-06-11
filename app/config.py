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

# Default font sizes - radically increased for better visibility
DEFAULT_FONT_SIZES = {
    "title": 32,
    "subtitle": 24,
    "axis_title": 22,
    "axis_labels": 20,
    "legend": 18,
    "tick_labels": 20,  # Increased for better readability of axis labels
    "annotations": 16
}

# Font size presets for easy selection
FONT_SIZE_PRESETS = {
    "Small": {
        "title": 18,
        "subtitle": 15,
        "axis_title": 13,
        "axis_labels": 11,
        "legend": 11,
        "tick_labels": 12,  # Increased for better readability
        "annotations": 9
    },
    "Medium": {
        "title": 24,
        "subtitle": 18,
        "axis_title": 16,
        "axis_labels": 14,
        "legend": 14,
        "tick_labels": 16,  # Increased for better readability
        "annotations": 12
    },
    "Large": DEFAULT_FONT_SIZES,
    "Extra Large": {
        "title": 36,
        "subtitle": 28,
        "axis_title": 26,
        "axis_labels": 24,
        "legend": 22,
        "tick_labels": 24,  # Increased for better readability
        "annotations": 20
    },
    "Presentation": {
        "title": 44,
        "subtitle": 36,
        "axis_title": 32,
        "axis_labels": 30,
        "legend": 28,
        "tick_labels": 30,  # Increased for better readability
        "annotations": 26
    }
}

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

    # Use the new radically increased default font sizes
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],  # Fallback fonts
        "font.size": DEFAULT_FONT_SIZES["annotations"],
        "axes.labelsize": DEFAULT_FONT_SIZES["axis_labels"],
        "axes.titlesize": DEFAULT_FONT_SIZES["axis_title"],
        "xtick.labelsize": DEFAULT_FONT_SIZES["tick_labels"],
        "ytick.labelsize": DEFAULT_FONT_SIZES["tick_labels"],
        "legend.fontsize": DEFAULT_FONT_SIZES["legend"],
        "figure.titlesize": DEFAULT_FONT_SIZES["title"],
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