"""
Plot-related utility functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from app.config import COLORBAR_PLOTS

def get_plot_color_palette(num_items, palette_name="colorblind"):
    """
    Get a color palette with the specified number of colors.
    
    Args:
        num_items: Number of colors needed
        palette_name: Name of the color palette to use
        
    Returns:
        List of colors
    """
    try:
        colors = sns.color_palette(palette_name, max(num_items, 3))
    except:
        colors = sns.color_palette("deep", max(num_items, 3))
    return colors

def create_figure(plot_type, width, height):
    """
    Create a matplotlib figure with the appropriate settings for the plot type.
    
    Args:
        plot_type: Type of plot
        width: Width in inches
        height: Height in inches
        
    Returns:
        Matplotlib figure
    """
    # Check if the plot type uses colorbars to determine the appropriate layout engine
    use_constrained_layout = plot_type not in COLORBAR_PLOTS
    
    # Create new figure with specified size and appropriate layout settings
    fig = plt.figure(
        figsize=(width, height),
        constrained_layout=use_constrained_layout
    )
    
    # Only set manual adjustments if not using constrained layout
    if not use_constrained_layout:
        fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
    
    return fig

def apply_y_axis_range(ax, use_custom_range, y_min=None, y_max=None):
    """
    Apply custom Y-axis range to an axis if requested.
    
    Args:
        ax: Matplotlib axis
        use_custom_range: Whether to use custom range
        y_min: Minimum value or None
        y_max: Maximum value or None
    """
    if not use_custom_range:
        return
        
    try:
        y_min = float(y_min) if y_min else None
        y_max = float(y_max) if y_max else None
        
        if y_min is not None and y_max is not None:
            if y_min >= y_max:
                print("Warning: Y-axis minimum must be less than maximum. Using auto-scaling.")
            else:
                ax.set_ylim(y_min, y_max)
        elif y_min is not None:
            ax.set_ylim(bottom=y_min)
        elif y_max is not None:
            ax.set_ylim(top=y_max)
    except ValueError:
        # If conversion fails, just use auto-scaling
        pass

def add_title_and_subtitle(fig, ax, custom_title, custom_subtitle, default_title):
    """
    Add title and subtitle to a plot.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axis
        custom_title: User-provided title or None
        custom_subtitle: User-provided subtitle or None
        default_title: Default title to use if no custom title provided
    """
    if ax:
        # Use custom title if provided, otherwise use default
        title = custom_title if custom_title else default_title
        ax.set_title(title, fontsize=14, pad=15)
        
        # Add subtitle if provided
        if custom_subtitle:
            ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                   fontsize=11, ha='center', va='top', style='italic')
    else:
        # For multi-subplot figures where ax might be None
        if custom_title:
            fig.suptitle(custom_title, fontsize=14, y=0.98)
        else:
            fig.suptitle(default_title, fontsize=14, y=0.98)
        
        # Add subtitle if provided
        if custom_subtitle:
            fig.text(0.5, 0.94, custom_subtitle, 
                   ha='center', va='top', fontsize=11, style='italic')

def format_axis_ticks(ax):
    """Format axis ticks for better readability."""
    if not ax:
        return
        
    if ax.get_xticklabels():  # Check if x-tick labels exist
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        # Adjust tick label properties directly for better control with rotation
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor") 