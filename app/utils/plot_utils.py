"""
Plot-related utility functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from app.config import COLORBAR_PLOTS, DEFAULT_FONT_SIZES

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
        fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.35)
    
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

def add_title_and_subtitle(fig, ax, custom_title, custom_subtitle, default_title, show_titles=True, font_sizes=None):
    """
    Add title and subtitle to a plot.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axis
        custom_title: User-provided title or None
        custom_subtitle: User-provided subtitle or None
        default_title: Default title to use if no custom title provided
        show_titles: Whether to show titles at all
        font_sizes: Dictionary with font sizes or None to use defaults
    """
    if not show_titles:
        return
    
    # Use provided font sizes or defaults
    if font_sizes is None:
        font_sizes = DEFAULT_FONT_SIZES
        
    if ax:
        # Use custom title if provided, otherwise use default
        title = custom_title if custom_title else default_title
        ax.set_title(title, fontsize=font_sizes["title"], pad=20)
        
        # Add subtitle if provided
        if custom_subtitle:
            ax.text(0.5, 0.98, custom_subtitle, transform=ax.transAxes, 
                   fontsize=font_sizes["subtitle"], ha='center', va='top', style='italic')
    else:
        # For multi-subplot figures where ax might be None
        if custom_title:
            fig.suptitle(custom_title, fontsize=font_sizes["title"], y=0.98)
        else:
            fig.suptitle(default_title, fontsize=font_sizes["title"], y=0.98)
        
        # Add subtitle if provided
        if custom_subtitle:
            fig.text(0.5, 0.94, custom_subtitle, 
                   ha='center', va='top', fontsize=font_sizes["subtitle"], style='italic')

def format_axis_ticks(ax, font_sizes=None):
    """Format axis ticks for better readability."""
    if not ax:
        return
    
    # Use provided font sizes or defaults
    if font_sizes is None:
        font_sizes = DEFAULT_FONT_SIZES
        
    if ax.get_xticklabels():  # Check if x-tick labels exist
        ax.tick_params(axis='x', rotation=45, labelsize=font_sizes["tick_labels"])
        # Adjust tick label properties directly for better control with rotation
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor") 

def apply_font_sizes_to_axis(ax, font_sizes=None):
    """
    Apply font sizes to various axis elements.
    
    Args:
        ax: Matplotlib axis
        font_sizes: Dictionary with font sizes or None to use defaults
    """
    if not ax or font_sizes is None:
        return
    
    # Apply font sizes to axis labels
    ax.xaxis.label.set_fontsize(font_sizes["axis_title"])
    ax.yaxis.label.set_fontsize(font_sizes["axis_title"])
    
    # Apply font sizes to tick labels
    ax.tick_params(axis='both', which='major', labelsize=font_sizes["tick_labels"])
    
    # Apply font sizes to legend if it exists
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(font_sizes["legend"])
        if legend.get_title():
            legend.get_title().set_fontsize(font_sizes["legend"])

def get_font_sizes_from_params(plot_params):
    """
    Extract font sizes from plot parameters or return defaults.
    
    Args:
        plot_params: Dictionary containing plot parameters
        
    Returns:
        Dictionary with font sizes
    """
    font_sizes = plot_params.get('font_sizes', DEFAULT_FONT_SIZES.copy())
    
    # Ensure all required keys exist
    for key, default_value in DEFAULT_FONT_SIZES.items():
        if key not in font_sizes:
            font_sizes[key] = default_value
    
    return font_sizes

def configure_legend(ax, legend_settings, font_sizes=None):
    """
    Configure legend positioning and appearance.
    
    Args:
        ax: Matplotlib axis
        legend_settings: Dictionary with legend configuration
        font_sizes: Dictionary with font sizes or None to use defaults
    """
    if not ax:
        return
    
    if font_sizes is None:
        font_sizes = DEFAULT_FONT_SIZES
    
    position = legend_settings.get('position', 'best')
    
    # Hide legend if requested
    if position == 'none':
        legend = ax.get_legend()
        if legend:
            legend.remove()
        return
    
    # Get legend parameters
    ncol = legend_settings.get('ncol', 1)
    frameon = legend_settings.get('frameon', True)
    alpha = legend_settings.get('alpha', 0.9)
    
    # Handle special position cases (outside plot area)
    bbox_to_anchor = None
    loc = position
    
    if position == 'bottom':
        bbox_to_anchor = (0.5, -0.15)
        loc = 'upper center'
        if ncol is None:
            ncol = min(len(ax.get_legend_handles_labels()[0]), 4)  # Auto-determine columns
    elif position == 'top':
        bbox_to_anchor = (0.5, 1.15)
        loc = 'lower center'
        if ncol is None:
            ncol = min(len(ax.get_legend_handles_labels()[0]), 4)
    elif position == 'left':
        bbox_to_anchor = (-0.15, 0.5)
        loc = 'center right'
        ncol = 1  # Force single column for left placement
    elif position == 'right':
        bbox_to_anchor = (1.15, 0.5)
        loc = 'center left'
        ncol = 1  # Force single column for right placement
    else:
        # Use built-in matplotlib positions
        if ncol is None:
            ncol = 1  # Default to single column for internal positions
    
    # Create or update legend
    try:
        if bbox_to_anchor:
            legend = ax.legend(
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                ncol=ncol,
                frameon=frameon,
                fontsize=font_sizes["legend"],
                fancybox=True,
                shadow=True if frameon else False
            )
        else:
            legend = ax.legend(
                loc=loc,
                ncol=ncol,
                frameon=frameon,
                fontsize=font_sizes["legend"],
                fancybox=True,
                shadow=True if frameon else False
            )
        
        # Set transparency
        if legend and frameon:
            legend.get_frame().set_alpha(alpha)
            
    except Exception as e:
        print(f"Warning: Could not configure legend: {e}")
        # Fall back to basic legend
        ax.legend(fontsize=font_sizes["legend"])

def get_legend_settings_from_params(plot_params):
    """
    Extract legend settings from plot parameters.
    
    Args:
        plot_params: Dictionary containing plot parameters
        
    Returns:
        Dictionary with legend settings
    """
    ncol = plot_params.get('legend_ncol', 'auto')
    if ncol == 'auto':
        ncol = None
    else:
        try:
            ncol = int(ncol)
        except (ValueError, TypeError):
            ncol = None
    
    return {
        'position': plot_params.get('legend_position', 'best'),
        'ncol': ncol,
        'frameon': plot_params.get('legend_frameon', True),
        'alpha': float(plot_params.get('legend_alpha', 0.9))
    } 