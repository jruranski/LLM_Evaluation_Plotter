"""
UI-related utility functions.
"""

import tkinter as tk
from tkinter import ttk

def create_loading_window(master, message="Loading..."):
    """
    Create a loading window for operations that might take time.
    
    Args:
        master: Parent window
        message: Text to display
        
    Returns:
        The loading window (tkinter.Toplevel)
    """
    loading_win = tk.Toplevel(master)
    loading_win.transient(master)  # Show above master
    loading_win.grab_set()  # Modal
    loading_win.title("")
    loading_win.resizable(False, False)
    
    ttk.Label(loading_win, text=message, font=("Arial", 11)).pack(padx=30, pady=20)
    
    # Center it
    master.update_idletasks()  # Ensure master window dimensions are current
    master_x = master.winfo_x()
    master_y = master.winfo_y()
    master_w = master.winfo_width()
    master_h = master.winfo_height()
    win_w = loading_win.winfo_reqwidth() + 60  # Add padding
    win_h = loading_win.winfo_reqheight() + 40
    x_pos = master_x + (master_w // 2) - (win_w // 2)
    y_pos = master_y + (master_h // 2) - (win_h // 2)
    loading_win.geometry(f'{win_w}x{win_h}+{x_pos}+{y_pos}')
    return loading_win

def validate_float(value):
    """
    Validate if input string is a valid float or empty string.
    
    Args:
        value: String to validate
        
    Returns:
        True if valid, False otherwise
    """
    if value == "" or value == "-":
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False 