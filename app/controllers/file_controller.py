"""
Controller for file operations.
"""

import os
from tkinter import filedialog, messagebox

class FileController:
    """Controller for file operations."""
    
    def __init__(self, model):
        """
        Initialize the file controller.
        
        Args:
            model: ExperimentData model instance
        """
        self.model = model
        self.selected_files = []
    
    def browse_files(self):
        """
        Open a file dialog for selecting CSV files.
        
        Returns:
            List of selected file paths or empty list if cancelled
        """
        files = filedialog.askopenfilenames(
            title="Select CSV Evaluation Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if files:
            self.selected_files = list(files)
        return self.selected_files
    
    def get_selected_filenames(self):
        """
        Get basenames of selected files.
        
        Returns:
            List of filenames without paths
        """
        return [os.path.basename(f) for f in self.selected_files]
    
    def load_and_process_files(self):
        """
        Load and process the selected files using the model.
        
        Returns:
            Tuple of (success, message)
        """
        success, message = self.model.load_and_process_files(self.selected_files)
        
        if success:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", message)
            
        return success
    
    def save_plot(self, figure):
        """
        Save the plot to a file.
        
        Args:
            figure: Matplotlib figure to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not figure:
            messagebox.showerror("Error", "No plot generated to save.")
            return False

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")],
            title="Save Plot As"
        )
        if not filepath:
            return False

        try:
            # Use bbox_inches='tight' for better layout in saved file
            figure.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to {filepath}")
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save plot: {e}")
            return False 