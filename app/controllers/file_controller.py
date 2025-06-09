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
        Open a file dialog for selecting CSV files and add them to the current selection.
        
        Returns:
            List of selected file paths
        """
        files = filedialog.askopenfilenames(
            title="Select CSV Evaluation Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if files:
            # Add only files that aren't already in the list
            for file in files:
                if file not in self.selected_files:
                    self.selected_files.append(file)
        return self.selected_files
    
    def clear_files(self):
        """
        Clear all selected files.
        
        Returns:
            Empty list
        """
        self.selected_files = []
        return self.selected_files
    
    def remove_file(self, file_path):
        """
        Remove a specific file from the selection.
        
        Args:
            file_path: Path of the file to remove
            
        Returns:
            Updated list of selected files
        """
        if file_path in self.selected_files:
            self.selected_files.remove(file_path)
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
    
    def save_latex_table(self, latex_code):
        """
        Save LaTeX table code to a file.
        
        Args:
            latex_code: The LaTeX table code to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not latex_code:
            messagebox.showerror("Error", "No LaTeX table code generated to save.")
            return False

        filepath = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Save LaTeX Table As"
        )
        if not filepath:
            return False

        try:
            with open(filepath, 'w') as f:
                f.write(latex_code)
            messagebox.showinfo("Success", f"LaTeX table saved to {filepath}")
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save LaTeX table: {e}")
            return False 