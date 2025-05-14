"""
Data model for experiment results.
"""

import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict

class ExperimentData:
    """Class for managing experiment data."""
    
    def __init__(self):
        """Initialize an empty experiment data store."""
        self.experiment_data = {}  # Stores {exp_name: {mean: pd.Series, std: pd.Series, all_values: {metric: [vals]}}}
        self.available_metrics = []  # Numeric columns for plotting
        self.test_case_data = {}  # Data for per-test case plots
        
    def get_experiment_base_name(self, file_path):
        """Extract the base experiment name from a file path."""
        filename = os.path.basename(file_path)
        # Remove _run_N.csv suffix
        base_name = re.sub(r'_run_\d+\.csv$', '', filename, flags=re.IGNORECASE)
        # Remove .csv if it's not a run file (e.g., experiment_name.csv)
        base_name = base_name.replace('.csv', '')
        return base_name
    
    def load_and_process_files(self, file_paths):
        """
        Load experiment data from CSV files and process it.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (success, error_message)
        """
        if not file_paths:
            return False, "No files selected."
        
        grouped_files = defaultdict(list)
        for f_path in file_paths:
            base_name = self.get_experiment_base_name(f_path)
            grouped_files[base_name].append(f_path)
        
        self.experiment_data.clear()
        self.available_metrics.clear()
        all_numeric_cols = set()
        
        # Store per-test case data for the "Per Test Case" plot
        self.test_case_data = {}
        
        try:
            for exp_name, file_paths in grouped_files.items():
                run_summaries = []  # List of pd.Series, one for each run's summary
                
                for f_path in file_paths:
                    try:
                        df_run = pd.read_csv(f_path)
                        if df_run.empty:
                            print(f"Warning: File {f_path} is empty. Skipping.")
                            continue
                        
                        numeric_cols = df_run.select_dtypes(include=np.number).columns
                        if not numeric_cols.empty:
                            # If multiple rows, average numeric metrics for this run
                            summary_series = df_run[numeric_cols].mean()
                            run_summaries.append(summary_series)
                            all_numeric_cols.update(summary_series.index)
                            
                            # Process per-test case data if "Test Case" column exists
                            if "Test Case" in df_run.columns and any(col.endswith("Score") for col in df_run.columns):
                                # Initialize structure for test case data if needed
                                if exp_name not in self.test_case_data:
                                    self.test_case_data[exp_name] = {}
                                
                                # Extract score columns
                                score_cols = [col for col in df_run.columns if col.endswith("Score") and df_run[col].dtype in [np.float64, np.int64]]
                                
                                for score_col in score_cols:
                                    if score_col not in self.test_case_data[exp_name]:
                                        self.test_case_data[exp_name][score_col] = {}
                                        
                                    for _, row in df_run.iterrows():
                                        if pd.notna(row["Test Case"]) and pd.notna(row[score_col]):
                                            test_case = int(row["Test Case"]) if isinstance(row["Test Case"], (int, float)) else str(row["Test Case"])
                                            
                                            if test_case not in self.test_case_data[exp_name][score_col]:
                                                self.test_case_data[exp_name][score_col][test_case] = []
                                                
                                            self.test_case_data[exp_name][score_col][test_case].append(float(row[score_col]))
                    except Exception as e:
                        print(f"Error reading or processing file {f_path}: {e}")
                        return False, f"Could not process {os.path.basename(f_path)}:\n{e}"
                
                if not run_summaries:
                    print(f"Warning: No valid data found for experiment {exp_name}. Skipping.")
                    continue
                
                # Combine summaries from all runs of this experiment
                exp_df = pd.DataFrame(run_summaries)
                
                mean_stats = exp_df.mean()
                std_stats = exp_df.std()
                
                all_values_for_boxplot = {}
                for col in exp_df.columns:
                    all_values_for_boxplot[col] = exp_df[col].dropna().tolist()
                
                self.experiment_data[exp_name] = {
                    'mean': mean_stats,
                    'std': std_stats,
                    'all_values': all_values_for_boxplot
                }
            
            self.available_metrics = sorted(list(all_numeric_cols))
            
            if not self.experiment_data:
                return False, "No valid data found in the selected files."
                
            return True, f"Processed {len(self.experiment_data)} experiments."
            
        except Exception as e:
            print(f"Processing error: {e}")
            return False, f"An error occurred during data processing: {e}"
    
    def get_experiments(self):
        """Return list of experiment names."""
        return list(self.experiment_data.keys())
    
    def get_metrics(self):
        """Return list of available metrics."""
        return self.available_metrics
    
    def get_experiment_mean(self, exp_name, metric):
        """Get mean value for a specific experiment and metric."""
        if exp_name in self.experiment_data and metric in self.experiment_data[exp_name]['mean']:
            return self.experiment_data[exp_name]['mean'][metric]
        return np.nan
    
    def get_experiment_std(self, exp_name, metric):
        """Get standard deviation for a specific experiment and metric."""
        if exp_name in self.experiment_data and metric in self.experiment_data[exp_name]['std']:
            return self.experiment_data[exp_name]['std'][metric]
        return np.nan
    
    def get_experiment_values(self, exp_name, metric):
        """Get all values for a specific experiment and metric."""
        if exp_name in self.experiment_data and metric in self.experiment_data[exp_name]['all_values']:
            return self.experiment_data[exp_name]['all_values'][metric]
        return []
    
    def get_test_case_data(self, exp_name, metric, test_case=None):
        """
        Get test case data for a specific experiment and metric.
        
        Args:
            exp_name: Name of the experiment
            metric: Metric name
            test_case: Optional specific test case to return
            
        Returns:
            If test_case is provided, returns list of values for that test case.
            Otherwise, returns dict of {test_case: [values]}
        """
        if exp_name not in self.test_case_data or metric not in self.test_case_data[exp_name]:
            return [] if test_case else {}
        
        if test_case:
            return self.test_case_data[exp_name][metric].get(test_case, [])
        
        return self.test_case_data[exp_name][metric]
    
    def has_test_case_data(self):
        """Check if any test case data is available."""
        return bool(self.test_case_data)
    
    def get_all_test_cases(self, selected_metrics, selected_experiments):
        """
        Get all unique test cases across selected metrics and experiments.
        
        Returns:
            List of sorted test cases
        """
        all_test_cases = set()
        
        for exp_name in selected_experiments:
            if exp_name in self.test_case_data:
                for metric in selected_metrics:
                    if metric in self.test_case_data[exp_name]:
                        all_test_cases.update(self.test_case_data[exp_name][metric].keys())
        
        # Sort test cases (numeric if possible)
        try:
            test_cases = sorted([int(tc) for tc in all_test_cases if str(tc).isdigit()])
            test_cases.extend(sorted([tc for tc in all_test_cases if not str(tc).isdigit()]))
        except:
            test_cases = sorted(all_test_cases)
            
        return test_cases 