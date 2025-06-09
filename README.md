# RAG Evaluation Plotter

A visualization tool for analyzing and comparing Retrieval-Augmented Generation (RAG) evaluation results from Deepeval across multiple experiments and metrics.



## Features

- **Multiple Experiment Comparison**: Load and visualize results from multiple RAG evaluation runs simultaneously
- **Flexible Plot Types**: Choose from bar plots, grouped bar plots, line plots, and box plots
- **Statistical Analysis**: Automatically calculates and displays means, standard deviations, and outliers
- **Y-Axis Zooming**: Focus on specific value ranges for better visualization of closely grouped results
- **Customizable Dimensions**: Adjust plot sizes with presets or custom dimensions
- **Title and Label Customization**: Rename metrics and experiments for clearer visualization
- **Box Plot Outlier Control**: Show or hide outliers in box plots
- **Comprehensive Font Controls**: Adjust font sizes for all plot elements with presets and individual controls
- **Legend Positioning**: Full control over legend placement with 15+ position options, column layout, transparency, and frame settings
- **Configuration Save/Load**: Save complete plot configurations including files, settings, and customizations for easy reproduction
- **High-Quality Export**: Save plots as PNG, PDF, or SVG for publications or presentations
## Architecture

The application has been refactored into a modular architecture following Model-View-Controller (MVC) pattern:

```
LLM_Evaluation_Plotter/
├── main.py                    # Entry point
├── app/
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Configuration and constants
│   ├── controllers/           # Controllers for application logic
│   │   ├── __init__.py
│   │   ├── file_controller.py  # Handles file operations
│   │   └── plot_controller.py  # Manages plot generation
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   └── experiment_data.py  # Manages experiment data
│   ├── ui/                    # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py      # Main application window
│   │   ├── control_panel.py    # Control panel UI
│   │   └── plot_area.py        # Plot display area
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── ui_utils.py         # UI helper functions
│       └── plot_utils.py       # Plot helper functions
```
### Components

- **Models**: Responsible for data handling and business logic
  - `ExperimentData`: Manages loading, processing, and storing experiment data

- **Controllers**: Handle application logic and connect models with views
  - `FileController`: Manages file operations (browse, load, save)
  - `PlotController`: Manages plot generation and configuration

- **UI**: User interface components
  - `MainWindow`: The main application window
  - `ControlPanel`: Control panel for user input
  - `PlotArea`: Area for displaying plots

- **Utils**: Helper functions and utilities
  - `ui_utils.py`: UI-related helper functions
  - `plot_utils.py`: Plot-related helper functions

- **Config**: Application configuration
  - `config.py`: Contains constants and default settings
## Installation

### Prerequisites

- Python 3.7+
- Required Python packages: 
  - tkinter
  - pandas
  - numpy
  - matplotlib
  - seaborn

### Setup

#### Recommended: Using Conda (Recommended)

This is the recommended installation method as it ensures all dependencies are properly managed:

1. Clone this repository:
   ```bash
   git clone https://github.com/jruranski/rag-evaluation-plotter.git
   cd rag-evaluation-plotter
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate rag-evaluation-plotter
   ```

3. Run the application:
   ```bash
   python main.py
   ```

#### Alternative: Using Pip

1. Clone this repository:
   ```bash
   git clone https://github.com/jruranski/rag-evaluation-plotter.git
   cd rag-evaluation-plotter
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
   Note: Tkinter is usually included with Python installations.

## Usage

### Starting the Application

Run the application using:
```bash
python evaluation_plotter.py
```

If using conda, make sure you've activated the environment first:
```bash
conda activate rag-evaluation-plotter
python evaluation_plotter.py
```

### Working with the Application

#### 1. Loading Evaluation Data

- Click **Browse Files** to select your CSV evaluation files
- Select one or multiple files
- Click **Load & Process Selected Files** to import the data

The tool automatically groups multiple runs of the same experiment, calculating means and standard deviations across runs.

#### 2. Saving and Loading Configurations

The application provides a comprehensive configuration system that saves all your plot settings for easy reuse:

**Saving a Configuration:**
- Set up your plot exactly as desired (files, metrics, plot type, font sizes, customizations, etc.)
- Click **Save Configuration** 
- Choose a location and filename (`.plotconfig` extension will be added automatically)
- The configuration file contains:
  - All selected files and their paths
  - All control panel settings (plot type, font sizes, Y-axis ranges, plot dimensions)
  - Selected metrics and label customizations
  - Plot titles and LaTeX settings
  - All UI customizations

**Loading a Configuration:**
- Click **Load Configuration**
- Select a previously saved `.plotconfig` file
- The application will:
  - Attempt to load all files from their saved paths
  - Restore all plot settings exactly as they were saved
  - Handle missing files gracefully (with user confirmation)
  - Validate and fix any invalid settings

**Configuration Features:**
- **File Path Validation**: Warns about missing files and offers to continue with available files
- **Version Compatibility**: Handles configuration files from different app versions
- **Error Recovery**: Provides sensible defaults for any missing or invalid settings
- **Timestamped Names**: Automatically suggests timestamped filenames for easy organization
- **Complete State Preservation**: Saves every aspect of your plot setup

This feature is perfect for:
- Reproducing plots with identical settings
- Sharing plot configurations with colleagues
- Maintaining consistent formatting across multiple plots
- Batch processing different datasets with the same visualization settings

#### 3. Compatible CSV Format

Your CSV files should contain:
- One or more rows per experiment run
- Numeric columns for metrics (e.g., precision, recall, F1, etc.)
- Multiple runs of the same experiment can be named with a `_run_N` suffix (e.g., `experiment_1_run_1.csv`, `experiment_1_run_2.csv`)

Example CSV content:
```
precision,recall,f1_score,latency_ms
0.92,0.87,0.894,150.2
```

#### 4. Creating Plots

1. Select metrics to compare
2. Choose plot type (Bar Plot, Grouped Bar Plot, Line Plot, or Box Plot)
3. Customize Y-axis range for "zooming" into specific value ranges
4. Set custom plot dimensions or use one of the presets
5. Add custom titles and subtitles
6. Click **Generate Plot**

#### 5. Saving Plots

After generating a plot, click **Save Plot** to export it as PNG, PDF, or SVG format.

## Example Workflows

### Comparing Metrics Across Models

1. Load evaluation results from multiple models/approaches
2. Select the key metric (e.g., F1 score)
3. Use a Bar Plot to visualize the comparison
4. Enable Y-axis zooming if scores are closely grouped

### Analyzing Metric Distribution

1. Load multiple runs of the same experiment
2. Select a metric of interest
3. Use a Box Plot to visualize the distribution and identify outliers

### Comparing Multiple Metrics

1. Load evaluation results
2. Select multiple metrics (e.g., precision, recall, F1)
3. Use a Grouped Bar Plot to compare metrics side by side

## Advanced Features

### Customizing Labels

- Rename metrics to display more readable names
- Rename experiment labels to make plots more understandable

### Y-Axis Zooming

When evaluation scores are closely grouped (e.g., all around 0.9), use the custom Y-axis range to "zoom in" on the relevant range (e.g., 0.85-0.95) to better visualize differences.

### Plot Size Adjustment

- Use preset sizes for standard visualization needs
- Set custom dimensions for specific publication or presentation requirements

### Legend Positioning

The application provides comprehensive legend control options:

**Position Options:**
- **Automatic**: "Best" position chosen by matplotlib
- **Standard Positions**: Upper/lower left/right, center positions
- **Outside Positions**: Place legends outside the plot area (top, bottom, left, right)
- **Hide Legend**: Completely remove legends when not needed

**Advanced Settings:**
- **Column Layout**: Control number of legend columns (auto-detect or specify 1-5)
- **Frame Control**: Show/hide legend frame with customizable transparency
- **Transparency**: Adjust legend background opacity (50%-100%)

**Usage Tips:**
- Use "Bottom (outside)" for plots with many legend items
- Use "Right (outside)" for plots with long experiment names
- Adjust transparency to prevent legends from obscuring data points
- Auto-column detection works well for most cases, but manual control is available for fine-tuning

## License

[MIT License](LICENSE)

## Design Patterns

This implementation uses several design patterns:

1. **MVC (Model-View-Controller)**: Separates data, presentation, and application logic.
2. **Observer Pattern**: Components communicate through events/callbacks.
3. **Factory Method**: Different plot types are created through specialized methods.
4. **Facade Pattern**: Controllers provide a simplified interface to complex systems.

## Future Enhancements

This modular architecture makes it easy to implement future enhancements:

1. Adding new plot types
2. Supporting additional data formats
3. Implementing data preprocessing options
4. Adding statistical analysis features
5. Creating plugin architecture for custom visualizations 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 