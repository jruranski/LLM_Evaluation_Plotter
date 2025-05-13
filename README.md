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
- **High-Quality Export**: Save plots as PNG, PDF, or SVG for publications or presentations

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
   python evaluation_plotter.py
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

#### 2. Compatible CSV Format

Your CSV files should contain:
- One or more rows per experiment run
- Numeric columns for metrics (e.g., precision, recall, F1, etc.)
- Multiple runs of the same experiment can be named with a `_run_N` suffix (e.g., `experiment_1_run_1.csv`, `experiment_1_run_2.csv`)

Example CSV content:
```
precision,recall,f1_score,latency_ms
0.92,0.87,0.894,150.2
```

#### 3. Creating Plots

1. Select metrics to compare
2. Choose plot type (Bar Plot, Grouped Bar Plot, Line Plot, or Box Plot)
3. Customize Y-axis range for "zooming" into specific value ranges
4. Set custom plot dimensions or use one of the presets
5. Add custom titles and subtitles
6. Click **Generate Plot**

#### 4. Saving Plots

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

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 