# Improved Plot Generator

This enhanced version of the plot generator supports multiple JSON files, conditional plotting, and fair comparisons between experiments.

## Key Improvements

### 1. Multiple JSON File Support
- Load and compare multiple experiment results simultaneously
- Each experiment gets a descriptive title for easy identification
- Automatic detection of minimum steps across all experiments for fair comparison

### 2. Conditional Plotting
- Generate only specific plot types using command-line arguments
- Choose from: `conversation`, `plans`, `acceptance`, `interactions`, `information`, or `all`
- Programmatic control over which plots to generate

### 3. Fair Comparisons
- All plots use the same time scale (minimum steps across all experiments)
- Side-by-side comparisons for easy visual analysis
- Consistent color schemes and formatting across experiments

### 4. Better Organization
- Object-oriented design with `PlotGenerator` and `ExperimentData` classes
- Clean separation of data loading, processing, and visualization
- Error handling for missing or invalid files

## Usage

### Command Line Interface

```bash
# Basic usage with a single experiment
python produce_plots.py \
    --json_files visualizations/experiment1/combined_metrics.json \
    --titles "Experiment 1"

# Compare multiple experiments
python produce_plots.py \
    --json_files file1.json file2.json file3.json \
    --titles "Control" "Treatment A" "Treatment B"

# Generate specific plots only
python produce_plots.py \
    --json_files file1.json file2.json \
    --titles "Exp A" "Exp B" \
    --plots conversation plans acceptance

# Save plots to a directory
python produce_plots.py \
    --json_files file1.json file2.json \
    --titles "Exp A" "Exp B" \
    --save_dir output_plots
```

### Programmatic Usage

```python
from produce_plots import PlotGenerator, ExperimentData

# Load experiments
experiments = []
for file_path, title in zip(json_files, titles):
    dataframes = PlotGenerator.load_json_to_dataframes(file_path)
    if dataframes:
        min_steps = PlotGenerator.calculate_min_steps(dataframes)
        experiments.append(ExperimentData(
            title=title, 
            file_path=file_path, 
            dataframes=dataframes, 
            min_steps=min_steps
        ))

# Create generator and plot
generator = PlotGenerator(experiments)
generator.plot_conversation_raster("output/conversation.png")
generator.plot_plan_changes_raster("output/plans.png")
```

## Available Plot Types

1. **Conversation Raster Plot** (`conversation`)
   - Shows when each agent participates in conversations
   - Color-coded by location
   - Time-aligned across experiments

2. **Plan Changes Raster Plot** (`plans`)
   - Displays when agents change their plans
   - Each agent gets a unique color
   - Easy to spot coordination patterns

3. **Acceptance-Rejection Network** (`acceptance`)
   - Network graph showing interaction acceptance ratios
   - Edge colors indicate acceptance rate (red=rejected, green=accepted)
   - Side-by-side comparison of different experiments

4. **Interaction Counts Network** (`interactions`)
   - Network graph showing number of interactions between agents
   - Edge thickness and color indicate interaction frequency
   - Useful for identifying social patterns

5. **Information Spread Network** (`information`)
   - Directed graph showing how information propagates
   - Nodes colored by message type
   - Edge colors indicate timing of spread

## File Structure

```
tools/
├── produce_plots.py      # Main plot generator
├── example_usage.py      # Usage examples
└── README_plots.md       # This documentation
```

## Requirements

- Python 3.7+
- pandas
- matplotlib
- seaborn
- networkx
- argparse

## Error Handling

The improved version includes robust error handling:
- Missing files are reported but don't crash the program
- Invalid JSON files are skipped with informative messages
- Empty datasets are handled gracefully
- File permission issues are caught and reported

## Output

- Plots are displayed interactively by default
- Optional saving to PNG files with high resolution (300 DPI)
- Automatic directory creation for saved plots
- Consistent naming convention for saved files

## Example Output Structure

When comparing multiple experiments, the plots will show:
- Side-by-side subplots for each experiment
- Consistent time scales across all experiments
- Clear titles identifying each experiment
- Shared colorbars and legends where appropriate

This makes it easy to spot differences in agent behavior, coordination patterns, and information flow between different experimental conditions. 