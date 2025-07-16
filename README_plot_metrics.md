# Improved Plot Metrics with Research Config Support

This enhanced version of the plot metrics system integrates with the research configuration file to automatically load experiment data and generate comparative visualizations.

## Key Features

### 1. Research Config Integration
- Automatically loads experiment configurations from `research_config.py`
- Supports filtering by environment, whisper mode, and whisper count
- Eliminates the need to manually specify JSON file paths and titles

### 2. Flexible Filtering
- Filter experiments by environment (e.g., "Karaoke Night", "Medical Waiting Room")
- Filter by whisper mode (e.g., "1 Agent", "2 Agents", "5 Agents")
- Filter by whisper count (e.g., 1, 2, 5)
- Combine multiple filters for precise experiment selection

### 3. Enhanced Command Line Interface
- Support for both research config and direct JSON file approaches
- Comprehensive filtering options
- Flexible plot type selection
- Custom output directory support

## Installation

No additional installation required. The system uses existing dependencies:
- Python 3.7+
- pandas
- matplotlib
- seaborn
- networkx

## Usage

### Command Line Interface

#### Using Research Config (Recommended)

```bash
# Generate all plots for all experiments in research config
python plot_metrics.py --config research_config.py

# Generate plots for specific environments
python plot_metrics.py --config research_config.py --environments "Karaoke Night" "Medical Waiting Room"

# Generate plots for specific whisper modes
python plot_metrics.py --config research_config.py --whisper_modes "1 Agent" "2 Agents"

# Generate plots for specific whisper counts
python plot_metrics.py --config research_config.py --whisper_counts 1 2

# Generate specific plot types only
python plot_metrics.py --config research_config.py --plots conversation plans acceptance

# Save plots to a specific directory
python plot_metrics.py --config research_config.py --save_dir output_plots

# Skip file existence check (useful for testing)
python plot_metrics.py --config research_config.py --no_file_check
```

#### Using Direct JSON Files (Legacy)

```bash
# Generate all plots for specific JSON files
python plot_metrics.py --json_files file1.json file2.json --titles "Exp A" "Exp B"

# Generate specific plots
python plot_metrics.py --json_files file1.json file2.json --titles "Exp A" "Exp B" --plots conversation plans
```

### Programmatic Usage

```python
from plot_metrics import load_research_configurations, filter_configurations, load_experiments_from_config
from tools.produce_plots import PlotGenerator

# Load all configurations
configurations = load_research_configurations()

# Filter for specific environments
karaoke_configs = filter_configurations(configurations, environments=["Karaoke Night"])

# Load experiments
experiments = load_experiments_from_config(karaoke_configs)

# Create plot generator
generator = PlotGenerator(experiments)

# Generate plots
generator.plot_conversation_raster("output/conversation.png")
generator.plot_plan_changes_raster("output/plans.png")
```

## Research Config File Format

The system expects a `research_config.py` file with the following structure:

```python
RESEARCH_CONFIGURATIONS = [
    {
        "environment": "Karaoke Night",
        "whisper_mode": "1 Agent", 
        "whisper_count": 1,
        "title": "Karaoke Night - 1 Whisper Agent",
        "file_path": "visualizations/experiment1/combined_metrics.json"
    },
    {
        "environment": "Medical Waiting Room",
        "whisper_mode": "2 Agents",
        "whisper_count": 2,
        "title": "Medical Waiting Room - 2 Whisper Agents",
        "file_path": "visualizations/experiment2/combined_metrics.json"
    }
]
```

### Configuration Fields

- **environment**: The simulation environment (e.g., "Karaoke Night", "Medical Waiting Room")
- **whisper_mode**: Description of the whisper configuration (e.g., "1 Agent", "2 Agents")
- **whisper_count**: Numeric count of whisper agents
- **title**: Display title for the experiment in plots
- **file_path**: Path to the JSON metrics file

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

## Testing

Run the test script to verify the functionality:

```bash
python test_plot_metrics.py
```

This will test:
- Research config loading
- Configuration filtering
- Experiment loading
- Plot generator creation

## Examples

### Example 1: Compare All Karaoke Night Experiments

```bash
python plot_metrics.py --config research_config.py --environments "Karaoke Night" --save_dir karaoke_analysis
```

### Example 2: Compare 1 vs 2 Agent Configurations

```bash
python plot_metrics.py --config research_config.py --whisper_counts 1 2 --plots conversation plans acceptance
```

### Example 3: Generate Only Network Plots for Medical Environment

```bash
python plot_metrics.py --config research_config.py --environments "Medical Waiting Room" --plots acceptance interactions information
```

## Error Handling

The system includes robust error handling:
- Missing files are reported but don't crash the program
- Invalid JSON files are skipped with warnings
- Configuration errors are clearly reported
- Graceful handling of missing data

## File Structure

```
├── plot_metrics.py              # Main script with research config support
├── tools/
│   └── produce_plots.py         # Enhanced plot generator
├── research_config.py           # Research configuration file
├── test_plot_metrics.py         # Test script
└── README_plot_metrics.md       # This documentation
```

## Migration from Old System

If you were using the old system with manual JSON file specification:

**Old way:**
```bash
python plot_metrics.py --json_files file1.json file2.json --titles "Exp A" "Exp B"
```

**New way:**
```bash
python plot_metrics.py --config research_config.py
```

The new system is backward compatible - you can still use the old method if needed.

## Troubleshooting

### Common Issues

1. **"No configurations found"**
   - Check that `research_config.py` exists and contains `RESEARCH_CONFIGURATIONS`
   - Verify the file path in the `--config` argument

2. **"No experiments loaded successfully"**
   - Check that JSON files exist at the paths specified in the config
   - Use `--no_file_check` to skip file existence validation

3. **"No configurations match the specified filters"**
   - Verify filter values match exactly (case-sensitive)
   - Check available environments, whisper modes, and counts in your config

### Debug Mode

For detailed debugging, you can run the test script:

```bash
python test_plot_metrics.py
```

This will show detailed information about configuration loading and filtering. 