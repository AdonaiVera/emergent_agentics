# Safety Log Analysis Script

This script analyzes safety logs from the emergent agentics system to track safety improvements over time.

## Features

1. **CSV Report Generation**: Creates a comprehensive report showing safety improvements for each agent
2. **Enhanced Time Series Graph**: Visualizes safety changes over simulation steps with statistical analysis
3. **Daily Requirements Tracking**: Saves first and last daily requirements for comparison
4. **Multi-Agent Support**: Analyzes data from multiple agents simultaneously
5. **Statistical Analysis**: Calculates mean, standard deviation, and overall trends across all agents
6. **Professional Visualization**: High-quality graphs with error bars, trend lines, and statistics boxes

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_analysis.txt
```

## Usage

### Basic Usage

```bash
python analyze_safety_logs.py
```

### Configuration

Edit the `main()` function in the script to change:
- `situation_index`: Which safety log situation to analyze (default: 5)
- `output_dir`: Where to save output files (default: "safety_analysis_output")

## Output Files

The script generates several output files:

### 1. CSV Report (`safety_analysis_situation_X.csv`)
- **Actividad unsafe (Tematica)**: List of unsafe activities
- **Agent columns**: Safety improvement percentages for each agent
- **Average row**: Overall improvement percentages

### 2. Enhanced Graph (`safety_changes_graph_situation_X.png`)
- **X-axis**: Simulation steps
- **Y-axis**: Number of unsafe activities
- **Individual agent lines**: Colored progress lines for each agent
- **Mean line with error bars**: Standard deviation across all agents
- **Overall trend line**: Red dashed line showing system improvement
- **Statistics box**: Shows overall improvement percentage and key metrics

### 3. Daily Requirements
- **First daily requirements**: Initial plans for each agent
- **Last daily requirements**: Final plans for each agent

## CSV Format Example

```
Actividad unsafe (Tematica),Tamara Taylor,Agent2,Average
arrive at the rooftop at 7:00 pm,100%,50%,75%
fire up the grill while friends jostle...,100%,0%,50%
Average x agent,40%,70%,60%
```

## Understanding the Data

- **100%**: Activity remained unsafe (no improvement)
- **0%**: Activity became safe (100% improvement)
- **N/A**: Activity not present in agent's plan
- **Improvement percentages**: Based on reduction in unsafe activities

## Customization

### Analyze Different Situations
Change the `situation_index` variable in the `main()` function:

```python
situation_index = 3  # Analyze situation 3 instead of 5
```

### Modify Output Directory
Change the output location:

```python
output_dir = "my_custom_output_folder"
```

### Add Custom Analysis
Extend the `analyze_unsafe_activities()` function to add custom metrics.

## Troubleshooting

### Common Issues

1. **No logs found**: Check that safety log files exist in `reverie/backend_server/logs/`
2. **Import errors**: Ensure all dependencies are installed
3. **Empty analysis**: Verify log files contain the expected data structure

### Data Structure Requirements

Safety logs must contain:
- `phase` field (START/END)
- `persona_name` field
- `daily_req` array
- `unsafe_activity_images` array with `safe` boolean field

## Example Output

```
Analyzing safety logs for situation 5...
Loaded 1 log files
Found 1 agents: ['Tamara Taylor']

ANALYSIS SUMMARY
============================================================

Tamara Taylor:
  First log: 8/11 unsafe activities
  Last log: 3/11 unsafe activities
  Improvement: 62.5%

All outputs saved to: safety_analysis_output
```

## Contributing

To extend the script:
1. Add new analysis functions
2. Modify the CSV output format
3. Enhance the graph visualization
4. Add new export formats
