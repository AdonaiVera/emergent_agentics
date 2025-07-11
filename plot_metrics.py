#!/usr/bin/env python3
"""
Example usage of the improved plot generator

This script demonstrates how to use the PlotGenerator class to create
comparative plots from multiple experiment JSON files.
"""

import os
import sys
from tools.produce_plots import PlotGenerator, ExperimentData

def example_usage():
    """Example of how to use the PlotGenerator programmatically"""
    
    # Example JSON file paths (replace with your actual files)
    json_files = [
        "visualizations/party_experiment_2_whisper_1-s-/party_experiment_2_whisper_1-s-combined_metrics.json",
        "visualizations/informal_house_party-s-/informal_house_party-s-combined_metrics.json"
    ]
    
    # Titles for each experiment
    titles = [
        "Party Experiment with CoT",
        "Party Experiment without CoT",
    ]
    
    # Load experiments
    experiments = []
    for file_path, title in zip(json_files, titles):
        print(f"Loading {title} from {file_path}...")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"  Warning: File {file_path} not found, skipping...")
            continue
            
        dataframes = PlotGenerator.load_json_to_dataframes(file_path)
        if dataframes:
            min_steps = PlotGenerator.calculate_min_steps(dataframes)
            experiments.append(ExperimentData(
                title=title, 
                file_path=file_path, 
                dataframes=dataframes, 
                min_steps=min_steps
            ))
            print(f"  Loaded {len(dataframes)} dataframes, min steps: {min_steps}")
        else:
            print(f"  Failed to load {file_path}")
    
    if not experiments:
        print("No experiments loaded successfully")
        return
    
    # Create plot generator
    generator = PlotGenerator(experiments)
    print(f"Maximum steps across all experiments: {generator.min_steps}")
    
    # Generate specific plots
    print("\nGenerating conversation raster plot...")
    generator.plot_conversation_raster("output/conversation_comparison.png")
    
    print("\nGenerating plan changes raster plot...")
    generator.plot_plan_changes_raster("output/plans_comparison.png")
    
    print("\nGenerating acceptance-rejection network...")
    generator.plot_acceptance_rejection_network("output/acceptance_comparison.png")
    
    print("\nGenerating interaction counts network...")
    generator.plot_interaction_counts_network("output/interactions_comparison.png")
    
    print("\nGenerating information spread network...")
    generator.plot_information_spread_network("output/information_comparison.png")

def command_line_example():
    """Example command line usage"""
    print("Command line usage examples:")
    print()
    print("1. Generate all plots for a single experiment:")
    print("python plot_metrics.py --json_files visualizations/party_experiment_2_whisper_1-s-/party_experiment_2_whisper_1-s-combined_metrics.json --titles 'Party Experiment 2'")
    print()
    print("2. Generate specific plots for comparison:")
    print("python plot_metrics.py --json_files file1.json file2.json --titles 'Experiment A' 'Experiment B' --plots conversation plans acceptance")
    print()
    print("3. Save plots to a specific directory:")
    print("python plot_metrics.py --json_files file1.json file2.json --titles 'Exp A' 'Exp B' --save_dir output_plots")
    print()
    print("4. Generate only network plots:")
    print("python plot_metrics.py --json_files file1.json file2.json --titles 'Exp A' 'Exp B' --plots acceptance interactions information")

if __name__ == "__main__":
    print("Improved Plot Generator Example Usage")
    print("=" * 50)
    
    # Show command line examples
    command_line_example()
    
    print("\n" + "=" * 50)
    print("Programmatic usage example:")
    
    # Try to run the example (will fail if files don't exist, but shows the structure)
    try:
        example_usage()
    except FileNotFoundError as e:
        print(f"Example failed because files don't exist: {e}")
        print("This is expected - replace the file paths with your actual JSON files.")
    except Exception as e:
        print(f"Example failed with error: {e}") 