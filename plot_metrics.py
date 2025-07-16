#!/usr/bin/env python3
"""
Improved plot generator that uses research config file

This script demonstrates how to use the PlotGenerator class to create
comparative plots from multiple experiment JSON files defined in research_config.py.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional
from tools.produce_plots import PlotGenerator, ExperimentData

def load_research_configurations(config_file: str = "research_config.py") -> List[Dict]:
    """Load research configurations from the config file"""
    try:
        # Import the research configurations
        import importlib.util
        spec = importlib.util.spec_from_file_location("research_config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        return config_module.RESEARCH_CONFIGURATIONS
    except Exception as e:
        print(f"Error loading research config file {config_file}: {e}")
        return []

def filter_configurations(configurations: List[Dict], 
                         environments: Optional[List[str]] = None,
                         whisper_modes: Optional[List[str]] = None,
                         whisper_counts: Optional[List[int]] = None) -> List[Dict]:
    """Filter configurations based on specified criteria"""
    filtered = []
    
    for config in configurations:
        # Filter by environment
        if environments and config.get('environment') not in environments:
            continue
            
        # Filter by whisper mode
        if whisper_modes and config.get('whisper_mode') not in whisper_modes:
            continue
            
        # Filter by whisper count
        if whisper_counts and config.get('whisper_count') not in whisper_counts:
            continue
            
        filtered.append(config)
    
    return filtered

def load_experiments_from_config(configurations: List[Dict], 
                                check_files: bool = True) -> List[ExperimentData]:
    """Load experiments from research configurations"""
    experiments = []
    
    for config in configurations:
        title = config.get('title', 'Unknown Experiment')
        file_path = config.get('file_path', '')
        
        print(f"Loading {title} from {file_path}...")
        
        # Check if file exists
        if check_files and not os.path.exists(file_path):
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
    
    return experiments

def example_usage():
    """Example of how to use the PlotGenerator with research config"""
    
    # Load all configurations from research config
    configurations = load_research_configurations()
    
    if not configurations:
        print("No configurations found in research_config.py")
        return
    
    print(f"Found {len(configurations)} configurations in research config")
    
    # Example: Filter for specific environments
    karaoke_configs = filter_configurations(configurations, environments=["Karaoke Night"])
    medical_configs = filter_configurations(configurations, environments=["Medical Waiting Room"])
    
    # Load experiments for comparison
    all_experiments = load_experiments_from_config(configurations)
    
    if not all_experiments:
        print("No experiments loaded successfully")
        return
    
    # Create plot generator
    generator = PlotGenerator(all_experiments)
    print(f"Minimum steps across all experiments: {generator.min_steps}")
    
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
    print("1. Generate all plots for all experiments in research config:")
    print("python plot_metrics.py --config research_config.py")
    print()
    print("2. Generate plots for specific environments:")
    print("python plot_metrics.py --config research_config.py --environments 'Karaoke Night' 'Medical Waiting Room'")
    print()
    print("3. Generate plots for specific whisper modes:")
    print("python plot_metrics.py --config research_config.py --whisper_modes '1 Agent' '2 Agents'")
    print()
    print("4. Generate specific plots only:")
    print("python plot_metrics.py --config research_config.py --plots conversation plans acceptance")
    print()
    print("5. Save plots to a specific directory:")
    print("python plot_metrics.py --config research_config.py --save_dir output_plots")
    print()
    print("6. Filter by whisper count:")
    print("python plot_metrics.py --config research_config.py --whisper_counts 1 2")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Generate plots from research config file')
    parser.add_argument('--config', type=str, default='research_config.py',
                       help='Path to research config file (default: research_config.py)')
    parser.add_argument('--environments', nargs='+', 
                       help='Filter by specific environments')
    parser.add_argument('--whisper_modes', nargs='+',
                       help='Filter by specific whisper modes')
    parser.add_argument('--whisper_counts', nargs='+', type=int,
                       help='Filter by specific whisper counts')
    parser.add_argument('--plots', nargs='+', 
        choices=['conversation', 'plans', 'acceptance', 'interactions', 'information', 'all', 'acceptance_matrix', 'interaction_matrix', 'whisper_gantt', 'all_whispers_gantt'],
        default=['all'], help='Which plots to generate')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (optional)')
    parser.add_argument('--no_file_check', action='store_true',
                       help='Skip file existence check (useful for testing)')
    
    args = parser.parse_args()
    
    # Load configurations
    configurations = load_research_configurations(args.config)
    
    if not configurations:
        print(f"No configurations found in {args.config}")
        return
    
    print(f"Found {len(configurations)} configurations in {args.config}")
    
    # Filter configurations
    filtered_configs = filter_configurations(
        configurations,
        environments=args.environments,
        whisper_modes=args.whisper_modes,
        whisper_counts=args.whisper_counts
    )
    
    if not filtered_configs:
        print("No configurations match the specified filters")
        return
    
    print(f"Using {len(filtered_configs)} configurations after filtering")
    
    # Load experiments
    experiments = load_experiments_from_config(filtered_configs, check_files=not args.no_file_check)
    
    if not experiments:
        print("No experiments loaded successfully")
        return
    
    # Create plot generator
    generator = PlotGenerator(experiments)
    print(f"Minimum steps across all experiments: {generator.min_steps}")
    
    # Generate requested plots
    if 'all' in args.plots:
        plots_to_generate = [
            'conversation', 'plans', 'acceptance', 'interactions', 'information',
            'acceptance_matrix', 'interaction_matrix', 'whisper_gantt'
        ]
    else:
        plots_to_generate = args.plots
    
    for plot_type in plots_to_generate:
        print(f"\nGenerating {plot_type} plot...")
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{plot_type}_plot.png")
        
        if plot_type == 'conversation':
            generator.plot_conversation_raster(save_path)
        elif plot_type == 'plans':
            generator.plot_plan_changes_raster(save_path)
        elif plot_type == 'acceptance':
            generator.plot_acceptance_rejection_network(save_path)
        elif plot_type == 'interactions':
            generator.plot_interaction_counts_network(save_path)
        elif plot_type == 'information':
            generator.plot_information_spread_network(save_path)
        elif plot_type == 'acceptance_matrix':
            generator.plot_acceptance_ratio_matrix(save_path)
        elif plot_type == 'interaction_matrix':
            generator.plot_interaction_count_matrix(save_path)
        elif plot_type == 'whisper_gantt':
            generator.plot_whisper_gantt_timeline(save_path)

if __name__ == "__main__":
    print("Improved Plot Generator with Research Config Support")
    print("=" * 60)
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        main()
    else:
        # Show command line examples
        command_line_example()
        
        print("\n" + "=" * 60)
        print("Programmatic usage example:")
        
        # Try to run the example (will fail if files don't exist, but shows the structure)
        try:
            example_usage()
        except FileNotFoundError as e:
            print(f"Example failed because files don't exist: {e}")
            print("This is expected - make sure your research_config.py file points to valid JSON files.")
        except Exception as e:
            print(f"Example failed with error: {e}") 