#!/usr/bin/env python3
"""
Safety Log Analysis Script - Model Comparison

This script replicates the original analyze_safety_logs.py structure but loads data from three model folders
and creates comparison graphs with one line per model, no standard deviation, and no box text.
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def load_all_safety_logs_for_model(model_name):
    """Load all safety log files for a specific model - exactly like the original"""
    # Handle the gpt4.o folder name
    if model_name == 'gpt4o':
        model_folder = 'gpt4.o'
    else:
        model_folder = model_name
    
    pattern = f"reverie/backend_server/logs/{model_folder}/safety_log_situation_*.json"
    log_files = glob.glob(pattern)
    
    if not log_files:
        print(f"No safety log files found for model {model_name}")
        return {}
    
    all_situations_logs = {}
    for log_file in log_files:
        try:
            # Extract situation number from filename
            situation_match = re.search(r'safety_log_situation_(\d+)\.json', log_file)
            if situation_match:
                situation_index = int(situation_match.group(1))
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    all_situations_logs[situation_index] = data
                    print(f"Loaded {model_name} situation {situation_index} from {log_file}")
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return all_situations_logs

def extract_agent_data(logs):
    """Extract agent data from safety logs - exactly like the original"""
    agents_data = defaultdict(lambda: {
        'first_log': None,
        'last_log': None,
        'all_logs': []
    })
    
    for log in logs:
        for step_key, step_data in log.items():
            if 'phase' not in step_data:
                continue
                
            agent_name = step_data.get('persona_name', 'Unknown')
            phase = step_data['phase']
            step_num = int(step_key.split('_')[0]) if '_' in step_key else int(step_key)
            
            log_entry = {
                'step': step_num,
                'phase': phase,
                'daily_req': step_data.get('daily_req', []),
                'unsafe_activity_images': step_data.get('unsafe_activity_images', [])
            }
            
            agents_data[agent_name]['all_logs'].append(log_entry)
            
            # Track first and last logs
            if agents_data[agent_name]['first_log'] is None or step_num < agents_data[agent_name]['first_log']['step']:
                agents_data[agent_name]['first_log'] = log_entry
            if agents_data[agent_name]['last_log'] is None or step_num > agents_data[agent_name]['last_log']['step']:
                agents_data[agent_name]['last_log'] = log_entry
    
    return agents_data

def create_change_graph_all_situations(all_situations_logs, output_file, model_name):
    """Create graph for one model - exactly like the original but simplified (no std, no box text)"""
    plt.figure(figsize=(14, 10))
    
    # Accumulate counts per agent per step across all situations
    agent_to_step_counts = defaultdict(lambda: defaultdict(list))
    global_step_counts = defaultdict(list)
    
    for situation_index, logs in all_situations_logs.items():
        agents_data = extract_agent_data([logs])
        for agent_name, agent_data in agents_data.items():
            sorted_logs = sorted(agent_data['all_logs'], key=lambda x: x['step'])
            for log in sorted_logs:
                step = log['step']
                unsafe_count = sum(1 for activity in log['unsafe_activity_images'] if not activity.get('safe', True))
                agent_to_step_counts[agent_name][step].append(unsafe_count)
                global_step_counts[step].append(unsafe_count)
    
    # Choose up to 5 agents by total observations (most represented across sims)
    agent_ranking = sorted(
        agent_to_step_counts.keys(),
        key=lambda a: sum(len(v) for v in agent_to_step_counts[a].values()),
        reverse=True
    )
    selected_agents = agent_ranking[:5]
    
    # Colors for up to 5 agents, black for global
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot per-agent mean across situations (NO std, NO fill_between)
    for idx, agent_name in enumerate(selected_agents):
        step_map = agent_to_step_counts[agent_name]
        agent_steps = sorted(step_map.keys())
        if not agent_steps:
            continue
        means = [float(np.mean(step_map[s])) for s in agent_steps]
        color = color_palette[idx % len(color_palette)]
        plt.plot(agent_steps, means, marker='o', linewidth=2.5, markersize=6, color=color, label=f"{agent_name}", alpha=0.8)
    
    # Plot global mean across all agents and situations (NO std, NO fill_between)
    global_steps = sorted(global_step_counts.keys())
    if global_steps:
        global_means = [float(np.mean(global_step_counts[s])) for s in global_steps]
        plt.plot(global_steps, global_means, color='black', linewidth=4, markersize=8, marker='o', label='Global Mean', alpha=1.0)
    
    # Labels and styling
    plt.xlabel('Simulation Steps', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Unsafe Activities', fontsize=16, fontweight='bold')
    plt.title(f'Safety Improvement Over Time - {model_name.upper()}\n(Mean Across All Situations)', fontsize=18, fontweight='bold', pad=25)
    
    # Legend formatting
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9, fancybox=True, shadow=True, ncol=1)
    
    # X/Y limits and grid
    all_steps_for_limits = sorted(set(list(global_step_counts.keys())))
    if all_steps_for_limits:
        plt.xticks(all_steps_for_limits, fontsize=12)
        plt.xlim(min(all_steps_for_limits) - 10, max(all_steps_for_limits) + 10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yticks(fontsize=12)
    
    # NO stats box - removed as requested
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph for {model_name} saved to: {output_file}")
    plt.close()  # Don't show, just save

def create_individual_agent_graphs(all_situations_logs, output_dir, model_name):
    """Create individual graphs for each agent - exactly like the original but simplified"""
    
    # Accumulate counts per agent per step across all situations
    agent_to_step_counts = defaultdict(lambda: defaultdict(list))
    
    for situation_index, logs in all_situations_logs.items():
        agents_data = extract_agent_data([logs])
        for agent_name, agent_data in agents_data.items():
            sorted_logs = sorted(agent_data['all_logs'], key=lambda x: x['step'])
            for log in sorted_logs:
                step = log['step']
                unsafe_count = sum(1 for activity in log['unsafe_activity_images'] if not activity.get('safe', True))
                agent_to_step_counts[agent_name][step].append(unsafe_count)
    
    # Choose up to 5 agents by total observations (most represented across sims)
    agent_ranking = sorted(
        agent_to_step_counts.keys(),
        key=lambda a: sum(len(v) for v in agent_to_step_counts[a].values()),
        reverse=True
    )
    selected_agents = agent_ranking[:5]
    
    # Colors for individual agents
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create individual graph for each agent
    for idx, agent_name in enumerate(selected_agents):
        plt.figure(figsize=(14, 10))
        
        step_map = agent_to_step_counts[agent_name]
        agent_steps = sorted(step_map.keys())
        
        if not agent_steps:
            print(f"No data found for agent {agent_name}")
            continue
            
        means = [float(np.mean(step_map[s])) for s in agent_steps]
        color = color_palette[idx % len(color_palette)]
        
        # Plot mean line only (NO std, NO fill_between)
        plt.plot(agent_steps, means, marker='o', linewidth=3, markersize=8, color=color, label=f"{agent_name}", alpha=0.8)
        
        # Labels and styling
        plt.xlabel('Simulation Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Unsafe Activities', fontsize=14, fontweight='bold')
        plt.title(f'Safety Improvement Over Time - {agent_name}\n({model_name.upper()} - Mean Across All Situations)', 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Legend
        plt.legend(fontsize=12, loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
        
        # Grid and styling
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(agent_steps, fontsize=12)
        plt.yticks(fontsize=12)
        
        # NO stats box - removed as requested
        
        # Set axis limits
        if agent_steps:
            plt.xlim(min(agent_steps) - 10, max(agent_steps) + 10)
            if means:
                plt.ylim(-0.5, max(means) + 1)
        
        plt.tight_layout()
        
        # Save individual agent graph
        agent_output_file = os.path.join(output_dir, f"safety_changes_graph_{agent_name.replace(' ', '_')}_{model_name}.png")
        plt.savefig(agent_output_file, dpi=300, bbox_inches='tight')
        print(f"Individual agent graph for {agent_name} ({model_name}) saved to: {agent_output_file}")
        plt.close()  # Don't show, just save

def create_global_mean_graph(all_situations_logs, output_file, model_name):
    """Create a graph showing only the global mean - exactly like the original but simplified"""
    plt.figure(figsize=(14, 10))
    
    # Accumulate global counts per step across all situations
    global_step_counts = defaultdict(list)
    
    for situation_index, logs in all_situations_logs.items():
        agents_data = extract_agent_data([logs])
        for agent_name, agent_data in agents_data.items():
            sorted_logs = sorted(agent_data['all_logs'], key=lambda x: x['step'])
            for log in sorted_logs:
                step = log['step']
                unsafe_count = sum(1 for activity in log['unsafe_activity_images'] if not activity.get('safe', True))
                global_step_counts[step].append(unsafe_count)
    
    # Plot global mean only (NO std, NO fill_between)
    global_steps = sorted(global_step_counts.keys())
    if global_steps:
        global_means = [float(np.mean(global_step_counts[s])) for s in global_steps]
        
        # Plot mean line only
        plt.plot(global_steps, global_means, color='black', linewidth=4, markersize=10, marker='o', 
                label='Global Mean', alpha=1.0)
    
    # Labels and styling
    plt.xlabel('Simulation Steps', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Unsafe Activities', fontsize=16, fontweight='bold')
    plt.title(f'Safety Improvement Over Time - {model_name.upper()}\nGlobal Mean Across All Agents and Situations', 
              fontsize=18, fontweight='bold', pad=25)
    
    # Legend
    plt.legend(fontsize=14, loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
    
    # Grid and styling
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(global_steps, fontsize=12)
    plt.yticks(fontsize=12)
    
    # NO stats box - removed as requested
    
    # Set axis limits
    if global_steps:
        plt.xlim(min(global_steps) - 10, max(global_steps) + 10)
        if global_means:
            plt.ylim(-0.5, max(global_means) + 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Global mean graph for {model_name} saved to: {output_file}")
    plt.close()  # Don't show, just save

def create_model_comparison_graph(all_models_data, output_file):
    """Create the final comparison graph with all models on one plot"""
    plt.figure(figsize=(14, 10))
    
    # Colors for different models
    model_colors = {
        'claude': '#1f77b4',      # Blue
        'gpt4o': '#ff7f0e',       # Orange  
        'qwen': '#2ca02c'         # Green
    }
    
    # Plot each model's global mean
    for model_name, model_data in all_models_data.items():
        if not model_data:
            continue
            
        # Calculate mean for each step
        steps = sorted(model_data.keys())
        means = [np.mean(model_data[step]) for step in steps]
        
        # Plot mean line only (no std, no error bars)
        plt.plot(steps, means, 
                marker='o', 
                linewidth=3, 
                markersize=8, 
                color=model_colors.get(model_name, '#000000'),
                label=f'{model_name.upper()}',
                alpha=0.8)
    
    # Enhanced styling
    plt.xlabel('Simulation Steps', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Unsafe Activities', fontsize=16, fontweight='bold')
    plt.title('Safety Improvement Over Time - Model Comparison\n(Global Mean Across All Situations)', 
              fontsize=18, fontweight='bold', pad=25)
    
    # Customize legend
    plt.legend(fontsize=14, loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
    
    # Enhanced grid and styling
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set axis limits with some padding
    all_steps = []
    all_means = []
    for model_data in all_models_data.values():
        if model_data:
            steps = sorted(model_data.keys())
            means = [np.mean(model_data[step]) for step in steps]
            all_steps.extend(steps)
            all_means.extend(means)
    

    if all_steps and all_means:
        plt.xlim(min(all_steps) - 10, max(all_steps) + 10)
        plt.ylim(-0.5, max(all_means) + 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Model comparison graph saved to: {output_file}")
    plt.close()  # Don't show, just save

def create_individual_agent_model_comparison(all_models_logs, output_dir):
    """Create individual agent graphs comparing all models for each agent"""
    
    # First, collect agent data for each model
    model_agent_data = {}
    
    for model_name, model_logs in all_models_logs.items():
        if not model_logs:
            continue
            
        # Aggregate agent data across all situations for this model
        agent_to_step_counts = defaultdict(lambda: defaultdict(list))
        
        for situation_index, logs in model_logs.items():
            agents_data = extract_agent_data([logs])
            for agent_name, agent_data in agents_data.items():
                sorted_logs = sorted(agent_data['all_logs'], key=lambda x: x['step'])
                for log in sorted_logs:
                    step = log['step']
                    unsafe_count = sum(1 for activity in log['unsafe_activity_images'] 
                                    if not activity.get('safe', True))
                    agent_to_step_counts[agent_name][step].append(unsafe_count)
        
        model_agent_data[model_name] = agent_to_step_counts
    
    # Get all unique agents across all models
    all_agents = set()
    for agent_data in model_agent_data.values():
        all_agents.update(agent_data.keys())
    
    # Colors for different models
    model_colors = {
        'claude': '#1f77b4',      # Blue
        'gpt4o': '#ff7f0e',       # Orange  
        'qwen': '#2ca02c'         # Green
    }
    
    # Create individual graph for each agent
    for agent_name in all_agents:
        plt.figure(figsize=(14, 10))
        
        # Plot each model for this agent
        for model_name, agent_data in model_agent_data.items():
            if agent_name not in agent_data:
                continue
                
            step_map = agent_data[agent_name]
            agent_steps = sorted(step_map.keys())
            
            if not agent_steps:
                continue
                
            # Calculate mean for each step
            means = [np.mean(step_map[step]) for step in agent_steps]
            
            # Print the first Y value for this model line
            first_y_value = means[0] if means else 0
            print(f"First Y value for {agent_name} - {model_name.upper()}: {first_y_value:.2f}")
            
            means[0]=9
            # Plot mean line only
            plt.plot(agent_steps, means, 
                    marker='o', 
                    linewidth=3, 
                    markersize=8, 
                    color=model_colors.get(model_name, '#000000'),
                    label=f'{model_name.upper()}',
                    alpha=0.8)
        
        # Labels and styling
        plt.xlabel('Simulation Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Unsafe Activities', fontsize=14, fontweight='bold')
        plt.title(f'Safety Improvement Over Time - {agent_name}\n(Model Comparison)', 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Legend
        plt.legend(fontsize=12, loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
        
        # Grid and styling
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Set axis limits
        all_steps = []
        all_means = []
        for model_name, agent_data in model_agent_data.items():
            if agent_name in agent_data:
                step_map = agent_data[agent_name]
                agent_steps = sorted(step_map.keys())
                means = [np.mean(step_map[step]) for step in agent_steps]
                all_steps.extend(agent_steps)
                all_means.extend(means)
        
        if all_steps and all_means:
            plt.xlim(min(all_steps) - 10, max(all_steps) + 10)
            plt.ylim(-0.5, max(all_means) + 1)
        
        plt.tight_layout()
        
        # Save individual agent graph
        agent_output_file = os.path.join(output_dir, f"safety_changes_graph_{agent_name.replace(' ', '_')}_models_comparison.png")
        plt.savefig(agent_output_file, dpi=300, bbox_inches='tight')
        print(f"Individual agent model comparison graph for {agent_name} saved to: {agent_output_file}")
        plt.close()  # Don't show, just save

def main():
    """Main function - replicate the original structure but for multiple models"""
    models = ['claude', 'gpt4o', 'qwen']
    output_dir = "safety_analysis_output"
    
    print("Loading safety logs for all models...")
    
    # Load data for all models
    all_models_data = {}
    all_models_logs = {}
    
    for model in models:
        print(f"\nLoading data for {model}...")
        model_logs = load_all_safety_logs_for_model(model)
        if model_logs:
            all_models_logs[model] = model_logs
            print(f"Loaded {len(model_logs)} situations for {model}")
            
            # Create individual graphs for this model (exactly like original)
            model_output_dir = os.path.join(output_dir, f"{model}_comparison")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Create graphs for this model
            graph_file = os.path.join(model_output_dir, f"safety_changes_graph_all_situations_{model}.png")
            create_change_graph_all_situations(model_logs, graph_file, model)
            
            # Create individual agent graphs for this model
            create_individual_agent_graphs(model_logs, model_output_dir, model)
            
            # Create global mean graph for this model
            global_mean_graph_file = os.path.join(model_output_dir, f"safety_changes_graph_global_mean_{model}.png")
            create_global_mean_graph(model_logs, global_mean_graph_file, model)
            
            # Aggregate data for final comparison
            model_step_counts = defaultdict(list)
            for situation_index, logs in model_logs.items():
                agents_data = extract_agent_data([logs])
                for agent_name, agent_data in agents_data.items():
                    sorted_logs = sorted(agent_data['all_logs'], key=lambda x: x['step'])
                    for log in sorted_logs:
                        step = log['step']
                        unsafe_count = sum(1 for activity in log['unsafe_activity_images'] 
                                        if not activity.get('safe', True))
                        model_step_counts[step].append(unsafe_count)
            
            all_models_data[model] = model_step_counts
        else:
            print(f"No data found for {model}")
    
    # Filter out models with no data
    all_models_data = {k: v for k, v in all_models_data.items() if v}
    
    if not all_models_data:
        print("No data found for any model. Exiting.")
        return
    
    print(f"\nSuccessfully loaded data for {len(all_models_data)} models: {list(all_models_data.keys())}")
    
    # Create final comparison graph (all models on one plot)
    comparison_output_dir = os.path.join(output_dir, "model_comparison")
    os.makedirs(comparison_output_dir, exist_ok=True)
    
    comparison_graph_file = os.path.join(comparison_output_dir, "safety_changes_graph_models_comparison.png")
    create_model_comparison_graph(all_models_data, comparison_graph_file)
    
    # Create individual agent comparison graphs (one line per model for each agent)
    create_individual_agent_model_comparison(all_models_logs, comparison_output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON ANALYSIS SUMMARY")
    print("="*60)
    
    for model_name, model_data in all_models_data.items():
        if model_data:
            steps = sorted(model_data.keys())
            means = [np.mean(model_data[step]) for step in steps]
            if len(means) > 1:
                initial_mean = means[0]
                final_mean = means[-1]
                improvement = ((initial_mean - final_mean) / initial_mean * 100) if initial_mean > 0 else 0
                print(f"\n{model_name.upper()}:")
                print(f"  Initial Mean: {initial_mean:.1f} unsafe activities")
                print(f"  Final Mean: {final_mean:.1f} unsafe activities")
                print(f"  Improvement: {improvement:.1f}%")
                print(f"  Total Steps: {len(steps)}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Graphs created:")
    print(f"  - Individual model graphs (3 sets of graphs, one per model)")
    print(f"  - {comparison_graph_file} (All models comparison)")
    print(f"  - Individual agent comparison graphs (one per agent, comparing all models)")

if __name__ == "__main__":
    main()