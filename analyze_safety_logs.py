#!/usr/bin/env python3
"""
Safety Log Analysis Script

This script analyzes ALL safety_log_situation_*.json files to:
1. Extract unsafe activity data for each agent across all situations
2. Calculate safety improvement percentages
3. Generate a CSV report for the current situation
4. Create graphs showing changes over time with averages across all situations
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def load_all_safety_logs():
    """Load all safety log files for all situations"""
    pattern = "safety_log_situation_*.json"
    log_files = glob.glob(f"reverie/backend_server/logs/{pattern}")
    
    if not log_files:
        print(f"No safety log files found")
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
                    print(f"Loaded situation {situation_index} from {log_file}")
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return all_situations_logs

def get_available_situations():
    """Return sorted list of available situation indices from logs folder"""
    pattern = "reverie/backend_server/logs/safety_log_situation_*.json"
    log_files = glob.glob(pattern)
    situations = []
    for path in log_files:
        m = re.search(r"safety_log_situation_(\d+)\.json", path)
        if m:
            situations.append(int(m.group(1)))
    return sorted(set(situations))

def load_safety_logs(situation_index):
    """Load safety log files for a specific situation index (for backward compatibility)"""
    pattern = f"safety_log_situation_{situation_index}.json"
    log_files = glob.glob(f"reverie/backend_server/logs/{pattern}")
    
    if not log_files:
        print(f"No safety log files found for situation {situation_index}")
        return []
    
    all_logs = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                all_logs.append(data)
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return all_logs

def extract_agent_data(logs):
    """Extract agent data from safety logs"""
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

def analyze_unsafe_activities(agents_data):
    """Analyze unsafe activities for each agent"""
    analysis = {}
    
    for agent_name, agent_data in agents_data.items():
        first_log = agent_data['first_log']
        last_log = agent_data['last_log']
        
        if not first_log or not last_log:
            continue
        
        # Count unsafe activities in first log
        first_unsafe_count = sum(1 for activity in first_log['unsafe_activity_images'] 
                               if not activity.get('safe', True))
        first_total = len(first_log['unsafe_activity_images'])
        
        # Count unsafe activities in last log
        last_unsafe_count = sum(1 for activity in last_log['unsafe_activity_images'] 
                              if not activity.get('safe', True))
        last_total = len(last_log['unsafe_activity_images'])
        
        # Calculate improvement percentage
        if first_unsafe_count > 0:
            improvement_pct = ((first_unsafe_count - last_unsafe_count) / first_unsafe_count) * 100
        else:
            improvement_pct = 0
        
        analysis[agent_name] = {
            'first_unsafe': first_unsafe_count,
            'first_total': first_total,
            'last_unsafe': last_unsafe_count,
            'last_total': last_total,
            'improvement_pct': improvement_pct,
            'first_daily_req': first_log['daily_req'],
            'last_daily_req': last_log['daily_req']
        }
    
    return analysis

def create_csv_report(analysis, agents_data, situation_index, output_file):
    """Create CSV report with one row per agent, counting safe activities from START to END"""
    
    print("Creating CSV report with one row per agent...")
    
    # Create DataFrame
    rows = []
    
    # Add one row per agent
    for agent_name in analysis.keys():
        row = {'Actividad unsafe (Tematica)': agent_name}
        
        # Calculate improvement for this agent
        agent_improvement = calculate_agent_improvement(agent_name, agents_data)
        row['Improvement'] = f"{agent_improvement:.0f}%"
        
        rows.append(row)
    
    # Add average row
    if rows:
        improvements = [float(row['Improvement'].replace('%', '')) for row in rows]
        overall_avg = np.mean(improvements)
        avg_row = {
            'Actividad unsafe (Tematica)': 'Average x agent',
            'Improvement': f"{overall_avg:.0f}%"
        }
        rows.append(avg_row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Enhanced CSV report saved to: {output_file}")
    
    # Print sample of the data for verification
    print("\nSample CSV data:")
    for i, row in enumerate(rows):
        print(f"Row {i+1}: {row}")
    
    return df

def calculate_agent_improvement(agent_name, agents_data):
    """Calculate improvement percentage for a specific agent from START to END"""
    
    agent_data = agents_data[agent_name]
    first_log = agent_data['first_log']
    last_log = agent_data['last_log']
    
    if not first_log or not last_log:
        print(f"  No START/END logs found for {agent_name}")
        return 0.0
    
    print(f"  Calculating improvement for {agent_name}")
    print(f"    START log step: {first_log['step']}, END log step: {last_log['step']}")
    
    # Count unsafe activities in START log
    start_unsafe_count = 0
    start_total_count = len(first_log['unsafe_activity_images'])
    
    for unsafe_data in first_log['unsafe_activity_images']:
        if not unsafe_data.get('safe', True):
            start_unsafe_count += 1
    
    # Count unsafe activities in END log
    end_unsafe_count = 0
    end_total_count = len(last_log['unsafe_activity_images'])
    
    for unsafe_data in last_log['unsafe_activity_images']:
        if not unsafe_data.get('safe', True):
            end_unsafe_count += 1
    
    print(f"    START: {start_unsafe_count}/{start_total_count} unsafe activities")
    print(f"    END: {end_unsafe_count}/{end_total_count} unsafe activities")
    
    # Calculate improvement percentage
    if start_unsafe_count == 0:
        improvement = 100.0  # Already safe
    elif end_unsafe_count == 0:
        improvement = 100.0  # Became completely safe
    elif end_unsafe_count == start_unsafe_count:
        improvement = 0.0    # No improvement
    else:
        improvement = ((start_unsafe_count - end_unsafe_count) / start_unsafe_count) * 100
    
    print(f"    Improvement: {improvement:.0f}%")
    return improvement



def create_change_graph(agents_data, situation_index, output_file):
    """Create enhanced graph showing changes over time with standard deviation and averages"""
    plt.figure(figsize=(14, 10))
    
    # Collect all data for statistical analysis
    all_steps_data = defaultdict(list)
    agent_colors = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # First pass: collect all data and assign colors
    for i, (agent_name, agent_data) in enumerate(agents_data.items()):
        agent_colors[agent_name] = color_palette[i % len(color_palette)]
        
        # Sort logs by step
        sorted_logs = sorted(agent_data['all_logs'], key=lambda x: x['step'])
        
        for log in sorted_logs:
            step = log['step']
            unsafe_count = sum(1 for activity in log['unsafe_activity_images'] 
                            if not activity.get('safe', True))
            
            all_steps_data[step].append(unsafe_count)
    
    # Calculate statistics for each step
    steps = sorted(all_steps_data.keys())
    means = []
    stds = []
    
    for step in steps:
        step_data = all_steps_data[step]
        means.append(np.mean(step_data))
        stds.append(np.std(step_data))
    
    # Plot individual agent lines
    for agent_name, agent_data in agents_data.items():
        sorted_logs = sorted(agent_data['all_logs'], key=lambda x: x['step'])
        
        agent_steps = []
        agent_unsafe_counts = []
        
        for log in sorted_logs:
            step = log['step']
            unsafe_count = sum(1 for activity in log['unsafe_activity_images'] 
                            if not activity.get('safe', True))
            
            agent_steps.append(step)
            agent_unsafe_counts.append(unsafe_count)
        
        if agent_steps:
            plt.plot(agent_steps, agent_unsafe_counts, 
                    marker='o', label=f'{agent_name}', 
                    color=agent_colors[agent_name],
                    linewidth=2, markersize=6, alpha=0.8)
    
    # Plot mean line with error bars (standard deviation)
    if len(steps) > 1:
        plt.errorbar(steps, means, yerr=stds, 
                    fmt='o-', color='black', 
                    label='Mean ± Std Dev', 
                    linewidth=3, markersize=8, 
                    capsize=5, capthick=2, alpha=0.9)
        
        # Add trend line for overall improvement
        if len(steps) > 1:
            z = np.polyfit(steps, means, 1)
            p = np.poly1d(z)
            plt.plot(steps, p(steps), "--", color='red', alpha=0.8, 
                    linewidth=2, label='Overall Trend')
    
    # Enhanced styling
    plt.xlabel('Simulation Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Unsafe Activities', fontsize=14, fontweight='bold')
    plt.title(f'Safety Improvement Over Time - Situation {situation_index}\n(Mean ± Standard Deviation)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Customize legend
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
    
    # Enhanced grid and styling
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(steps, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add statistics text box
    if len(steps) > 1:
        initial_mean = means[0]
        final_mean = means[-1]
        improvement = ((initial_mean - final_mean) / initial_mean * 100) if initial_mean > 0 else 0
        
        stats_text = f'Overall Improvement: {improvement:.1f}%\n'
        stats_text += f'Initial Mean: {initial_mean:.1f} unsafe activities\n'
        stats_text += f'Final Mean: {final_mean:.1f} unsafe activities\n'
        stats_text += f'Total Steps: {len(steps)}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set axis limits with some padding
    if steps:
        plt.xlim(min(steps) - 10, max(steps) + 10)
        if means:
            max_y = max(max(means) + max(stds), max([max(all_steps_data[step]) for step in steps]))
            plt.ylim(-0.5, max_y + 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Enhanced graph saved to: {output_file}")
    plt.show()

def create_change_graph_all_situations(all_situations_logs, output_file):
    """Plot 5 agents' mean±std across all simulations, plus a global mean±std."""
    plt.figure(figsize=(16, 12))
    
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
    
    # Plot per-agent mean ± std across situations
    for idx, agent_name in enumerate(selected_agents):
        step_map = agent_to_step_counts[agent_name]
        agent_steps = sorted(step_map.keys())
        if not agent_steps:
            continue
        means = [float(np.mean(step_map[s])) for s in agent_steps]
        stds = [float(np.std(step_map[s])) for s in agent_steps]
        color = color_palette[idx % len(color_palette)]
        plt.plot(agent_steps, means, marker='o', linewidth=2.5, markersize=6, color=color, label=f"{agent_name} (mean)", alpha=0.6)
        plt.fill_between(agent_steps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.1, label=f"{agent_name} (std)")
    
    # Plot global mean ± std across all agents and situations
    global_steps = sorted(global_step_counts.keys())
    if global_steps:
        global_means = [float(np.mean(global_step_counts[s])) for s in global_steps]
        global_stds = [float(np.std(global_step_counts[s])) for s in global_steps]
        plt.plot(global_steps, global_means, color='black', linewidth=4.5, markersize=10, marker='o', label='Global mean', alpha=1.0)
        plt.fill_between(global_steps, np.array(global_means) - np.array(global_stds), np.array(global_means) + np.array(global_stds), color='black', alpha=0.25, label='Global std')
    
    # Labels and styling
    plt.xlabel('Simulation Steps', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Unsafe Activities', fontsize=16, fontweight='bold')
    plt.title('Safety Improvement Over Time - Aggregated Across Simulations (5 agents + global)', fontsize=18, fontweight='bold', pad=25)
    
    # Legend formatting
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9, fancybox=True, shadow=True, ncol=1)
    
    # X/Y limits and grid
    all_steps_for_limits = sorted(set(list(global_step_counts.keys())))
    if all_steps_for_limits:
        plt.xticks(all_steps_for_limits, fontsize=12)
        plt.xlim(min(all_steps_for_limits) - 10, max(all_steps_for_limits) + 10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yticks(fontsize=12)
    
    # Stats box for global series
    if global_steps:
        initial_mean = global_means[0]
        final_mean = global_means[-1]
        improvement = ((initial_mean - final_mean) / initial_mean * 100) if initial_mean > 0 else 0
        stats_text = f"Global Improvement: {improvement:.1f}%\nInitial Mean: {initial_mean:.1f}\nFinal Mean: {final_mean:.1f}\nAgents plotted: {len(selected_agents)}\nSituations: {len(all_situations_logs)}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Aggregated graph (5 agents + global) saved to: {output_file}")
    plt.show()

def save_daily_requirements(analysis, situation_index, output_dir):
    """Save daily requirements for first and last logs"""
    os.makedirs(output_dir, exist_ok=True)
    
    for agent_name, agent_data in analysis.items():
        # Save first daily requirements
        first_file = os.path.join(output_dir, f"{agent_name}_first_daily_req_situation_{situation_index}.txt")
        with open(first_file, 'w', encoding='utf-8') as f:
            f.write(f"First Daily Requirements for {agent_name} - Situation {situation_index}\n")
            f.write("=" * 60 + "\n\n")
            for i, activity in enumerate(agent_data['first_daily_req'], 1):
                f.write(f"{i}. {activity}\n")
        
        # Save last daily requirements
        last_file = os.path.join(output_dir, f"{agent_name}_last_daily_req_situation_{situation_index}.txt")
        with open(last_file, 'w', encoding='utf-8') as f:
            f.write(f"Last Daily Requirements for {agent_name} - Situation {situation_index}\n")
            f.write("=" * 60 + "\n\n")
            for i, activity in enumerate(agent_data['last_daily_req'], 1):
                f.write(f"{i}. {activity}\n")
    
    print(f"Daily requirements saved to: {output_dir}")

def main():
    """Main function to run the analysis"""
    output_dir = "safety_analysis_output"
    
    # Load ALL safety logs and detect available situations
    all_situations_logs = load_all_safety_logs()
    if not all_situations_logs:
        print("No logs found for any situation. Exiting.")
        return
    available = sorted(all_situations_logs.keys())
    print(f"Loaded logs for {len(available)} situations: {available}")
    
    # Choose one situation for CSV/daily requirements (first available)
    situation_index = available[0]
    print(f"Analyzing safety logs for situation {situation_index} (auto-detected)...")
    
    # Load safety logs for specific situation (for CSV and daily requirements)
    logs = [all_situations_logs[situation_index]]
    print(f"Loaded 1 log file for situation {situation_index}")
    
    # Extract agent data for specific situation
    agents_data = extract_agent_data(logs)
    print(f"Found {len(agents_data)} agents: {list(agents_data.keys())}")
    
    # Analyze unsafe activities
    analysis = analyze_unsafe_activities(agents_data)
    
    if not analysis:
        print("No analysis data available. Exiting.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate CSV report (for specific situation)
    csv_file = os.path.join(output_dir, f"safety_analysis_situation_{situation_index}.csv")
    df = create_csv_report(analysis, agents_data, situation_index, csv_file)
    
    # Create enhanced graph for all situations (averaged across all JSONs)
    graph_file = os.path.join(output_dir, f"safety_changes_graph_all_situations.png")
    create_change_graph_all_situations(all_situations_logs, graph_file)
    
    # Also create the original graph for the specific situation
    original_graph_file = os.path.join(output_dir, f"safety_changes_graph_situation_{situation_index}.png")
    create_change_graph(agents_data, situation_index, original_graph_file)
    
    # Save daily requirements
    daily_req_dir = os.path.join(output_dir, f"daily_requirements_situation_{situation_index}")
    save_daily_requirements(analysis, situation_index, daily_req_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    for agent_name, agent_data in analysis.items():
        print(f"\n{agent_name}:")
        print(f"  First log: {agent_data['first_unsafe']}/{agent_data['first_total']} unsafe activities")
        print(f"  Last log: {agent_data['last_unsafe']}/{agent_data['last_total']} unsafe activities")
        print(f"  Improvement: {agent_data['improvement_pct']:.1f}%")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Graphs created:")
    print(f"  - {graph_file} (All situations average)")
    print(f"  - {original_graph_file} (Situation {situation_index} only)")

if __name__ == "__main__":
    main()
