import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from typing import Dict, List
import numpy as np
import glob
import os
from datetime import datetime
import matplotlib.dates as mdates

def load_metrics(filepath: str) -> Dict:
    """Load metrics data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_interaction_network(metrics: Dict, output_path: str):
    """Create a network visualization of agent interactions."""
    plt.figure(figsize=(12, 8))
    G = nx.Graph()
    
    # Add nodes and edges based on interaction counts
    for agent, interactions in metrics['interaction_counts'].items():
        G.add_node(agent)
        for other_agent, count in interactions.items():
            G.add_edge(agent, other_agent, weight=count)
    
    # Draw the network
    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, width=weights, edge_color='gray')
    
    plt.title('Agent Interaction Network')
    plt.savefig(f"{output_path}/interaction_network.png")
    plt.close()

def plot_interaction_heatmap(metrics: Dict, output_path: str):
    """Create a heatmap of interaction frequencies between agents."""
    agents = list(metrics['interaction_counts'].keys())
    interaction_matrix = np.zeros((len(agents), len(agents)))
    
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if agent2 in metrics['interaction_counts'].get(agent1, {}):
                interaction_matrix[i, j] = metrics['interaction_counts'][agent1][agent2]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, xticklabels=agents, yticklabels=agents,
                annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Agent Interaction Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_path}/interaction_heatmap.png")
    plt.close()

def plot_information_spread(metrics: Dict, output_path: str):
    """Plot the amount of information each agent has received."""
    agents = list(metrics['information_spread'].keys())
    info_counts = [len(metrics['information_spread'][agent]) for agent in agents]
    
    plt.figure(figsize=(10, 6))
    plt.bar(agents, info_counts)
    plt.title('Information Spread per Agent')
    plt.xlabel('Agents')
    plt.ylabel('Number of Information Items')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/information_spread.png")
    plt.close()

def plot_acceptance_rejection(metrics: Dict, output_path: str):
    """Plot the acceptance/rejection rates."""
    plt.figure(figsize=(8, 6))
    plt.pie([metrics['acceptance_rejection']['accept'], 
             metrics['acceptance_rejection']['reject']],
            labels=['Accepted', 'Rejected'],
            autopct='%1.1f%%',
            colors=['lightgreen', 'lightcoral'])
    plt.title('Interaction Acceptance/Rejection Rates')
    plt.savefig(f"{output_path}/acceptance_rejection.png")
    plt.close()

def plot_zone_movements(metrics: Dict, output_path: str):
    """Plot the number of zone movements per agent."""
    agents = list(metrics['zone_movements'].keys())
    movement_counts = [metrics['zone_movements'][agent]['count'] for agent in agents]
    
    plt.figure(figsize=(10, 6))
    plt.bar(agents, movement_counts)
    plt.title('Zone Movements per Agent')
    plt.xlabel('Agents')
    plt.ylabel('Number of Zone Movements')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/zone_movements.png")
    plt.close()

def plot_conversation_durations(metrics: Dict, output_path: str):
    """Plot the distribution of conversation durations."""
    durations = [d['duration'] for d in metrics['conversation_durations']]
    
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Conversation Durations')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_path}/conversation_durations.png")
    plt.close()

def plot_temporal_interaction_network(metrics: Dict, output_path: str):
    """Create a temporal network visualization showing how interactions evolve over time."""
    plt.figure(figsize=(15, 10))
    G = nx.Graph()
    
    # Create a timeline of interactions
    timeline = []
    for conv in metrics['conversation_durations']:
        participants = conv['participants']
        timestamp = datetime.fromisoformat(conv['timestamp'])
        timeline.append((timestamp, participants))
    
    # Sort timeline
    timeline.sort(key=lambda x: x[0])
    
    # Create subplots for different time periods
    n_periods = min(4, len(timeline))
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, (timestamp, participants) in enumerate(timeline[:n_periods]):
        G = nx.Graph()
        for p in participants:
            G.add_node(p)
        for j in range(len(participants)):
            for k in range(j+1, len(participants)):
                G.add_edge(participants[j], participants[k])
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, ax=axes[i])
        axes[i].set_title(f'Interaction Network at {timestamp.strftime("%H:%M:%S")}')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/temporal_interaction_network.png")
    plt.close()

def plot_zone_movement_patterns(metrics: Dict, output_path: str):
    """Visualize zone movement patterns and sequences."""
    plt.figure(figsize=(15, 10))
    
    # Create a timeline of zone movements
    for agent, movements in metrics['zone_movements'].items():
        # Check if we have any movement data
        if not movements or not isinstance(movements, dict):
            continue
            
        # Try different possible data structures
        if 'zones' in movements and movements['zones']:
            # Plot zone history
            plt.figure(figsize=(12, 6))
            zones = movements['zones']
            # Create timestamps if not available
            timestamps = [datetime.now() for _ in range(len(zones))]
            
            plt.plot(timestamps, zones, marker='o', label=agent)
            plt.title(f'Zone Movement Timeline for {agent}')
            plt.xlabel('Time')
            plt.ylabel('Zone')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_path}/zone_movements_{agent}.png")
            plt.close()
        elif 'count' in movements and movements['count'] > 0:
            # Plot movement count
            plt.figure(figsize=(8, 6))
            plt.bar(['Movement Count'], [movements['count']])
            plt.title(f'Total Zone Movements for {agent}')
            plt.ylabel('Number of Movements')
            plt.tight_layout()
            plt.savefig(f"{output_path}/zone_movements_count_{agent}.png")
            plt.close()

def plot_interaction_density_timeline(metrics: Dict, output_path: str):
    """Plot the density of interactions over time."""
    # Check if interaction_density data exists
    if 'interaction_density' not in metrics or not metrics['interaction_density']:
        print("No interaction density data available")
        return
        
    try:
        plt.figure(figsize=(15, 8))
        
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in metrics['interaction_density']]
        counts = [d['count'] for d in metrics['interaction_density']]
        
        plt.plot(timestamps, counts, marker='o')
        plt.title('Interaction Density Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Interactions')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_path}/interaction_density_timeline.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting interaction density: {e}")

def plot_conversation_analysis(metrics: Dict, output_path: str):
    """Create multiple focused visualizations for conversation analysis."""
    if 'conversation_durations' not in metrics or not metrics['conversation_durations']:
        print("No conversation duration data available")
        return
        
    try:
        # Extract conversation data
        conversations = []
        for conv in metrics['conversation_durations']:
            if conv.get('context'):
                first_message = conv['context'][0][1]
                topic = ' '.join(first_message.split()[:5]) + '...'
                duration = conv['duration']
                participants = conv['participants']
                conversations.append({
                    'topic': topic,
                    'duration': duration,
                    'participants': participants,
                    'participant_count': len(participants)
                })
        
        if not conversations:
            print("No conversation context data available")
            return
            
        df = pd.DataFrame(conversations)
        
        # 1. Top 5 Longest Conversations
        plt.figure(figsize=(10, 6))
        top_5 = df.nlargest(5, 'duration')
        bars = plt.barh(range(len(top_5)), top_5['duration'], height=0.6)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, i, f'{width:.0f}s', ha='left', va='center', fontweight='bold')
        plt.yticks(range(len(top_5)), top_5['topic'])
        plt.title('Top 5 Longest Conversations')
        plt.xlabel('Duration (seconds)')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_path}/top_5_conversations.png", dpi=300)
        plt.close()
        
        # 2. Conversation Duration Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['duration'], bins=10, edgecolor='black')
        plt.title('Distribution of Conversation Durations')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Number of Conversations')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_path}/conversation_duration_distribution.png", dpi=300)
        plt.close()
        
        # 3. Participant Analysis
        plt.figure(figsize=(10, 6))
        participant_counts = df['participant_count'].value_counts().sort_index()
        plt.bar(participant_counts.index, participant_counts.values)
        plt.title('Number of Participants per Conversation')
        plt.xlabel('Number of Participants')
        plt.ylabel('Number of Conversations')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_path}/participant_analysis.png", dpi=300)
        plt.close()
        
        # 4. Topic Analysis (Alternative to Word Cloud)
        plt.figure(figsize=(10, 6))
        # Count occurrences of each topic
        topic_counts = df['topic'].value_counts().head(10)
        plt.barh(range(len(topic_counts)), topic_counts.values)
        plt.yticks(range(len(topic_counts)), topic_counts.index)
        plt.title('Top 10 Most Common Conversation Topics')
        plt.xlabel('Number of Occurrences')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_path}/topic_analysis.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting conversation analysis: {e}")
        # If there's an error, try to save at least the basic data
        try:
            with open(f"{output_path}/conversation_data.json", 'w') as f:
                json.dump(conversations, f, indent=2)
            print("Saved raw conversation data to conversation_data.json")
        except:
            print("Could not save conversation data")

def plot_agent_activity_timeline(metrics: Dict, output_path: str):
    """Create a timeline visualization of agent activities."""
    plt.figure(figsize=(15, 10))
    
    # Collect all agent activities
    activities = []
    for conv in metrics['conversation_durations']:
        timestamp = datetime.fromisoformat(conv['timestamp'])
        for participant in conv['participants']:
            activities.append((timestamp, participant, 'conversation'))
    
    # Create a timeline plot
    df = pd.DataFrame(activities, columns=['timestamp', 'agent', 'activity'])
    df = df.sort_values('timestamp')
    
    plt.figure(figsize=(12, 8))
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        plt.plot(agent_data['timestamp'], [agent] * len(agent_data), 'o', label=agent)
    
    plt.title('Agent Activity Timeline')
    plt.xlabel('Time')
    plt.ylabel('Agent')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/agent_activity_timeline.png")
    plt.close()

def plot_information_propagation(metrics: Dict, output_path: str):
    """Create visualizations of information propagation through the network."""
    if 'propagation_metrics' not in metrics:
        print("No propagation metrics available")
        return
    
    try:
        # Create a bar chart of information reach
        plt.figure(figsize=(12, 8))
        infos = list(metrics['propagation_metrics'].keys())
        reaches = [metrics['propagation_metrics'][info]['total_agents_reached'] for info in infos]
        
        # Truncate long information strings for display
        display_infos = [info[:30] + '...' if len(info) > 30 else info for info in infos]
        
        plt.bar(display_infos, reaches)
        plt.title('Information Reach by Content')
        plt.xlabel('Information Content')
        plt.ylabel('Number of Agents Reached')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_path}/information_reach.png")
        plt.close()
        
        # Create a visualization of propagation time
        plt.figure(figsize=(12, 8))
        prop_times = [metrics['propagation_metrics'][info]['average_propagation_time'] for info in infos]
        
        plt.bar(display_infos, prop_times)
        plt.title('Average Propagation Time by Information Content')
        plt.xlabel('Information Content')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_path}/propagation_time.png")
        plt.close()
        
        # For each piece of information, create a network graph showing propagation
        for info, info_data in metrics['propagation_metrics'].items():
            if len(info_data['propagation_paths']) > 1:  # Only if there's actual propagation
                plt.figure(figsize=(12, 10))
                G = nx.DiGraph()
                
                # Add all nodes and edges from the propagation paths
                for path in info_data['propagation_paths']:
                    if path['source'] not in G:
                        G.add_node(path['source'])
                    if path['target'] not in G:
                        G.add_node(path['target'])
                    G.add_edge(path['source'], path['target'])
                
                # Draw the network
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='lightblue',
                        node_size=2000, font_size=10, font_weight='bold',
                        arrows=True, arrowsize=20)
                plt.title(f'Information Propagation Network: {info[:30]}...' if len(info) > 30 else info)
                plt.tight_layout()
                # Create safe filename by replacing problematic characters
                safe_info = info.replace(' ', '_').replace("'", "").replace('"', '')[:30]
                plt.savefig(f"{output_path}/propagation_network_{safe_info}.png")
                plt.close()
        
    except Exception as e:
        print(f"Error plotting information propagation: {e}")

def generate_all_visualizations(metrics_file: str, output_dir: str, test_prefix: str):
    """Generate all visualizations from metrics data."""
    metrics = load_metrics(metrics_file)
    
    # Create a subdirectory for this test
    test_output_dir = os.path.join(output_dir, test_prefix)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create all visualizations
    plot_interaction_network(metrics, test_output_dir)
    plot_interaction_heatmap(metrics, test_output_dir)
    plot_information_spread(metrics, test_output_dir)
    plot_acceptance_rejection(metrics, test_output_dir)
    plot_zone_movements(metrics, test_output_dir)
    plot_conversation_durations(metrics, test_output_dir)
    
    # New visualizations
    plot_temporal_interaction_network(metrics, test_output_dir)
    plot_zone_movement_patterns(metrics, test_output_dir)
    plot_interaction_density_timeline(metrics, test_output_dir)
    plot_conversation_analysis(metrics, test_output_dir)
    plot_agent_activity_timeline(metrics, test_output_dir)
    plot_information_propagation(metrics, test_output_dir)

def find_metrics_folders(base_path: str, test_prefix: str) -> List[str]:
    """Find all folders matching the pattern test_X-s-Y-* and return their paths."""
    # Get the base directory from the input path
    base_dir = os.path.dirname(base_path)
    # Extract the test number from the prefix (e.g., "test_4" -> "4")
    test_num = test_prefix.split('_')[1]
    
    # Create pattern to match all possible subfolders (0-8)
    folders = []
    for i in range(9):  # 0 to 8 inclusive
        pattern = os.path.join(base_dir, f"{test_prefix}{i}-*")
        matching_folders = glob.glob(pattern)
        folders.extend(matching_folders)
    
    return folders

def combine_metrics(folders: List[str]) -> Dict:
    """Combine metrics from multiple JSON files into a single metrics dictionary."""
    combined_metrics = {
        'information_spread': {},
        'whisper_history': [],
        'interaction_counts': {},
        'plan_changes': {},
        'interaction_density': [],
        'acceptance_rejection': {
            'accept': 0,
            'reject': 0,
            'total': 0,
            'acceptance_rate': 0.0,
            'rejection_rate': 0.0,
            'by_agent': {},
            'by_interaction_type': {},
            'timestamps': [],
            'interaction_history': []
        },
        'zone_movements': {},
        'conversation_durations': [],
        'propagation_metrics': {}
    }
    
    # Track seen conversations to avoid duplicates
    seen_conversations = set()
    
    for folder in folders:
        metrics_file = os.path.join(folder, 'metrics.json')
        if not os.path.exists(metrics_file):
            continue
            
        metrics = load_metrics(metrics_file)
        
        # Combine information_spread
        for agent, info in metrics.get('information_spread', {}).items():
            if agent not in combined_metrics['information_spread']:
                combined_metrics['information_spread'][agent] = []
            combined_metrics['information_spread'][agent].extend(info)
        
        # Combine whisper_history
        combined_metrics['whisper_history'].extend(metrics.get('whisper_history', []))
        
        # Combine interaction_counts
        for agent, interactions in metrics.get('interaction_counts', {}).items():
            if agent not in combined_metrics['interaction_counts']:
                combined_metrics['interaction_counts'][agent] = {}
            for other_agent, count in interactions.items():
                combined_metrics['interaction_counts'][agent][other_agent] = \
                    combined_metrics['interaction_counts'][agent].get(other_agent, 0) + count
        
        # Combine plan_changes
        for agent, changes in metrics.get('plan_changes', {}).items():
            if agent not in combined_metrics['plan_changes']:
                combined_metrics['plan_changes'][agent] = []
            combined_metrics['plan_changes'][agent].extend(changes)
        
        # Combine interaction_density
        combined_metrics['interaction_density'].extend(metrics.get('interaction_density', []))
        
        # Combine acceptance_rejection data
        if 'acceptance_rejection' in metrics:
            ar_data = metrics['acceptance_rejection']
            
            # Update basic counts
            combined_metrics['acceptance_rejection']['accept'] += ar_data.get('accept', 0)
            combined_metrics['acceptance_rejection']['reject'] += ar_data.get('reject', 0)
            
            # Update by_agent data
            for agent, agent_data in ar_data.get('by_agent', {}).items():
                if agent not in combined_metrics['acceptance_rejection']['by_agent']:
                    combined_metrics['acceptance_rejection']['by_agent'][agent] = {
                        'accept': 0,
                        'reject': 0,
                        'total': 0,
                        'acceptance_rate': 0.0,
                        'rejection_rate': 0.0
                    }
                combined_metrics['acceptance_rejection']['by_agent'][agent]['accept'] += agent_data.get('accept', 0)
                combined_metrics['acceptance_rejection']['by_agent'][agent]['reject'] += agent_data.get('reject', 0)
                combined_metrics['acceptance_rejection']['by_agent'][agent]['total'] += agent_data.get('total', 0)
            
            # Update by_interaction_type data
            for i_type, type_data in ar_data.get('by_interaction_type', {}).items():
                if i_type not in combined_metrics['acceptance_rejection']['by_interaction_type']:
                    combined_metrics['acceptance_rejection']['by_interaction_type'][i_type] = {
                        'accept': 0,
                        'reject': 0,
                        'total': 0,
                        'acceptance_rate': 0.0,
                        'rejection_rate': 0.0
                    }
                combined_metrics['acceptance_rejection']['by_interaction_type'][i_type]['accept'] += type_data.get('accept', 0)
                combined_metrics['acceptance_rejection']['by_interaction_type'][i_type]['reject'] += type_data.get('reject', 0)
                combined_metrics['acceptance_rejection']['by_interaction_type'][i_type]['total'] += type_data.get('total', 0)
            
            # Update timestamps
            combined_metrics['acceptance_rejection']['timestamps'].extend(ar_data.get('timestamps', []))
            
            # Update interaction_history
            combined_metrics['acceptance_rejection']['interaction_history'].extend(ar_data.get('interaction_history', []))
        
        # Combine zone_movements
        for agent, movements in metrics.get('zone_movements', {}).items():
            if agent not in combined_metrics['zone_movements']:
                combined_metrics['zone_movements'][agent] = {
                    'count': 0,
                    'zones': [],
                    'zone_patterns': {},
                    'internal_movements': {},
                    'zone_sequences': [],
                    'zone_durations': {}
                }
            
            # Update count
            combined_metrics['zone_movements'][agent]['count'] += movements.get('count', 0)
            
            # Update zones
            combined_metrics['zone_movements'][agent]['zones'].extend(movements.get('zones', []))
            
            # Update zone_patterns
            for pattern, count in movements.get('zone_patterns', {}).items():
                combined_metrics['zone_movements'][agent]['zone_patterns'][pattern] = \
                    combined_metrics['zone_movements'][agent]['zone_patterns'].get(pattern, 0) + count
            
            # Update internal_movements
            for movement, count in movements.get('internal_movements', {}).items():
                combined_metrics['zone_movements'][agent]['internal_movements'][movement] = \
                    combined_metrics['zone_movements'][agent]['internal_movements'].get(movement, 0) + count
            
            # Update zone_sequences
            combined_metrics['zone_movements'][agent]['zone_sequences'].extend(movements.get('zone_sequences', []))
            
            # Update zone_durations
            for zone, duration in movements.get('zone_durations', {}).items():
                if zone not in combined_metrics['zone_movements'][agent]['zone_durations']:
                    combined_metrics['zone_movements'][agent]['zone_durations'][zone] = 0
                combined_metrics['zone_movements'][agent]['zone_durations'][zone] += duration
        
        # Combine conversation_durations with deduplication
        for conv in metrics.get('conversation_durations', []):
            # Create a unique key for the conversation based on its content
            conv_key = (
                tuple(sorted(conv['participants'])),
                conv['timestamp'],
                tuple((msg[0], msg[1]) for msg in conv['context'])
            )
            
            if conv_key not in seen_conversations:
                seen_conversations.add(conv_key)
                combined_metrics['conversation_durations'].append(conv)
        
        # Combine propagation_metrics
        if 'summary' in metrics and 'propagation_metrics' in metrics['summary']:
            for info, info_data in metrics['summary']['propagation_metrics'].items():
                if info not in combined_metrics['propagation_metrics']:
                    # First time seeing this information piece, initialize it
                    combined_metrics['propagation_metrics'][info] = {
                        'total_agents_reached': 0,
                        'propagation_paths': [],
                        'average_propagation_time': 0
                    }
                
                # Append all propagation paths
                combined_metrics['propagation_metrics'][info]['propagation_paths'].extend(
                    info_data.get('propagation_paths', [])
                )
    
    # Calculate total and rates for acceptance_rejection
    total = combined_metrics['acceptance_rejection']['accept'] + combined_metrics['acceptance_rejection']['reject']
    combined_metrics['acceptance_rejection']['total'] = total
    if total > 0:
        combined_metrics['acceptance_rejection']['acceptance_rate'] = combined_metrics['acceptance_rejection']['accept'] / total
        combined_metrics['acceptance_rejection']['rejection_rate'] = combined_metrics['acceptance_rejection']['reject'] / total
    
    # Calculate rates for each agent
    for agent_data in combined_metrics['acceptance_rejection']['by_agent'].values():
        agent_total = agent_data['total']
        if agent_total > 0:
            agent_data['acceptance_rate'] = agent_data['accept'] / agent_total
            agent_data['rejection_rate'] = agent_data['reject'] / agent_total
    
    # Calculate rates for each interaction type
    for type_data in combined_metrics['acceptance_rejection']['by_interaction_type'].values():
        type_total = type_data['total']
        if type_total > 0:
            type_data['acceptance_rate'] = type_data['accept'] / type_total
            type_data['rejection_rate'] = type_data['reject'] / type_total
    
    # Recalculate propagation metrics statistics
    for info, info_data in combined_metrics['propagation_metrics'].items():
        # Get unique set of agents who received this information
        agents_reached = set()
        for path in info_data['propagation_paths']:
            agents_reached.add(path['target'])
        
        # Update total_agents_reached
        info_data['total_agents_reached'] = len(agents_reached)
        
        # Recalculate average propagation time if we have multiple paths
        paths = info_data['propagation_paths']
        if len(paths) > 1:
            try:
                # Sort paths by timestamp
                paths.sort(key=lambda p: datetime.fromisoformat(p['timestamp']))
                
                # Calculate time differences from first message
                first_time = datetime.fromisoformat(paths[0]['timestamp'])
                time_diffs = [(datetime.fromisoformat(p['timestamp']) - first_time).total_seconds() 
                             for p in paths[1:]]
                
                # Calculate average propagation time
                if time_diffs:
                    info_data['average_propagation_time'] = sum(time_diffs) / len(time_diffs)
                else:
                    info_data['average_propagation_time'] = 0
            except Exception as e:
                print(f"Error calculating propagation time for '{info}': {e}")
                info_data['average_propagation_time'] = 0
    
    return combined_metrics

if __name__ == "__main__":
    # Change manually to the path 
    base_metrics_file = "environment/frontend_server/storage/"
    test_prefix = "informal_house_party-s-"
    output_dir = "visualizations"
    
    # Find all relevant folders
    folders = find_metrics_folders(base_metrics_file, test_prefix)

    print(f"Found {len(folders)} folders to process")
    
    # Combine metrics from all folders
    combined_metrics = combine_metrics(folders)
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined metrics to a new file in the visualizations folder with test prefix
    combined_metrics_file = os.path.join(output_dir, f"{test_prefix}combined_metrics.json")
    
    print(f"Saving combined metrics to {combined_metrics_file}")
    with open(combined_metrics_file, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    
    # Generate visualizations using the combined metrics
    generate_all_visualizations(combined_metrics_file, output_dir, test_prefix)