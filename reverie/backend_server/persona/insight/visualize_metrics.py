import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from typing import Dict, List
import numpy as np

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

def generate_all_visualizations(metrics_file: str, output_dir: str):
    """Generate all visualizations from metrics data."""
    metrics = load_metrics(metrics_file)
    
    # Create all visualizations
    plot_interaction_network(metrics, output_dir)
    plot_interaction_heatmap(metrics, output_dir)
    plot_information_spread(metrics, output_dir)
    plot_acceptance_rejection(metrics, output_dir)
    plot_zone_movements(metrics, output_dir)
    plot_conversation_durations(metrics, output_dir)

if __name__ == "__main__":
    # Example usage
    metrics_file = "/Users/adonaivera/Documents/emergent_agentics/environment/frontend_server/storage/test_1-s-0-0-186_good/metrics.json"
    output_dir = "visualizations"
    generate_all_visualizations(metrics_file, output_dir) 