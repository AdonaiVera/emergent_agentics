import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import textwrap
import argparse
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExperimentData:
    """Container for experiment data with metadata"""
    title: str
    file_path: str
    dataframes: Dict[str, pd.DataFrame]
    min_steps: int

class PlotGenerator:
    """Main class for generating plots from experiment data"""
    
    def __init__(self, experiments: List[ExperimentData]):
        self.experiments = experiments
        self.min_steps = min(exp.min_steps for exp in experiments) if experiments else 0
        
    @staticmethod
    def load_json_to_dataframes(file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load JSON file and create DataFrames for each top-level key"""
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            dataframes = {}
            for key, value in data.items():
                if isinstance(value, list):
                    dataframes[key] = pd.DataFrame(value)
                else:
                    dataframes[key] = pd.DataFrame([value])
            
            return dataframes
        
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            return None
        except Exception as e:
            print(f"An error occurred loading {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def calculate_min_steps(dataframes: Dict[str, pd.DataFrame]) -> int:
        """Calculate the minimum number of steps across all dataframes"""
        min_steps = float('inf')
        for key, df in dataframes.items():
            if 'step' in df.columns:
                min_steps = min(min_steps, df['step'].max())
        return min_steps if min_steps != float('inf') else 0
    
    def plot_conversation_raster(self, save_path: Optional[str] = None):
        """Generate conversation raster plot for all experiments"""
        fig, axes = plt.subplots(len(self.experiments), 1, figsize=(15, 4 * len(self.experiments)))
        if len(self.experiments) == 1:
            axes = [axes]
        
        for i, exp in enumerate(self.experiments):
            ax = axes[i]
            
            # Get conversation data
            df_conversation = exp.dataframes.get('conversation_durations', pd.DataFrame())
            if df_conversation.empty:
                continue
                
            # Clean and prepare data
            df_cleaned = pd.DataFrame(df_conversation)[['step', 'participants', 'location']].copy()
            df_cleaned['participant_set'] = df_cleaned['participants'].apply(tuple)
            df_unique = df_cleaned.drop_duplicates(subset=['step', 'participant_set']).drop(columns='participant_set').reset_index(drop=True)
            df_long = df_unique.explode('participants')
            
            # Assign colors to locations
            locations = df_long['location'].unique()
            location_palette = {loc: (random.random(), random.random(), random.random()) for loc in locations}
            
            # Map participants to positions
            participants = sorted(df_long['participants'].unique())
            participant_index = {name: i for i, name in enumerate(participants)}
            
            # Plot for each location
            for loc, color in location_palette.items():
                loc_data = df_long[df_long['location'] == loc]
                ax.scatter(
                    x=loc_data['step'],
                    y=loc_data['participants'].map(participant_index),
                    color=color,
                    label=loc,
                    s=60,
                    alpha=0.85
                )
            
            # Configure axes
            ax.set_yticks(range(len(participants)))
            ax.set_yticklabels(participants)
            ax.set_xlabel('Step')
            ax.set_ylabel('Participant')
            ax.set_title(f'{exp.title} - Agent Conversations Raster Plot')
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            ax.legend(title='Location', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.set_xlim(0, self.min_steps)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_plan_changes_raster(self, save_path: Optional[str] = None):
        """Generate plan changes raster plot for all experiments"""
        fig, axes = plt.subplots(len(self.experiments), 1, figsize=(15, 3 * len(self.experiments)))
        if len(self.experiments) == 1:
            axes = [axes]
        
        for i, exp in enumerate(self.experiments):
            ax = axes[i]
            
            # Get plan changes data
            df_plan_changes = exp.dataframes.get('plan_changes', pd.DataFrame())
            if df_plan_changes.empty:
                continue
            
            # Flatten the data
            flattened_data = []
            for person, events_series in df_plan_changes.items():
                for _, event_list in events_series.items():
                    for event in event_list:
                        if event["step"] != 0:
                            flattened_data.append({
                                "name": person,
                                "step": event["step"]
                            })
            
            df_flat = pd.DataFrame(flattened_data)
            if df_flat.empty:
                continue
            
            # Create palette
            unique_names = df_flat['name'].unique()
            palette = {name: sns.color_palette("hsv", len(unique_names))[i] for i, name in enumerate(unique_names)}
            
            # Create stripplot
            sns.stripplot(
                data=df_flat,
                x="step",
                y="name",
                palette=palette,
                jitter=False,
                size=6,
                linewidth=0.5,
                ax=ax
            )
            
            ax.set_title(f"{exp.title} - Plan Changes Over Time")
            ax.set_xlabel("Step")
            ax.set_ylabel("Person")
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            ax.set_xlim(0, self.min_steps)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_acceptance_rejection_network(self, save_path: Optional[str] = None):
        """Generate acceptance-rejection network for all experiments"""
        fig, axes = plt.subplots(1, len(self.experiments), figsize=(6 * len(self.experiments), 5))
        if len(self.experiments) == 1:
            axes = [axes]
        
        for i, exp in enumerate(self.experiments):
            ax = axes[i]
            
            # Get acceptance-rejection data
            df_acceptance = exp.dataframes.get('acceptance_rejection', pd.DataFrame())
            if df_acceptance.empty:
                continue
            
            interaction_history = df_acceptance.get('interaction_history', {})
            
            # Calculate interaction scores
            interaction_scores = {}
            for _, list_interactions in interaction_history.items():
                for event in list_interactions:
                    initiator = event["initiator"]
                    target = event["target"]
                    accepted = event["accepted"]
                    
                    pair = tuple(sorted((initiator, target)))
                    
                    if pair not in interaction_scores:
                        interaction_scores[pair] = {"accepted": 0, "rejected": 0}
                    
                    if accepted:
                        interaction_scores[pair]["accepted"] += 1
                    else:
                        interaction_scores[pair]["rejected"] += 1
            
            if not interaction_scores:
                continue
            
            # Create color map
            color_map = LinearSegmentedColormap.from_list("acceptance_gradient", ["red", "#d4af37", "green"])
            
            edge_colors = {}
            for pair, scores in interaction_scores.items():
                total = scores["accepted"] + scores["rejected"]
                ratio = scores["accepted"] / total if total > 0 else 0.5
                edge_colors[pair] = color_map(ratio)
            
            # Build graph
            G = nx.Graph()
            for pair in interaction_scores:
                G.add_edge(pair[0], pair[1])
            
            # Position and draw
            pos = nx.circular_layout(G)
            edges = G.edges()
            
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=400)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=edges,
                edge_color=[edge_colors[tuple(sorted(edge))] for edge in edges],
                width=2
            )
            
            ax.set_title(f"{exp.title} - Acceptance Ratio Network", fontsize=10)
            ax.axis('off')
            ax.margins(x=0.1)
        
        # Add colorbar
        norm = Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.03)
        cbar.set_label("Acceptance Ratio", fontsize=8)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["0 (Rejected)", "0.5 (Neutral)", "1 (Accepted)"])
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interaction_counts_network(self, save_path: Optional[str] = None):
        """Generate interaction counts network for all experiments"""
        fig, axes = plt.subplots(1, len(self.experiments), figsize=(6 * len(self.experiments), 5))
        if len(self.experiments) == 1:
            axes = [axes]
        
        for i, exp in enumerate(self.experiments):
            ax = axes[i]
            
            # Get interaction counts data
            df_interactions = exp.dataframes.get('interaction_counts', pd.DataFrame())
            if df_interactions.empty:
                continue
            
            # Process interaction counts
            interaction_counts = []
            seen = set()
            
            for _, row in df_interactions.iterrows():
                for person1_name in df_interactions.columns:
                    targets = row[person1_name]
                    
                    if isinstance(targets, dict):
                        for person2, count in targets.items():
                            pair = tuple(sorted((person1_name, person2)))
                            
                            if pair not in seen:
                                seen.add(pair)
                                interaction_counts.append({
                                    "Source": pair[0],
                                    "Target": pair[1],
                                    "Count": count
                                })
            
            df_cleaned = pd.DataFrame(interaction_counts)
            if df_cleaned.empty:
                continue
            
            # Normalize counts
            max_count = df_cleaned["Count"].max()
            min_count = df_cleaned["Count"].min()
            
            def normalize(val, min_val, max_val):
                return (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            # Create color map
            cmap = LinearSegmentedColormap.from_list("interaction_gradient", ["red", "#d4af37", "green"])
            
            # Build graph
            G = nx.Graph()
            for _, row in df_cleaned.iterrows():
                G.add_edge(row["Source"], row["Target"], weight=row["Count"])
            
            pos = nx.circular_layout(G)
            
            # Prepare edge colors
            edge_colors = []
            for u, v in G.edges():
                row = df_cleaned[((df_cleaned["Source"] == u) & (df_cleaned["Target"] == v)) | 
                               ((df_cleaned["Source"] == v) & (df_cleaned["Target"] == u))].iloc[0]
                norm_val = normalize(row["Count"], min_count, max_count)
                edge_colors.append(cmap(norm_val))
            
            # Draw graph
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', edgecolors='black', node_size=400)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=2)
            
            ax.set_title(f"{exp.title} - Interaction Counts Network", fontsize=10)
            ax.axis('off')
            ax.margins(x=0.1)
        
        # Add colorbar
        norm = Normalize(vmin=min_count, vmax=max_count)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.03)
        mid_val = round((min_count + max_count) / 2, 1)
        cbar.set_ticks([min_count, mid_val, max_count])
        cbar.set_ticklabels([f"{min_count}", f"{mid_val}", f"{max_count}"])
        cbar.set_label("Number of Interactions", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_information_spread_network(self, save_path: Optional[str] = None):
        """Generate information spread network for all experiments"""
        for i, exp in enumerate(self.experiments):
            # Get propagation metrics data
            df_propagation = exp.dataframes.get('propagation_metrics', pd.DataFrame())
            if df_propagation.empty:
                continue
            
            # Process whisper paths
            whisper_paths = {}
            for _, row in df_propagation.iterrows():
                for whisper, propagation in row.items():
                    whisper_path = []
                    if isinstance(propagation, dict):
                        paths = propagation['propagation_paths']
                        for path in paths:
                            if path['target'] not in [wpath[1] for wpath in whisper_path]:
                                whisper_path.append((path['source'], path['target'], path['step']))
                    whisper_paths[whisper] = whisper_path
            
            if not whisper_paths:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Initialize directed graph
            G = nx.DiGraph()
            
            # Group nodes by message
            nodes_by_msg = {}
            for msg, events in whisper_paths.items():
                nodes_by_msg[msg] = set()
                for src, tgt, t in events:
                    nodes_by_msg[msg].add((msg, tgt))
                    G.add_node((msg, tgt))
                    if src != 'system':
                        nodes_by_msg[msg].add((msg, src))
                        G.add_node((msg, src))
                        G.add_edge((msg, src), (msg, tgt), time=t)
            
            # Compute positions
            pos = nx.spring_layout(G, seed=42, k=1)
            
            # Assign colors
            cmap_nodes = plt.cm.Pastel1
            msg_to_color = {msg: cmap_nodes(i) for i, msg in enumerate(nodes_by_msg)}
            
            # Create edge colormap
            edge_cmap = LinearSegmentedColormap.from_list('time_cmap', ['#F3E03B', '#5D1451'])
            
            # Draw nodes
            for msg, nodes in nodes_by_msg.items():
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=list(nodes),
                    node_color=[msg_to_color[msg]] * len(nodes),
                    node_size=300,
                    ax=ax
                )
            
            # Prepare edges
            edge_list = []
            edge_colors = []
            for u, v, data in G.edges(data=True):
                edge_list.append((u, v))
                normalized_t = data['time'] / self.min_steps
                edge_colors.append(edge_cmap(normalized_t))
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edge_list,
                edge_color=edge_colors,
                arrowstyle='->',
                arrowsize=8,
                width=1.5,
                ax=ax
            )
            
            # Add labels
            labels = {node: node[1] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', ax=ax)
            
            # Add legend
            patches = []
            for msg, color in msg_to_color.items():
                wrapped_label = textwrap.fill(msg, 30)
                patches.append(mpatches.Patch(color=color, label=wrapped_label))
            
            legend = ax.legend(
                handles=patches,
                loc='center right',
                bbox_to_anchor=(1.4, 0.5),
                fontsize=8,
                handlelength=1.5,
                borderaxespad=0.1
            )
            legend._legend_box.align = "right"
            
            # Add colorbar
            norm = Normalize(vmin=0, vmax=self.min_steps)
            sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(
                sm,
                ax=ax,
                orientation='horizontal',
                fraction=0.05,
                pad=0.02,
                ticks=[0, self.min_steps//2, self.min_steps]
            )
            cbar.set_label('Time step', fontsize=8)
            cbar.ax.tick_params(labelsize=8)
            
            ax.set_title(f'{exp.title} - Information Spread Network', fontsize=12)
            ax.axis('off')
            ax.margins(x=0.1)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_exp_{i}.png", dpi=300, bbox_inches='tight')
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate plots from experiment JSON files')
    parser.add_argument('--json_files', nargs='+', required=True, 
                       help='List of JSON file paths')
    parser.add_argument('--titles', nargs='+', required=True,
                       help='List of titles for each experiment (must match number of JSON files)')
    parser.add_argument('--plots', nargs='+', choices=['conversation', 'plans', 'acceptance', 'interactions', 'information', 'all'],
                       default=['all'], help='Which plots to generate')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (optional)')
    
    args = parser.parse_args()
    
    if len(args.json_files) != len(args.titles):
        print("Error: Number of JSON files must match number of titles")
        return
    
    # Load experiments
    experiments = []
    for file_path, title in zip(args.json_files, args.titles):
        print(f"Loading {title} from {file_path}...")
        dataframes = PlotGenerator.load_json_to_dataframes(file_path)
        if dataframes:
            min_steps = PlotGenerator.calculate_min_steps(dataframes)
            experiments.append(ExperimentData(title=title, file_path=file_path, 
                                           dataframes=dataframes, min_steps=min_steps))
            print(f"  Loaded {len(dataframes)} dataframes, min steps: {min_steps}")
        else:
            print(f"  Failed to load {file_path}")
    
    if not experiments:
        print("No experiments loaded successfully")
        return
    
    # Create plot generator
    generator = PlotGenerator(experiments)
    print(f"Minimum steps across all experiments (for fair comparison): {generator.min_steps}")
    
    # Generate requested plots
    if 'all' in args.plots:
        plots_to_generate = ['conversation', 'plans', 'acceptance', 'interactions', 'information']
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

if __name__ == '__main__':
    main()