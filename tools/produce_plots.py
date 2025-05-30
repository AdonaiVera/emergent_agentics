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

JSON_FILE_PATH = 'formal_networking.json'
def load_json_to_dataframes(file_path):
    """
    Load JSON file and create DataFrames for each top-level key
    Returns a dictionary of DataFrames
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Create DataFrames for each top-level key
        dataframes = {}
        for key, value in data.items():
            # If the value is a list of dictionaries, perfect for DataFrame
            if isinstance(value, list):
                dataframes[key] = pd.DataFrame(value)
            else:
                # If not a list, create DataFrame with the value
                dataframes[key] = pd.DataFrame([value])
        
        return dataframes
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == '__main__':
    dfs = load_json_to_dataframes(JSON_FILE_PATH)
    
    if dfs is not None:
        # Extract the 9 features of the raw data
        if len(dfs) == 9:
            # Unpack the dictionary into individual DataFrames
            df1, df2, df3, df4, df5, df6, df7, df8, df9 = dfs.values()
            
            # Print confirmation
            print("Successfully created 8 DataFrames:")
            for i, key in enumerate(dfs.keys(), 1):
                print(f"DataFrame {i}: {key} (shape: {dfs[key].shape})")

            ###########################################################
            #     Start Raster plot for  convos vs time steps         #
            ###########################################################

            # Start clean: select relevant columns and eliminate redundant entries
            df8_cleaned = pd.DataFrame(df8)[['step', 'participants', 'location']].copy()
            df8_cleaned['participant_set'] = df8_cleaned['participants'].apply(tuple)
            df8_unique = df8_cleaned.drop_duplicates(subset=['step', 'participant_set']).drop(columns='participant_set').reset_index(drop=True)

            # Break out each participant to their own row for proper plotting
            df_long = df8_unique.explode('participants')

            # Assign each unique location a random RGB color
            locations = df_long['location'].unique()
            location_palette = {loc: (random.random(), random.random(), random.random()) for loc in locations}

            # Map participants to numerical y-axis positions
            participants = sorted(df_long['participants'].unique())
            participant_index = {name: i for i, name in enumerate(participants)}

            # Begin plotting
            plt.figure(figsize=(12, len(participants) * 0.4))

            # One scatter plot per location so we can control color by group
            for loc, color in location_palette.items():
                loc_data = df_long[df_long['location'] == loc]
                plt.scatter(
                    x=loc_data['step'],
                    y=loc_data['participants'].map(participant_index),
                    color=color,
                    label=loc,
                    s=80,
                    alpha=0.85
                )

            # Polish the axes and legend
            plt.yticks(ticks=range(len(participants)), labels=participants)
            plt.xlabel('Step')
            plt.ylabel('Participant')
            plt.title('Agent Conversations Raster Plot (colored by location)')
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.legend(title='Location', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

            
            ###########################################################
            #         End Raster plot for convos vs time steps        #
            ###########################################################
            
            ###########################################################
            #    Start Raster plot for change in plans vs time step   #
            ###########################################################
            
            # Step 1: Normalize the JSON-like structure into a flat DataFrame
            flattened_data = []

            for person, events_series in df4.items():
                for _, event_list in events_series.items():
                    for event in event_list: 
                        if event["step"] != 0: 
                            flattened_data.append({
                                "name": person,
                                "step": event["step"]
                            })

            df_flat = pd.DataFrame(flattened_data)

            # Step 2: Assign a color per person
            unique_names = df_flat['name'].unique()
            palette = {name: sns.color_palette("hsv", len(unique_names))[i] for i, name in enumerate(unique_names)}

            # Step 3: Create the raster plot
            plt.figure(figsize=(12, len(unique_names) * 0.5))
            sns.stripplot(
                data=df_flat,
                x="step",
                y="name",
                palette=palette,
                jitter=False,  # No vertical jitter, since it’s a raster plot
                size=6,        # Dot size
                linewidth=0.5
            )

            plt.title("Raster plot of actions over time by Person")
            plt.xlabel("Step")
            plt.ylabel("Person")
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            ###########################################################
            #    End Raster plot for change in plans vs time step   #
            ###########################################################
            
            ###########################################################
            #    Start network graph for acceptance-rejection         #
            ###########################################################
            
            df6_final = df6['interaction_history']

            interaction_scores = {}
            for _, list_interactions in df6_final.items():
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

            # Acceptance ratio colormap: red (0) -> gold (0.5) -> green (1)
            color_map = LinearSegmentedColormap.from_list("acceptance_gradient", ["red", "#d4af37", "green"])

            edge_colors = {}
            ratios = {}
            for pair, scores in interaction_scores.items():
                total = scores["accepted"] + scores["rejected"]
                ratio = scores["accepted"] / total if total > 0 else 0.5  # Neutral gray if no data
                edge_colors[pair] = color_map(ratio)
                ratios[pair] = ratio

            # Build graph
            G = nx.Graph()
            for pair in interaction_scores:
                G.add_edge(pair[0], pair[1])

            # Position and figure
            pos = nx.circular_layout(G)
            edges = G.edges()

            fig, ax = plt.subplots(figsize=(6, 5))

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=400)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=edges,
                edge_color=[edge_colors[tuple(sorted(edge))] for edge in edges],
                width=2
            )

            ax.set_title("Interaction Graph Colored by Acceptance Ratio", fontsize=10)
            ax.axis('off')
            ax.margins(x=0.1)

            # Add colorbar from 0 to 1
            norm = Normalize(vmin=0, vmax=1)
            sm = cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.03)
            cbar.set_label("Acceptance Ratio", fontsize=8)
            cbar.set_ticks([0.0, 0.5, 1.0])
            cbar.set_ticklabels(["0 (Rejected)", "0.5 (Neutral)", "1 (Accepted)"])
            cbar.ax.tick_params(labelsize=8)

            plt.tight_layout()
            plt.show()
            
            ###########################################################
            #       End network graph for acceptance-rejection        #
            ###########################################################
            
            ###########################################################
            #      Start network graph for interaction counts         #
            ###########################################################
            
            interaction_counts = []
            seen = set()

            for _, row in df3.iterrows():
                # Access each person's name from the index of the row
                for person1_name in df3.columns:
                    # Extract the dictionary of interactions for this person
                    targets = row[person1_name]
                    
                    if isinstance(targets, dict):  # Ensure that targets is a dictionary
                        for person2, count in targets.items():
                            # Sort the pair lexicographically to avoid duplicate edges
                            pair = tuple(sorted((person1_name, person2)))
                            
                            # Only add this edge if it's not already seen
                            if pair not in seen:
                                seen.add(pair)
                                interaction_counts.append({
                                    "Source": pair[0],
                                    "Target": pair[1],
                                    "Count": count
                                })
                       
            df6_cleaned = pd.DataFrame(interaction_counts)
            
            # Normalize counts to [0, 1] for color mapping
            max_count = df6_cleaned["Count"].max()
            min_count = df6_cleaned["Count"].min()

            def normalize(val, min_val, max_val):
                return (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5

            # Create color map: dark red -> dark yellow -> dark green
            cmap = LinearSegmentedColormap.from_list("interaction_gradient", ["red", "#d4af37", "green"])

            # Build graph
            G = nx.Graph()
            for _, row in df6_cleaned.iterrows():
                G.add_edge(row["Source"], row["Target"], weight=row["Count"])

            # Position nodes in a circle
            pos = nx.circular_layout(G)

            # Prepare edge colors based on normalized counts
            edge_colors = []
            for u, v in G.edges():
                # Get count for this edge (regardless of order)
                row = df6_cleaned[((df6_cleaned["Source"] == u) & (df6_cleaned["Target"] == v)) | ((df6_cleaned["Source"] == v) & (df6_cleaned["Target"] == u))].iloc[0]
                norm_val = normalize(row["Count"], min_count, max_count)
                edge_colors.append(cmap(norm_val))

            # Create a ScalarMappable for the colorbar
            norm = Normalize(vmin=min_count, vmax=max_count)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(6, 5))  # Reserve space for colorbar

            # Draw graph on ax
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', edgecolors='black', node_size=400)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=2)

            ax.set_title("Agent Network color-coded by number of interactions", fontsize=10)
            ax.axis('off')
            ax.margins(x=0.1)

            # Add colorbar, tied to ax
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.03)
            mid_val = round((min_count + max_count) / 2, 1)
            cbar.set_ticks([min_count, mid_val, max_count])
            cbar.set_ticklabels([f"{min_count}", f"{mid_val}", f"{max_count}"])
            cbar.set_label("Number of Interactions", fontsize=8)
            cbar.ax.tick_params(labelsize=8)

            plt.show()
            
            ###########################################################
            #      End network graph for interaction counts           #
            ###########################################################
            
            ###########################################################
            #      Start network graph for information spread         #
            ###########################################################

            whisper_paths = {}
            seen = set()

            for _, row in df9.iterrows():
                for whisper, propagation in row.items():
                    whisper_path = []
                    if isinstance(propagation, dict):
                        paths = propagation['propagation_paths']
                        for  path in paths:
                            if path['target'] not in [wpath[1] for wpath in whisper_path]:
                                whisper_path.append((path['source'], path['target'], path['step']))
                    whisper_paths[whisper] = whisper_path

            fig, ax = plt.subplots(figsize=(8, 6))

            # Initialize a directed graph
            G = nx.DiGraph()

            # Group nodes by message and populate the graph
            nodes_by_msg = {}
            for msg, events in whisper_paths.items():
                nodes_by_msg[msg] = set()
                for src, tgt, t in events:
                    # Add each target node
                    nodes_by_msg[msg].add((msg, tgt))
                    G.add_node((msg, tgt))
                    if src != 'system':
                        nodes_by_msg[msg].add((msg, src))
                        G.add_node((msg, src))
                        G.add_edge((msg, src), (msg, tgt), time=t)

            # Compute positions for all nodes using a force-directed layout
            pos = nx.spring_layout(G, seed=42, k=1)

            # Assign a unique pastel color to each message group
            cmap_nodes = plt.cm.Pastel1
            msg_to_color = {msg: cmap_nodes(i) for i, msg in enumerate(nodes_by_msg)}

            # Create a continuous colormap for edge timestamps
            edge_cmap = LinearSegmentedColormap.from_list('time_cmap', ['#F3E03B', '#5D1451'])

            # Draw nodes for each message group
            for msg, nodes in nodes_by_msg.items():
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=list(nodes),
                    node_color=[msg_to_color[msg]] * len(nodes),
                    node_size=300,
                    ax=ax
                )

            # Prepare edge list and color mapping based on timestamp
            edge_list = []
            edge_colors = []
            for u, v, data in G.edges(data=True):
                edge_list.append((u, v))
                normalized_t = data['time'] / 1800  # normalize to [0,1]
                edge_colors.append(edge_cmap(normalized_t))

            # Draw edges with arrows and timestamp-based colors
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edge_list,
                edge_color=edge_colors,
                arrowstyle='->',
                arrowsize=8,
                width=1.5,
                ax=ax
            )
            ax.margins(x=0.1)

            # Label each node with the agent’s name
            labels = {node: node[1] for node in G.nodes()}
            nx.draw_networkx_labels(
                G, pos,
                labels,
                font_size=8,
                font_color='black',
                ax=ax
            )

            # Build a legend mapping each message to its node color, with text wrapping
            patches = []
            for msg, color in msg_to_color.items():
                wrapped_label = textwrap.fill(msg, 30)
                patches.append(mpatches.Patch(color=color, label=wrapped_label))

            # Place the legend outside the main plot area
            legend = ax.legend(
                handles=patches,
                loc='center right',
                bbox_to_anchor=(1.4,0.5),
                fontsize=8,
                handlelength=1.5,
                borderaxespad=0.1
            )
            legend._legend_box.align = "right"

            # Configure a colorbar for edge timestamps
            norm = Normalize(vmin=0, vmax=1800)
            sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(
                sm,
                ax=ax,
                orientation='horizontal',
                fraction=0.05,
                pad=0.02,
                ticks=[0, 900, 1800]
            )
            cbar.set_label('Time step', fontsize=8)
            cbar.ax.tick_params(labelsize=8)

            ax.set_title('Directed graph of Information spread', fontsize=10)
            ax.axis('off')
            fig.tight_layout()
            plt.show()

            ###########################################################
            #        End network graph for information spread         #
            ###########################################################

        else:
            print(f"Expected 8 top-level keys, found {len(dfs)}")
    else:
        print("Failed to load data.")