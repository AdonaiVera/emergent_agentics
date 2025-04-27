"""
Metrics tracking system for party simulation.
Tracks various quantifiable metrics about agent interactions and behaviors.
"""

import json
from datetime import datetime
from typing import Dict, List, Set, Tuple

class PartyMetrics:
    def __init__(self):
        # Information spread metrics
        self.information_spread = {}  # Track what information each agent knows
        self.whisper_history = []  # Track whisper events and their spread
        
        # Social clustering metrics
        self.interaction_counts = {}  # Count interactions between pairs of agents
        self.social_groups = {}  # Track social groups over time
        
        # Adaptation metrics
        self.plan_changes = {}  # Track how often agents modify their plans
        
        # Interaction metrics
        self.interaction_density = []  # Count of interactions per time unit
        self.acceptance_rejection = {"accept": 0, "reject": 0}  # Track accept/reject rates
        
        # Mobility metrics
        self.zone_movements = {}  # Track agent movements between zones
        self.internal_movements = {}  # Track movements within zones
        self.zone_sequences = {}  # Track sequences of zone visits
        self.zone_durations = {}  # Track time spent in each zone
        self.last_zone_update = {}  # Track last zone update time
        self.conversation_durations = []  # Track duration of conversations
        self.seen_conversations = set()  # Track unique conversations to prevent duplicates
        
        # Time tracking
        self.start_time = None
        self.current_step = 0
        
    def initialize_agent(self, agent_name: str):
        """Initialize metrics tracking for a new agent."""
        if agent_name not in self.interaction_counts:
            self.interaction_counts[agent_name] = {}
        if agent_name not in self.information_spread:
            self.information_spread[agent_name] = set()
        if agent_name not in self.zone_movements:
            self.zone_movements[agent_name] = {"count": 0, "zones": set()}
        if agent_name not in self.internal_movements:
            self.internal_movements[agent_name] = {}
        if agent_name not in self.zone_sequences:
            self.zone_sequences[agent_name] = []
        if agent_name not in self.zone_durations:
            self.zone_durations[agent_name] = {}
        if agent_name not in self.last_zone_update:
            self.last_zone_update[agent_name] = None
            
    def track_information_spread(self, source: str, target: str, information: str):
        """Track how information spreads between agents."""
        if target not in self.information_spread:
            self.initialize_agent(target)
        self.information_spread[target].add(information)
        self.whisper_history.append({
            "step": self.current_step,
            "source": source,
            "target": target,
            "information": information,
            "timestamp": datetime.now().isoformat()
        })
        
        # Track propagation path
        if not hasattr(self, 'propagation_paths'):
            self.propagation_paths = {}
        if information not in self.propagation_paths:
            self.propagation_paths[information] = []
        self.propagation_paths[information].append({
            "step": self.current_step,
            "source": source,
            "target": target,
            "timestamp": datetime.now().isoformat()
        })
        
    def track_interaction(self, agent1: str, agent2: str):
        """Track interactions between two agents."""
        if agent1 not in self.interaction_counts:
            self.initialize_agent(agent1)
        if agent2 not in self.interaction_counts[agent1]:
            self.interaction_counts[agent1][agent2] = 0
        self.interaction_counts[agent1][agent2] += 1
        
    def track_plan_change(self, agent: str, old_plan: str, new_plan: str, daily_req: List[str] = None):
        """Track when agents modify their plans.
        
        Args:
            agent: Name of the agent
            old_plan: Previous plan address
            new_plan: New plan address
            daily_req: List of daily requirements for the agent
        """
        if agent not in self.plan_changes:
            self.plan_changes[agent] = []
        self.plan_changes[agent].append({
            "step": self.current_step,
            "old_plan": old_plan,
            "new_plan": new_plan,
            "daily_req": daily_req,
            "timestamp": datetime.now().isoformat()
        })
        
    def track_interaction_density(self, count: int, active_agents: List[str] = None, interaction_types: Dict[str, int] = None):
        """Track the density of interactions per time unit.
        
        Args:
            count: Total number of interactions in this step
            active_agents: List of agents who were involved in interactions
            interaction_types: Dictionary mapping interaction types to their counts
        """
        self.interaction_density.append({
            "step": self.current_step,
            "count": count,
            "active_agents": active_agents or [],
            "interaction_types": interaction_types or {},
            "timestamp": datetime.now().isoformat()
        })
        
    def track_acceptance_rejection(self, accepted: bool, initiator: str, target: str):
        """Track when agents accept or reject interactions.
        
        Args:
            accepted: Whether the interaction was accepted
            initiator: Name of the agent who initiated the interaction
            target: Name of the target agent
            reason: Optional reason for acceptance/rejection
        """
        if accepted:
            self.acceptance_rejection["accept"] += 1
        else:
            self.acceptance_rejection["reject"] += 1
            
        # Track detailed interaction history
        if "interaction_history" not in self.acceptance_rejection:
            self.acceptance_rejection["interaction_history"] = []
            
        self.acceptance_rejection["interaction_history"].append({
            "step": self.current_step,
            "initiator": initiator,
            "target": target,
            "accepted": accepted,
            "timestamp": datetime.now().isoformat()
        })
        
    def track_zone_movement(self, agent: str, from_zone: str, to_zone: str):
        """Track agent movements between and within zones.
        
        Args:
            agent: Name of the agent
            from_zone: Zone the agent is moving from
            to_zone: Zone the agent is moving to
            from_tile: (x,y) coordinates of the starting tile
            to_tile: (x,y) coordinates of the destination tile
        """
        if agent not in self.zone_movements:
            self.initialize_agent(agent)
            
        current_time = datetime.now()
        
        # Track time spent in previous zone
        if self.last_zone_update[agent] and from_zone:
            duration = (current_time - self.last_zone_update[agent]).total_seconds()
            if from_zone not in self.zone_durations[agent]:
                self.zone_durations[agent][from_zone] = 0
            self.zone_durations[agent][from_zone] += duration
            
        # Update last zone update time
        self.last_zone_update[agent] = current_time
        
        # Track zone-to-zone movement
        if from_zone != to_zone:
            self.zone_movements[agent]["count"] += 1
            self.zone_movements[agent]["zones"].add(from_zone)
            self.zone_movements[agent]["zones"].add(to_zone)
            
            # Track zone sequence
            self.zone_sequences[agent].append({
                "from_zone": from_zone,
                "to_zone": to_zone,
                "timestamp": current_time.isoformat(),
                "step": self.current_step
            })
            
            # Track movement patterns between specific zones
            if "zone_patterns" not in self.zone_movements[agent]:
                self.zone_movements[agent]["zone_patterns"] = {}
            pattern_key = f"{from_zone}->{to_zone}"
            if pattern_key not in self.zone_movements[agent]["zone_patterns"]:
                self.zone_movements[agent]["zone_patterns"][pattern_key] = 0
            self.zone_movements[agent]["zone_patterns"][pattern_key] += 1
            
            
    def track_conversation_duration(self, duration: float, participants: List[str], location: str = None, context: str = None):
        """Track the duration of conversations with detailed information.
        
        Args:
            duration: Duration of the conversation in seconds
            participants: List of agent names involved in the conversation
            location: Zone or location where the conversation took place
            context: Optional context or topic of the conversation
        """
        # Create a unique key for the conversation based on its content
        hashable_context = None
        if context:
            if isinstance(context, list):
                # If context is a list of messages, make it hashable
                # Each message might be [speaker, text] or similar
                hashable_context = tuple(
                    tuple(item) if isinstance(item, list) else item
                    for item in context
                )
            else:
                # If it's already a string or other hashable type
                hashable_context = context
        
        conv_key = (
            tuple(sorted(participants)),
            hashable_context
        )
        
        # Only track if we haven't seen this conversation before
        if conv_key not in self.seen_conversations:
            self.seen_conversations.add(conv_key)
            self.conversation_durations.append({
                "step": self.current_step,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "participants": participants,
                "location": location,
                "context": context
            })
        
    def update_step(self, step: int):
        """Update the current simulation step."""
        self.current_step = step
        if self.start_time is None:
            self.start_time = datetime.now()
            
    def get_metrics_summary(self) -> Dict:
        """Get a summary of all tracked metrics."""
        # Calculate information propagation metrics
        propagation_metrics = {}
        if hasattr(self, 'propagation_paths'):
            for info, paths in self.propagation_paths.items():
                # Calculate how many agents received this information
                agents_received = set()
                for path in paths:
                    agents_received.add(path['target'])
                
                # Calculate average propagation time
                if len(paths) > 1:
                    timestamps = [datetime.fromisoformat(p['timestamp']) for p in paths]
                    time_diffs = [(timestamps[i] - timestamps[0]).total_seconds() 
                                for i in range(1, len(timestamps))]
                    avg_prop_time = sum(time_diffs) / len(time_diffs)
                else:
                    avg_prop_time = 0
                
                propagation_metrics[info] = {
                    "total_agents_reached": len(agents_received),
                    "propagation_paths": paths,
                    "average_propagation_time": avg_prop_time
                }
        
        # Add topic mentions summary
        topic_mention_count = {}
        if hasattr(self, 'topic_mentions'):
            for mention in self.topic_mentions:
                topic = mention['topic']
                if topic not in topic_mention_count:
                    topic_mention_count[topic] = 0
                topic_mention_count[topic] += 1
        
        return {
            "information_spread": {
                agent: len(info) for agent, info in self.information_spread.items()
            },
            "propagation_metrics": propagation_metrics,
            "interaction_counts": self.interaction_counts,
            "plan_changes": {
                agent: len(changes) for agent, changes in self.plan_changes.items()
            },
            "interaction_density": len(self.interaction_density),
            "acceptance_rejection_ratio": (
                self.acceptance_rejection["accept"] / 
                (self.acceptance_rejection["accept"] + self.acceptance_rejection["reject"])
                if (self.acceptance_rejection["accept"] + self.acceptance_rejection["reject"]) > 0 
                else 0
            ),
            "zone_movements": {
                agent: {
                    "count": data["count"],
                    "zones": list(data["zones"]),
                    "zone_patterns": data.get("zone_patterns", {}),
                    "internal_movements": self.internal_movements[agent],
                    "zone_sequences": self.zone_sequences[agent],
                    "zone_durations": self.zone_durations[agent]
                } for agent, data in self.zone_movements.items()
            },
            "average_conversation_duration": (
                sum(d["duration"] for d in self.conversation_durations) / 
                len(self.conversation_durations)
                if self.conversation_durations else 0
            ),
            "total_steps": self.current_step,
            "simulation_duration": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time else 0
            ),
            "topic_mentions": topic_mention_count
        }
        
    def save_metrics(self, filepath: str):
        """Save all metrics to a JSON file."""
        try:
            # Convert all sets to lists for JSON serialization
            metrics_data = {
                "information_spread": {
                    agent: list(info) for agent, info in self.information_spread.items()
                },
                "whisper_history": self.whisper_history,
                "interaction_counts": self.interaction_counts,
                "plan_changes": self.plan_changes,
                "interaction_density": self.interaction_density,
                "acceptance_rejection": self.acceptance_rejection,
                "zone_movements": {
                    agent: {
                        "count": data["count"],
                        "zones": list(data["zones"]),
                        "zone_patterns": data.get("zone_patterns", {}),
                        "internal_movements": {
                            zone: {
                                "count": movements["count"],
                                "tiles_visited": [list(tile) for tile in movements["tiles_visited"]],
                                "movement_history": movements["movement_history"]
                            } for zone, movements in self.internal_movements[agent].items()
                        },
                        "zone_sequences": self.zone_sequences[agent],
                        "zone_durations": self.zone_durations[agent]
                    } for agent, data in self.zone_movements.items()
                },
                "conversation_durations": self.conversation_durations,
                "topic_mentions": getattr(self, 'topic_mentions', []),
                "summary": self.get_metrics_summary()
            }
            
            print("Metrics data")
            print(metrics_data)
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2) 
        except Exception as e:
            print(f"Error saving metrics: {e}")
            raise e

    def track_topic_mention(self, speaker: str, listener: str, topic: str, context: str = None):
        """Track when a specific topic is mentioned in a conversation.
        
        Args:
            speaker: Name of the agent who mentioned the topic
            listener: Name of the agent who heard the topic
            topic: The topic that was mentioned
            context: Optional context from the conversation
        """
        if not hasattr(self, 'topic_mentions'):
            self.topic_mentions = []
        
        self.topic_mentions.append({
            "step": self.current_step,
            "speaker": speaker,
            "listener": listener,
            "topic": topic,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        
        # Also track as information spread
        self.track_information_spread(speaker, listener, topic)
