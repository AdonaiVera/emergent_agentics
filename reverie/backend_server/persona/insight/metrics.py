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
        self.conversation_durations = []  # Track duration of conversations
        
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
        
    def track_interaction(self, agent1: str, agent2: str):
        """Track interactions between two agents."""
        if agent1 not in self.interaction_counts:
            self.initialize_agent(agent1)
        if agent2 not in self.interaction_counts[agent1]:
            self.interaction_counts[agent1][agent2] = 0
        self.interaction_counts[agent1][agent2] += 1
        
    def track_plan_change(self, agent: str, old_plan: str, new_plan: str):
        """Track when agents modify their plans."""
        if agent not in self.plan_changes:
            self.plan_changes[agent] = []
        self.plan_changes[agent].append({
            "step": self.current_step,
            "old_plan": old_plan,
            "new_plan": new_plan,
            "timestamp": datetime.now().isoformat()
        })
        
    def track_interaction_density(self, count: int):
        """Track the density of interactions per time unit."""
        self.interaction_density.append({
            "step": self.current_step,
            "count": count,
            "timestamp": datetime.now().isoformat()
        })
        
    def track_acceptance_rejection(self, accepted: bool):
        """Track when agents accept or reject interactions."""
        if accepted:
            self.acceptance_rejection["accept"] += 1
        else:
            self.acceptance_rejection["reject"] += 1
            
    def track_zone_movement(self, agent: str, from_zone: str, to_zone: str):
        """Track agent movements between zones."""
        if agent not in self.zone_movements:
            self.initialize_agent(agent)
        self.zone_movements[agent]["count"] += 1
        self.zone_movements[agent]["zones"].add(from_zone)
        self.zone_movements[agent]["zones"].add(to_zone)
        
    def track_conversation_duration(self, duration: float):
        """Track the duration of conversations."""
        self.conversation_durations.append({
            "step": self.current_step,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
    def update_step(self, step: int):
        """Update the current simulation step."""
        self.current_step = step
        if self.start_time is None:
            self.start_time = datetime.now()
            
    def get_metrics_summary(self) -> Dict:
        """Get a summary of all tracked metrics."""
        return {
            "information_spread": {
                agent: len(info) for agent, info in self.information_spread.items()
            },
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
            "zone_movements": self.zone_movements,
            "average_conversation_duration": (
                sum(d["duration"] for d in self.conversation_durations) / 
                len(self.conversation_durations)
                if self.conversation_durations else 0
            ),
            "total_steps": self.current_step,
            "simulation_duration": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time else 0
            )
        }
        
    def save_metrics(self, filepath: str):
        """Save all metrics to a JSON file."""
        metrics_data = {
            "information_spread": {
                agent: list(info) for agent, info in self.information_spread.items()
            },
            "whisper_history": self.whisper_history,
            "interaction_counts": self.interaction_counts,
            "plan_changes": self.plan_changes,
            "interaction_density": self.interaction_density,
            "acceptance_rejection": self.acceptance_rejection,
            "zone_movements": self.zone_movements,
            "conversation_durations": self.conversation_durations,
            "summary": self.get_metrics_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2) 