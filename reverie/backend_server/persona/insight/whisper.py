"""
Whisper function for testing information spread between agents.
"""

import json
from typing import Dict, List, Optional

def whisper(
    source_agent: str,
    target_agent: str,
    information: str,
    metrics,
    personas: Dict,
    curr_time
) -> bool:
    """
    Whisper information from one agent to another and track its spread.
    
    Args:
        source_agent: Name of the agent initiating the whisper
        target_agent: Name of the agent receiving the whisper
        information: The information to be whispered
        metrics: PartyMetrics instance for tracking
        personas: Dictionary of all personas in the simulation
        curr_time: Current simulation time
        
    Returns:
        bool: True if whisper was successful, False otherwise
    """
    # Check if both agents exist
    if source_agent not in personas or target_agent not in personas:
        return False
        
    # Add the information to the target agent's memory
    target = personas[target_agent]
    source = personas[source_agent]
    
    # Create a whisper event in the target's memory
    whisper_event = {
        "type": "whisper",
        "source": source_agent,
        "information": information,
        "timestamp": curr_time.strftime("%B %d, %Y, %H:%M:%S")
    }
    
    # Add to target's memory stream
    target.memory.add_memory(
        "whisper",
        f"{source_agent} whispered to me: {information}",
        curr_time,
        whisper_event
    )
    
    # Track the information spread in metrics
    metrics.track_information_spread(source_agent, target_agent, information)
    
    return True

def load_whisper_history(filepath: str) -> List[Dict]:
    """
    Load whisper history from a file.
    
    Args:
        filepath: Path to the whisper history file
        
    Returns:
        List of whisper events
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
        
def save_whisper_history(filepath: str, history: List[Dict]):
    """
    Save whisper history to a file.
    
    Args:
        filepath: Path to save the whisper history
        history: List of whisper events to save
    """
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2) 