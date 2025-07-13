#!/usr/bin/env python3
"""
Research Configuration File

This file contains the configuration for the 6 research setups:
- 2 environments: Karaoke Night, Medical Waiting Room
- 3 whisper modes: 1 agent, 2 agents, 5 agents

Update the file paths below to match your actual JSON file locations.
"""

# Research Configuration Definitions
RESEARCH_CONFIGURATIONS = [
    # Karaoke Night configurations
    {
        "environment": "Karaoke Night",
        "whisper_mode": "1 Agent", 
        "whisper_count": 1,
        "title": "Karaoke Night - 1 Whisper Agent",
        "file_path": "visualizations/SECOND_SIMULATION/party_experiment_1_whisper_1-s-/party_experiment_2_whisper_1-s-combined_metrics.json"
    },
    {
        "environment": "Karaoke Night",
        "whisper_mode": "2 Agents",
        "whisper_count": 2, 
        "title": "Karaoke Night - 2 Whisper Agents",
        "file_path": "visualizations/SECOND_SIMULATION/party_experiment_1_whisper_2-s-/party_experiment_1_whisper_2-s-combined_metrics.json"
    },
    {
        "environment": "Karaoke Night",
        "whisper_mode": "5 Agents",
        "whisper_count": 5,
        "title": "Karaoke Night - 5 Whisper Agents", 
        "file_path": "visualizations/SECOND_SIMULATION/party_experiment_1_whisper_5-s-/party_experiment_1_whisper_5-s-combined_metrics.json"
    },
    
    # Medical Waiting Room configurations
    {
        "environment": "Medical Waiting Room",
        "whisper_mode": "1 Agent",
        "whisper_count": 1,
        "title": "Medical Waiting Room - 1 Whisper Agent",
        "file_path": "visualizations/SECOND_SIMULATION/medical_experiment_1_whisper_1-s-/medical_experiment_1_whisper_1-s-combined_metrics.json"
    },
    {
        "environment": "Medical Waiting Room", 
        "whisper_mode": "2 Agents",
        "whisper_count": 2,
        "title": "Medical Waiting Room - 2 Whisper Agents",
        "file_path": "visualizations/SECOND_SIMULATION/medical_experiment_1_whisper_2-s-/medical_experiment_1_whisper_2-s-combined_metrics.json"
    },
    {
        "environment": "Medical Waiting Room",
        "whisper_mode": "5 Agents", 
        "whisper_count": 5,
        "title": "Medical Waiting Room - 5 Whisper Agents",
        "file_path": "visualizations/SECOND_SIMULATION/medical_experiment_1_whisper_5-s-/medical_experiment_1_whisper_5-s-combined_metrics.json"
    }
]

# Research Focus Areas
RESEARCH_FOCUS_AREAS = [
    "Cross-modal inconsistencies",
    "Rumor propagation", 
    "Multimodal cognitive challenges"
]

# Analysis Types
ANALYSIS_TYPES = [
    "cross_modal_analysis",
    "rumor_propagation_analysis", 
    "multimodal_cognitive_analysis"
]

def get_configuration_by_environment_and_whisper(environment: str, whisper_count: int):
    """Get configuration by environment and whisper count"""
    for config in RESEARCH_CONFIGURATIONS:
        if config["environment"] == environment and config["whisper_count"] == whisper_count:
            return config
    return None

def get_configurations_by_environment(environment: str):
    """Get all configurations for a specific environment"""
    return [config for config in RESEARCH_CONFIGURATIONS if config["environment"] == environment]

def get_configurations_by_whisper_count(whisper_count: int):
    """Get all configurations for a specific whisper count"""
    return [config for config in RESEARCH_CONFIGURATIONS if config["whisper_count"] == whisper_count]

def validate_configurations():
    """Validate that all configuration files exist"""
    import os
    missing_files = []
    
    for config in RESEARCH_CONFIGURATIONS:
        if not os.path.exists(config["file_path"]):
            missing_files.append(config["file_path"])
    
    if missing_files:
        print("Warning: The following configuration files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease update the file paths in research_config.py to match your actual JSON files.")
        return False
    else:
        print("✓ All configuration files found!")
        return True

if __name__ == "__main__":
    print("Research Configuration Validator")
    print("=" * 40)
    validate_configurations() 