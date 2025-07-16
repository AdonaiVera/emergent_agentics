#!/usr/bin/env python3
"""
Test script for the improved plot metrics with research config support

This script demonstrates how to use the new functionality to load experiments
from the research config file and generate plots.
"""

import os
import sys
from plot_metrics import load_research_configurations, filter_configurations, load_experiments_from_config
from tools.produce_plots import PlotGenerator

def test_research_config_loading():
    """Test loading configurations from research config file"""
    print("Testing research config loading...")
    
    # Load configurations
    configurations = load_research_configurations()
    
    if not configurations:
        print("❌ No configurations found in research_config.py")
        return False
    
    print(f"✅ Found {len(configurations)} configurations")
    
    # Print configuration details
    for i, config in enumerate(configurations):
        print(f"  {i+1}. {config.get('title', 'Unknown')}")
        print(f"     Environment: {config.get('environment', 'Unknown')}")
        print(f"     Whisper Mode: {config.get('whisper_mode', 'Unknown')}")
        print(f"     File: {config.get('file_path', 'Unknown')}")
        print()
    
    return True

def test_configuration_filtering():
    """Test filtering configurations"""
    print("Testing configuration filtering...")
    
    configurations = load_research_configurations()
    if not configurations:
        print("❌ No configurations to filter")
        return False
    
    # Test environment filtering
    karaoke_configs = filter_configurations(configurations, environments=["Karaoke Night"])
    print(f"✅ Found {len(karaoke_configs)} Karaoke Night configurations")
    
    medical_configs = filter_configurations(configurations, environments=["Medical Waiting Room"])
    print(f"✅ Found {len(medical_configs)} Medical Waiting Room configurations")
    
    # Test whisper mode filtering
    one_agent_configs = filter_configurations(configurations, whisper_modes=["1 Agent"])
    print(f"✅ Found {len(one_agent_configs)} 1 Agent configurations")
    
    # Test whisper count filtering
    count_1_configs = filter_configurations(configurations, whisper_counts=[1])
    print(f"✅ Found {len(count_1_configs)} configurations with whisper count 1")
    
    return True

def test_experiment_loading():
    """Test loading experiments from configurations"""
    print("Testing experiment loading...")
    
    configurations = load_research_configurations()
    if not configurations:
        print("❌ No configurations to load experiments from")
        return False
    
    # Load experiments (skip file check for testing)
    experiments = load_experiments_from_config(configurations, check_files=False)
    
    if experiments:
        print(f"✅ Successfully loaded {len(experiments)} experiments")
        for exp in experiments:
            print(f"  - {exp.title}: {len(exp.dataframes)} dataframes, {exp.min_steps} steps")
    else:
        print("⚠️  No experiments loaded (this is expected if files don't exist)")
    
    return True

def test_plot_generator():
    """Test creating plot generator from research config"""
    print("Testing plot generator creation...")
    
    configurations = load_research_configurations()
    if not configurations:
        print("❌ No configurations to test with")
        return False
    
    # Load experiments (skip file check for testing)
    experiments = load_experiments_from_config(configurations, check_files=False)
    
    if not experiments:
        print("⚠️  No experiments to create generator with")
        return True  # This is not an error, just no data
    
    # Create plot generator
    generator = PlotGenerator(experiments)
    print(f"✅ Created plot generator with {len(generator.experiments)} experiments")
    print(f"   Minimum steps across experiments: {generator.min_steps}")
    
    return True

def main():
    """Run all tests"""
    print("Testing Improved Plot Metrics with Research Config Support")
    print("=" * 60)
    
    tests = [
        ("Research Config Loading", test_research_config_loading),
        ("Configuration Filtering", test_configuration_filtering),
        ("Experiment Loading", test_experiment_loading),
        ("Plot Generator Creation", test_plot_generator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The research config integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\nUsage Examples:")
    print("1. Generate all plots for all experiments:")
    print("   python plot_metrics.py --config research_config.py")
    print()
    print("2. Generate plots for specific environments:")
    print("   python plot_metrics.py --config research_config.py --environments 'Karaoke Night'")
    print()
    print("3. Generate specific plot types:")
    print("   python plot_metrics.py --config research_config.py --plots conversation plans")

if __name__ == "__main__":
    main() 