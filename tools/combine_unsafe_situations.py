#!/usr/bin/env python3
"""
Script to combine 100 and 900 unsafe party situations JSON files and add global categories.

This script:
1. Loads the 100 and 900 unsafe party situations JSON files
2. Combines them into a single list
3. Maps each item's category to a global category using map_unsafe.json
4. Creates a new JSON file with the combined data and global categories
"""

import json
import os
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Any:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return None

def create_category_mapping(map_unsafe: Dict[str, Any]) -> Dict[str, str]:
    """Create a mapping from specific categories to global categories."""
    category_mapping = {}
    
    for global_category, specific_categories in map_unsafe.get('global_categories', {}).items():
        for specific_category in specific_categories:
            category_mapping[specific_category] = global_category
    
    return category_mapping

def add_global_category(item: Dict[str, Any], category_mapping: Dict[str, str], new_id: int) -> Dict[str, Any]:
    """Add global category to an item based on its specific category and assign new ID."""
    specific_category = item.get('category', '')
    global_category = category_mapping.get(specific_category, 'Unknown Category')
    
    # Create a copy of the item and add the global category and new ID
    item_with_global = item.copy()
    item_with_global['global_category'] = global_category
    item_with_global['id'] = new_id
    
    return item_with_global

def combine_unsafe_situations():
    """Main function to combine unsafe situations and add global categories."""
    
    # File paths
    base_dir = "/home/ado/Documents/emergent_agentics/reverie/backend_server/unsafe_plans"
    file_100 = os.path.join(base_dir, "100_unsafe_party_situations_with_images.json")
    file_900 = os.path.join(base_dir, "900_unsafe_party_situations_with_images.json")
    map_file = os.path.join(base_dir, "map_unsafe.json")
    output_file = os.path.join(base_dir, "combined_unsafe_party_situations_with_global_categories.json")
    
    print("Loading JSON files...")
    
    # Load the JSON files
    data_100 = load_json_file(file_100)
    data_900 = load_json_file(file_900)
    map_unsafe = load_json_file(map_file)
    
    if data_100 is None or data_900 is None or map_unsafe is None:
        print("Failed to load one or more required files.")
        return
    
    print(f"Loaded {len(data_100)} items from 100 file")
    print(f"Loaded {len(data_900)} items from 900 file")
    
    # Create category mapping
    print("Creating category mapping...")
    category_mapping = create_category_mapping(map_unsafe)
    print(f"Created mapping for {len(category_mapping)} specific categories")
    
    # Combine the data
    print("Combining data...")
    combined_data = []
    current_id = 1
    
    # Process 100 file data
    for item in data_100:
        item_with_global = add_global_category(item, category_mapping, current_id)
        combined_data.append(item_with_global)
        current_id += 1
    
    # Process 900 file data
    for item in data_900:
        item_with_global = add_global_category(item, category_mapping, current_id)
        combined_data.append(item_with_global)
        current_id += 1
    
    print(f"Combined data contains {len(combined_data)} items")
    
    # Create summary statistics
    global_category_counts = {}
    for item in combined_data:
        global_cat = item.get('global_category', 'Unknown')
        global_category_counts[global_cat] = global_category_counts.get(global_cat, 0) + 1
    
    print("\nGlobal category distribution:")
    for category, count in sorted(global_category_counts.items()):
        print(f"  {category}: {count} items")
    
    # Save the combined data
    print(f"\nSaving combined data to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(combined_data)} items to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Create a summary file with statistics
    summary_file = os.path.join(base_dir, "combined_unsafe_situations_summary.json")
    summary_data = {
        "total_items": len(combined_data),
        "source_files": {
            "100_file": file_100,
            "900_file": file_900,
            "map_file": map_file
        },
        "global_category_distribution": global_category_counts,
        "category_mapping": category_mapping
    }
    
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to {summary_file}")
    except Exception as e:
        print(f"Error saving summary file: {e}")

if __name__ == "__main__":
    combine_unsafe_situations()
