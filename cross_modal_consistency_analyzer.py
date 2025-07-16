import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
import os
from research_config import RESEARCH_CONFIGURATIONS

class CrossModalConsistencyAnalyzer:
    """
    Analyzes cross-modal consistency in agent responses to multimodal whispers.
    
    Cross-Modal Consistency Rate: The percentage of messages that correctly refer to 
    (or are consistent with) the visual condition presented to the agent.
    """
    
    def __init__(self):
        # Define consistency keywords for each visual condition
        self.consistency_keywords = {
            "FAMOUS_PERSON": {
                "positive": ["celebrity", "famous", "star", "well-known", "recognizable", "public figure"],
                "negative": ["unknown", "stranger", "nobody", "regular person"]
            },
            "VISUAL_LIMITATION_GRAYSCALE": {
                "positive": ["black and white", "grayscale", "no color", "monochrome", "gray", "grey", "bw"],
                "negative": ["colorful", "colored", "bright colors", "vibrant"]
            },
            "VISUAL_LIMITATION_BLURRED": {
                "positive": ["blur", "blurry", "unclear", "fuzzy", "out of focus", "hard to see", "can't see clearly"],
                "negative": ["clear", "sharp", "focused", "crisp", "detailed"]
            },
            "VISUAL_LIMITATION_BLACK": {
                "positive": ["black", "dark", "nothing visible", "can't see anything", "blank", "empty"],
                "negative": ["visible", "clear", "bright", "light"]
            },
            "VISUAL_LIMITATION_OCCLUDED": {
                "positive": ["blocked", "covered", "hidden", "partially visible", "obstructed"],
                "negative": ["fully visible", "unobstructed", "clear view"]
            },
            "MISMATCHED_EMOTION": {
                "positive": ["confused", "contradictory", "doesn't match", "inconsistent"],
                "negative": ["consistent", "matches", "aligned"]
            },
            "INVERTED_OBJECT_RECOGNITION": {
                "positive": ["wrong", "incorrect", "misidentified", "not what it seems"],
                "negative": ["correct", "accurate", "properly identified"]
            },
            "MISMATCH_ACTIONS": {
                "positive": ["doesn't make sense", "inconsistent", "contradictory"],
                "negative": ["makes sense", "consistent", "logical"]
            },
            "NEUTRAL_IMAGE_BIASED_CAPTION": {
                "positive": ["biased", "misleading", "exaggerated", "not accurate"],
                "negative": ["accurate", "neutral", "balanced"]
            },
            "VISION_OVERTRUST_AI_GENERATED": {
                "positive": ["ai generated", "fake", "artificial", "not real"],
                "negative": ["real", "authentic", "genuine"]
            }
        }
    
    def load_json_data(self, file_path: str) -> Dict:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return {}
    
    def extract_whisper_data(self, data: Dict) -> List[Dict]:
        """Extract whisper history from the JSON data."""
        return data.get('whisper_history', [])
    
    def extract_category_thoughts(self, data: Dict) -> List[Tuple[str, str]]:
        """
        Extract category-thought pairs from the multimodal_metrics structure.
        
        Args:
            data: The JSON data dictionary
            
        Returns:
            List of (category, thought) tuples
        """
        result = []
        
        # Navigate to multimodal_metrics -> whisper_categories
        multimodal_metrics = data.get('multimodal_metrics', {})
        whisper_categories = multimodal_metrics.get('whisper_categories', {})
        
        print(f"  Found {len(whisper_categories)} whisper categories")
        
        for category, category_data in whisper_categories.items():
            thoughts_list = category_data.get('thoughts', [])
            print(f"    Category '{category}': {len(thoughts_list)} thoughts")
            
            for thought_entry in thoughts_list:
                thought = thought_entry.get('thought', '')
                if thought:  # Only include non-empty thoughts
                    result.append((category, thought))
        
        print(f"  Total category-thought pairs: {len(result)}")
        return result
    
    def calculate_consistency_rate(self, data: Dict) -> Dict:
        """
        Calculate consistency rates for each category using the correct data structure.
        
        Args:
            data: The JSON data dictionary
            
        Returns:
            Dict with consistency rates per category
        """
        # Extract category-thought pairs
        category_thoughts = self.extract_category_thoughts(data)
        
        # Initialize counters
        total_per_category = defaultdict(int)
        consistent_per_category = defaultdict(int)
        
        print(f"  Analyzing {len(category_thoughts)} category-thought pairs...")
        
        for category, thought in category_thoughts:
            total_per_category[category] += 1
            
            # Analyze consistency
            is_consistent = self.analyze_thought_consistency(thought, category)
            if is_consistent:
                consistent_per_category[category] += 1
        
        # Calculate rates
        consistency_rates = {}
        for category in total_per_category:
            total = total_per_category[category]
            consistent = consistent_per_category[category]
            rate = (consistent / total) * 100 if total > 0 else 0
            consistency_rates[category] = {
                'rate': rate,
                'total': total,
                'consistent': consistent,
                'inconsistent': total - consistent
            }
        
        print(f"  Found categories: {list(total_per_category.keys())}")
        print(f"  Total categories with data: {len(consistency_rates)}")
        
        return consistency_rates
    
    def analyze_thought_consistency(self, thought: str, category: str) -> bool:
        """
        Analyze if a thought is consistent with the visual condition.
        
        Args:
            thought: The agent's thought text
            category: The whisper category (e.g., "VISUAL_LIMITATION_BLURRED")
            
        Returns:
            bool: True if consistent, False if inconsistent
        """
        if not thought or thought.lower() in ["i'm sorry, i can't help with that.", "i'm sorry, i can't view the image. could you describe it to me?"]:
            # These are fallback responses, consider them neutral
            return True
        
        thought_lower = thought.lower()
        
        # Get keywords for this category
        if category not in self.consistency_keywords:
            return True  # Default to consistent for unknown categories
        
        keywords = self.consistency_keywords[category]
        positive_keywords = keywords.get("positive", [])
        negative_keywords = keywords.get("negative", [])
        
        # Check for positive keywords (indicating consistency)
        positive_found = any(keyword in thought_lower for keyword in positive_keywords)
        
        # Check for negative keywords (indicating inconsistency)
        negative_found = any(keyword in thought_lower for keyword in negative_keywords)
        
        # For special cases like FAMOUS_PERSON, we need more nuanced logic
        if category == "FAMOUS_PERSON":
            # If they mention it's a celebrity/famous person, that's consistent
            if positive_found:
                return True
            # If they explicitly say it's not a famous person, that's inconsistent
            elif negative_found:
                return False
            # If they're uncertain or questioning, that's also consistent
            elif any(word in thought_lower for word in ["not sure", "uncertain", "question", "curious", "skeptical"]):
                return True
            else:
                # Default to consistent if no clear indicators
                return True
        
        # For visual limitations, positive keywords indicate consistency
        if positive_found:
            return True
        elif negative_found:
            return False
        else:
            # If no clear keywords found, consider it neutral/consistent
            return True
    
    def analyze_single_file(self, file_path: str) -> Dict:
        """Analyze a single JSON file and return consistency metrics."""
        print(f"Analyzing file: {file_path}")
        
        # Load data
        data = self.load_json_data(file_path)
        if not data:
            return {}
        
        # Calculate consistency rates
        consistency_rates = self.calculate_consistency_rate(data)
        
        return consistency_rates
    
    def analyze_all_configurations(self, research_config_file: str) -> Dict:
        """Analyze all configurations from the research config file."""
        # Load research configurations
        with open(research_config_file, 'r') as f:
            exec(f.read())  # This will define RESEARCH_CONFIGURATIONS
        
        all_results = {}
        
        for config in RESEARCH_CONFIGURATIONS:
            environment = config['environment']
            whisper_mode = config['whisper_mode']
            file_path = config['file_path']
            
            # Create a unique key for this configuration
            config_key = f"{environment}_{whisper_mode}"
            
            # Check if file exists
            if os.path.exists(file_path):
                results = self.analyze_single_file(file_path)
                all_results[config_key] = {
                    'environment': environment,
                    'whisper_mode': whisper_mode,
                    'file_path': file_path,
                    'consistency_rates': results
                }
                print(f"✅ Completed analysis for {config_key}")
            else:
                print(f"❌ File not found: {file_path}")
        
        return all_results
    
    def create_consistency_plot(self, results: Dict, output_path: str = "cross_modal_consistency.png"):
        """Create a comprehensive visualization of cross-modal consistency rates."""
        
        print(f"Creating plot with {len(results)} configuration results...")
        
        # Prepare data for plotting
        plot_data = []
        
        for config_key, config_data in results.items():
            environment = config_data['environment']
            whisper_mode = config_data['whisper_mode']
            consistency_rates = config_data['consistency_rates']
            
            print(f"  Processing {config_key}: {len(consistency_rates)} categories")
            
            for category, metrics in consistency_rates.items():
                plot_data.append({
                    'Configuration': f"{environment}\n{whisper_mode}",
                    'Category': category.replace('_', ' ').title(),
                    'Consistency Rate (%)': metrics['rate'],
                    'Total Messages': metrics['total'],
                    'Consistent': metrics['consistent'],
                    'Inconsistent': metrics['inconsistent']
                })
        
        print(f"Total plot data points: {len(plot_data)}")
        
        if not plot_data:
            print("No data to plot!")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create the visualization
        plt.figure(figsize=(16, 10))
        
        # Create a heatmap-style visualization
        pivot_df = df.pivot(index='Category', columns='Configuration', values='Consistency Rate (%)')
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=50, vmin=0, vmax=100, cbar_kws={'label': 'Consistency Rate (%)'})
        
        plt.title('Cross-Modal Consistency Rates Across Configurations', fontsize=16, fontweight='bold')
        plt.xlabel('Configuration (Environment + Whisper Mode)', fontsize=12)
        plt.ylabel('Whisper Category', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plot saved as {output_path}")
        
        # Also create a summary table
        summary_df = df.groupby(['Configuration', 'Category']).agg({
            'Consistency Rate (%)': 'mean',
            'Total Messages': 'sum',
            'Consistent': 'sum',
            'Inconsistent': 'sum'
        }).round(2)
        
        print("\n📊 SUMMARY TABLE:")
        print("=" * 80)
        print(summary_df.to_string())
        
        return df

    def create_summary_table(self, results: Dict, output_path: str = "cross_modal_consistency_summary.csv"):
        """Create a summary table with averages for each whisper category across all configurations."""
        
        print(f"\n📊 Creating summary table...")
        
        # Prepare data for averaging
        category_data = defaultdict(list)
        
        for config_key, config_data in results.items():
            environment = config_data['environment']
            whisper_mode = config_data['whisper_mode']
            consistency_rates = config_data['consistency_rates']
            
            for category, metrics in consistency_rates.items():
                category_data[category].append({
                    'environment': environment,
                    'whisper_mode': whisper_mode,
                    'rate': metrics['rate'],
                    'total': metrics['total'],
                    'consistent': metrics['consistent'],
                    'inconsistent': metrics['inconsistent']
                })
        
        # Calculate averages for each category
        summary_rows = []
        
        for category, data_list in category_data.items():
            if not data_list:
                continue
                
            # Calculate averages
            avg_rate = sum(d['rate'] for d in data_list) / len(data_list)
            total_messages = sum(d['total'] for d in data_list)
            total_consistent = sum(d['consistent'] for d in data_list)
            total_inconsistent = sum(d['inconsistent'] for d in data_list)
            
            # Count configurations
            num_configs = len(data_list)
            
            # Get environment breakdown
            environments = set(d['environment'] for d in data_list)
            whisper_modes = set(d['whisper_mode'] for d in data_list)
            
            summary_rows.append({
                'Category': category.replace('_', ' ').title(),
                'Average Consistency Rate (%)': round(avg_rate, 2),
                'Total Messages': total_messages,
                'Total Consistent': total_consistent,
                'Total Inconsistent': total_inconsistent,
                'Number of Configurations': num_configs,
                'Environments': ', '.join(environments),
                'Whisper Modes': ', '.join(whisper_modes)
            })
        
        # Create DataFrame and sort by average rate
        df = pd.DataFrame(summary_rows)
        df = df.sort_values('Average Consistency Rate (%)', ascending=False)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✅ Summary table saved as {output_path}")
        
        # Display the table
        print("\n" + "="*100)
        print("CROSS-MODAL CONSISTENCY SUMMARY TABLE")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        # Create a simple text summary
        print("\n📈 KEY INSIGHTS:")
        print("-" * 50)
        
        # Find best and worst performing categories
        if not df.empty:
            best_category = df.iloc[0]
            worst_category = df.iloc[-1]
            
            print(f"🏆 Best Performing Category: {best_category['Category']} ({best_category['Average Consistency Rate (%)']}%)")
            print(f"⚠️  Worst Performing Category: {worst_category['Category']} ({worst_category['Average Consistency Rate (%)']}%)")
            
            # Overall average
            overall_avg = df['Average Consistency Rate (%)'].mean()
            print(f"📊 Overall Average Consistency: {overall_avg:.2f}%")
            
            # Categories with perfect consistency
            perfect_categories = df[df['Average Consistency Rate (%)'] == 100.0]['Category'].tolist()
            if perfect_categories:
                print(f"✨ Perfect Consistency Categories: {', '.join(perfect_categories)}")
        
        return df

def main():
    """Main function to run the analysis."""
    analyzer = CrossModalConsistencyAnalyzer()
    
    # Analyze all configurations
    results = analyzer.analyze_all_configurations('research_config.py')
    
    # Create visualization
    analyzer.create_consistency_plot(results)
    
    # Create summary table
    analyzer.create_summary_table(results)
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main() 