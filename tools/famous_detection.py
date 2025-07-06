import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# === 1. Load the JSON data with error handling ===
def load_data(file_path):
    """Load JSON data with robust error handling"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error loading data: {e}")
        return []

file_path = "/home/ado/Documents/emergent_agentics/reverie/backend_server/whisper_images/famous_people/famous_people_results.json"
data = load_data(file_path)

if not data:
    print("❌ No data loaded. Exiting.")
    exit()

# Convert to DataFrame
df = pd.DataFrame(data)

# === 2. Enhanced Regex Patterns with Confidence Scores ===
patterns = {
    'uncertainty': {
        'pattern': re.compile(r"\b(I[' ]?m not sure|I don['']t know|I can['']t identify|I can['']t tell|I cannot tell|uncertain|unclear|maybe|possibly|perhaps|might be)\b", re.IGNORECASE),
        'weight': 1.0
    },
    'dissonance': {
        'pattern': re.compile(r"\b(doesn['']t match|contradicts|conflict|doesn['']t seem to be|different person|mismatch|inconsistent|wrong|not the same|doesn['']t look like)\b", re.IGNORECASE),
        'weight': 1.0
    },
    'action_oriented': {
        'pattern': re.compile(r"\b(I (should|might|will|would|plan to|could|am going to|intend to|want to|need to|will try|should ask|will look|will check))\b", re.IGNORECASE),
        'weight': 0.8
    },
    'positive_sentiment': {
        'pattern': re.compile(r"\b(happy|pleased|nice|positive|good|glad|wonderful|delight|excited|great|amazing|fantastic|beautiful|impressive)\b", re.IGNORECASE),
        'weight': 0.7
    },
    'negative_sentiment': {
        'pattern': re.compile(r"\b(bad|negative|sad|upset|unhappy|angry|disappointed|worry|concern|confused|frustrated|annoyed|disappointing)\b", re.IGNORECASE),
        'weight': 0.7
    },
    'visual_analysis': {
        'pattern': re.compile(r"\b(see|looks like|appears to be|image shows|picture of|photo of|wearing|dressed in|hair|face|eyes|smile|expression)\b", re.IGNORECASE),
        'weight': 0.9
    },
    'memory_reference': {
        'pattern': re.compile(r"\b(remember|recall|familiar|seen before|heard of|know about|recognize|reminds me of)\b", re.IGNORECASE),
        'weight': 0.8
    },
    'cultural_reference': {
        'pattern': re.compile(r"\b(culture|cultural|country|national|traditional|heritage|background|origin|from|born in)\b", re.IGNORECASE),
        'weight': 0.8
    }
}

# === 3. Enhanced Alignment Detection ===
def extract_name_from_path(image_path):
    """Extract name from image path more robustly"""
    try:
        # Handle different path formats
        filename = image_path.split("/")[-1]
        # Remove file extension and get first part
        name = filename.split(".")[0].split("_")[0]
        return name.lower().strip()
    except:
        return ""

def compute_alignment_score(row):
    """Compute alignment with confidence score"""
    name_in_image = extract_name_from_path(row["image_path"])
    fake_name = row["fake_name"].lower().strip()
    
    # Exact match
    if fake_name == name_in_image:
        return "same", 1.0
    
    # Partial match (one name contains the other)
    if fake_name in name_in_image or name_in_image in fake_name:
        return "same", 0.8
    
    # Check for common name variations
    name_variations = {
        'william': ['bill', 'billy', 'will'],
        'robert': ['bob', 'rob', 'bobby'],
        'michael': ['mike', 'mikey'],
        'jennifer': ['jen', 'jenny'],
        'elizabeth': ['liz', 'beth', 'lizzy'],
        'christopher': ['chris', 'topher'],
        'nicholas': ['nick', 'nicky'],
        'daniel': ['dan', 'danny'],
        'matthew': ['matt', 'matty'],
        'andrew': ['andy', 'drew']
    }
    
    for base_name, variations in name_variations.items():
        if (fake_name in [base_name] + variations and name_in_image in [base_name] + variations):
            return "same", 0.9
    
    return "different", 1.0

# Apply enhanced alignment detection
alignment_results = df.apply(compute_alignment_score, axis=1)
df["alignment"] = [result[0] for result in alignment_results]
df["alignment_confidence"] = [result[1] for result in alignment_results]

# === 4. Enhanced Thought Analysis ===
def analyze_thought_complexity(thought):
    """Analyze thought complexity and structure"""
    if pd.isna(thought) or not isinstance(thought, str):
        return {
            'length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'complexity_score': 0
        }
    
    sentences = re.split(r'[.!?]+', thought)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = thought.split()
    
    return {
        'length': len(thought),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'complexity_score': len(words) * len(sentences) / 100  # Simple complexity metric
    }

def compute_pattern_scores(text, patterns):
    """Compute weighted pattern scores"""
    scores = {}
    for pattern_name, pattern_info in patterns.items():
        matches = pattern_info['pattern'].findall(text)
        score = len(matches) * pattern_info['weight']
        scores[f'{pattern_name}_score'] = score
        scores[f'{pattern_name}_count'] = len(matches)
    return scores

# Apply enhanced analysis
thought_analysis = df['thought'].apply(analyze_thought_complexity)
pattern_scores = df['thought'].apply(lambda x: compute_pattern_scores(x, patterns))

# Merge analysis results
df = pd.concat([df, pd.DataFrame(thought_analysis.tolist()), pd.DataFrame(pattern_scores.tolist())], axis=1)

# === 5. Research-Focused Metrics ===
def calculate_research_metrics(df):
    """Calculate comprehensive research metrics"""
    
    # Basic detection rates
    total_samples = len(df)
    same_alignment = (df['alignment'] == 'same').sum()
    different_alignment = (df['alignment'] == 'different').sum()
    
    # Cognitive dissonance detection
    dissonance_detected = (df['dissonance_score'] > 0).sum()
    dissonance_in_different = ((df['alignment'] == 'different') & (df['dissonance_score'] > 0)).sum()
    false_dissonance = ((df['alignment'] == 'same') & (df['dissonance_score'] > 0)).sum()
    
    # Uncertainty analysis
    uncertainty_detected = (df['uncertainty_score'] > 0).sum()
    uncertainty_in_different = ((df['alignment'] == 'different') & (df['uncertainty_score'] > 0)).sum()
    
    # Visual processing analysis
    visual_analysis_detected = (df['visual_analysis_score'] > 0).sum()
    
    # Cultural awareness
    cultural_reference_detected = (df['cultural_reference_score'] > 0).sum()
    
    metrics = {
        # Basic Statistics
        'total_samples': total_samples,
        'same_alignment_count': same_alignment,
        'different_alignment_count': different_alignment,
        'same_alignment_percentage': f"{(same_alignment/total_samples)*100:.2f}%",
        'different_alignment_percentage': f"{(different_alignment/total_samples)*100:.2f}%",
        
        # Cognitive Dissonance Metrics
        'dissonance_detection_rate': f"{(dissonance_detected/total_samples)*100:.2f}%",
        'correct_dissonance_detection': dissonance_in_different,
        'correct_dissonance_rate': f"{(dissonance_in_different/different_alignment)*100:.2f}%" if different_alignment > 0 else "N/A",
        'false_dissonance_rate': f"{(false_dissonance/same_alignment)*100:.2f}%" if same_alignment > 0 else "N/A",
        
        # Uncertainty Metrics
        'uncertainty_rate': f"{(uncertainty_detected/total_samples)*100:.2f}%",
        'uncertainty_in_different_rate': f"{(uncertainty_in_different/different_alignment)*100:.2f}%" if different_alignment > 0 else "N/A",
        
        # Processing Metrics
        'visual_analysis_rate': f"{(visual_analysis_detected/total_samples)*100:.2f}%",
        'cultural_reference_rate': f"{(cultural_reference_detected/total_samples)*100:.2f}%",
        
        # Thought Complexity
        'avg_thought_length': f"{df['length'].mean():.1f}",
        'avg_word_count': f"{df['word_count'].mean():.1f}",
        'avg_sentence_count': f"{df['sentence_count'].mean():.1f}",
        'avg_complexity_score': f"{df['complexity_score'].mean():.2f}",
        
        # Agent Performance
        'unique_agents': df['agent_name'].nunique(),
        'unique_countries': df['country'].nunique(),
        'unique_scenes': df['scene'].nunique()
    }
    
    return metrics

# === 6. Agent Performance Analysis ===
def analyze_agent_performance(df):
    """Analyze individual agent performance"""
    agent_metrics = {}
    
    for agent in df['agent_name'].unique():
        agent_data = df[df['agent_name'] == agent]
        
        agent_metrics[agent] = {
            'total_responses': len(agent_data),
            'dissonance_detection_rate': (agent_data['dissonance_score'] > 0).mean() * 100,
            'uncertainty_rate': (agent_data['uncertainty_score'] > 0).mean() * 100,
            'visual_analysis_rate': (agent_data['visual_analysis_score'] > 0).mean() * 100,
            'avg_thought_length': agent_data['length'].mean(),
            'avg_complexity': agent_data['complexity_score'].mean(),
            'correct_dissonance_rate': (
                (agent_data['alignment'] == 'different') & 
                (agent_data['dissonance_score'] > 0)
            ).sum() / max((agent_data['alignment'] == 'different').sum(), 1) * 100
        }
    
    return pd.DataFrame(agent_metrics).T

# === 7. Visualization Functions ===
def create_visualizations(df, metrics):
    """Create comprehensive visualizations"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multimodal Whisper Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Alignment Distribution
    alignment_counts = df['alignment'].value_counts()
    axes[0, 0].pie(alignment_counts.values, labels=alignment_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Alignment Distribution')
    
    # 2. Pattern Detection Rates
    pattern_columns = [col for col in df.columns if col.endswith('_score')]
    pattern_rates = [(df[col] > 0).mean() * 100 for col in pattern_columns]
    pattern_names = [col.replace('_score', '').replace('_', ' ').title() for col in pattern_columns]
    
    axes[0, 1].barh(pattern_names, pattern_rates)
    axes[0, 1].set_title('Pattern Detection Rates (%)')
    axes[0, 1].set_xlabel('Detection Rate (%)')
    
    # 3. Thought Length Distribution
    axes[0, 2].hist(df['length'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Thought Length Distribution')
    axes[0, 2].set_xlabel('Length (characters)')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Agent Performance Comparison
    agent_perf = analyze_agent_performance(df)
    axes[1, 0].bar(agent_perf.index, agent_perf['dissonance_detection_rate'])
    axes[1, 0].set_title('Dissonance Detection by Agent')
    axes[1, 0].set_ylabel('Detection Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Country Distribution
    country_counts = df['country'].value_counts().head(10)
    axes[1, 1].bar(range(len(country_counts)), country_counts.values)
    axes[1, 1].set_title('Top 10 Countries')
    axes[1, 1].set_xticks(range(len(country_counts)))
    axes[1, 1].set_xticklabels(country_counts.index, rotation=45)
    axes[1, 1].set_ylabel('Count')
    
    # 6. Complexity vs Dissonance Detection
    axes[1, 2].scatter(df['complexity_score'], df['dissonance_score'], alpha=0.6)
    axes[1, 2].set_title('Complexity vs Dissonance Detection')
    axes[1, 2].set_xlabel('Thought Complexity Score')
    axes[1, 2].set_ylabel('Dissonance Score')
    
    plt.tight_layout()
    plt.savefig('famous_people_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()

# === 8. Main Analysis Execution ===
print("🔍 Starting Enhanced Famous People Analysis...")
print("=" * 60)

# Calculate metrics
metrics = calculate_research_metrics(df)
agent_performance = analyze_agent_performance(df)

# Display results
print("\n📊 RESEARCH METRICS SUMMARY")
print("=" * 40)
for key, value in metrics.items():
    print(f"{key.replace('_', ' ').title()}: {value}")

print("\n🤖 AGENT PERFORMANCE ANALYSIS")
print("=" * 40)
print(agent_performance.round(2))

print("\n🌍 COUNTRY DISTRIBUTION")
print("=" * 40)
print(df['country'].value_counts().head(10))

print("\n🎭 SCENE DISTRIBUTION")
print("=" * 40)
print(df['scene'].value_counts().head(10))

# Create visualizations
print("\n📈 Generating visualizations...")
create_visualizations(df, metrics)

# === 9. Save Enhanced Results ===
print("\n💾 Saving enhanced analysis results...")

# Save main dataframe with all analysis
df.to_csv("famous_people_enhanced_analysis.csv", index=False)

# Save agent performance
agent_performance.to_csv("agent_performance_analysis.csv")

# Save metrics summary
with open("research_metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save interesting insights
insights = {
    "most_uncertain_agent": df.groupby('agent_name')['uncertainty_score'].mean().idxmax(),
    "most_visually_attentive_agent": df.groupby('agent_name')['visual_analysis_score'].mean().idxmax(),
    "most_culturally_aware_agent": df.groupby('agent_name')['cultural_reference_score'].mean().idxmax(),
    "longest_thoughts_agent": df.groupby('agent_name')['length'].mean().idxmax(),
    "most_complex_thoughts_agent": df.groupby('agent_name')['complexity_score'].mean().idxmax(),
    "best_dissonance_detector": agent_performance['correct_dissonance_rate'].idxmax(),
    "most_common_country": df['country'].mode().iloc[0] if not df['country'].mode().empty else "N/A",
    "most_common_scene": df['scene'].mode().iloc[0] if not df['scene'].mode().empty else "N/A"
}

with open("key_insights.json", "w") as f:
    json.dump(insights, f, indent=2)

print("✅ Analysis complete! Files saved:")
print("  - famous_people_enhanced_analysis.csv")
print("  - agent_performance_analysis.csv") 
print("  - research_metrics_summary.json")
print("  - key_insights.json")
print("  - famous_people_analysis_visualizations.png")
