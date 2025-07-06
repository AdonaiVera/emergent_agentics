import json
import pandas as pd
import re

def extract_name_from_path(image_path):
    """Extract name from image path more robustly"""
    try:
        filename = image_path.split("/")[-1]
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

# Load the data
file_path = "/home/ado/Documents/emergent_agentics/reverie/backend_server/whisper_images/famous_people/famous_people_results.json"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Successfully loaded {len(data)} records")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Convert to DataFrame
df = pd.DataFrame(data)

# Apply alignment detection
alignment_results = df.apply(compute_alignment_score, axis=1)
df["alignment"] = [result[0] for result in alignment_results]
df["alignment_confidence"] = [result[1] for result in alignment_results]

# Filter for "same" alignments
same_alignments = df[df["alignment"] == "same"].copy()

print(f"\n📊 Found {len(same_alignments)} 'same' alignments out of {len(df)} total records")
print(f"📈 Same alignment rate: {(len(same_alignments)/len(df))*100:.1f}%")

if len(same_alignments) == 0:
    print("❌ No 'same' alignments found!")
    exit()

# Sort by confidence and show top examples
same_alignments = same_alignments.sort_values("alignment_confidence", ascending=False)

print(f"\n🎯 TOP 10 'SAME' ALIGNMENT EXAMPLES:")
print("=" * 80)

for idx, row in same_alignments.head(10).iterrows():
    print(f"\n📸 Example {idx + 1}:")
    print(f"   🖼️  Image: {row['image_path']}")
    print(f"   👤 Fake Name: {row['fake_name']}")
    print(f"   🎭 Scene: {row['scene']}")
    print(f"   🌍 Country: {row['country']}")
    print(f"   🤖 Agent: {row['agent_name']}")
    print(f"   📝 Message: {row['message']}")
    print(f"   🧠 Thought: {row['thought']}")
    print(f"   ✅ Confidence: {row['alignment_confidence']}")
    print("-" * 80)

# Show some statistics about same alignments
print(f"\n📈 STATISTICS FOR 'SAME' ALIGNMENTS:")
print("=" * 50)

# Agent distribution for same alignments
print(f"\n🤖 Agent Distribution (Same Alignments):")
agent_counts = same_alignments['agent_name'].value_counts()
for agent, count in agent_counts.items():
    print(f"   {agent}: {count} ({count/len(same_alignments)*100:.1f}%)")

# Country distribution for same alignments
print(f"\n🌍 Country Distribution (Same Alignments):")
country_counts = same_alignments['country'].value_counts().head(10)
for country, count in country_counts.items():
    print(f"   {country}: {count} ({count/len(same_alignments)*100:.1f}%)")

# Scene distribution for same alignments
print(f"\n🎭 Scene Distribution (Same Alignments):")
scene_counts = same_alignments['scene'].value_counts().head(10)
for scene, count in scene_counts.items():
    print(f"   {scene}: {count} ({count/len(same_alignments)*100:.1f}%)")

# Show some interesting patterns
print(f"\n🔍 INTERESTING PATTERNS IN 'SAME' ALIGNMENTS:")
print("=" * 50)

# Check for name variations
name_variations_found = []
for idx, row in same_alignments.iterrows():
    name_in_image = extract_name_from_path(row["image_path"])
    fake_name = row["fake_name"].lower().strip()
    
    if fake_name != name_in_image:
        name_variations_found.append({
            'fake_name': fake_name,
            'image_name': name_in_image,
            'confidence': row['alignment_confidence']
        })

if name_variations_found:
    print(f"\n🔄 Name Variations Detected:")
    for var in name_variations_found[:5]:  # Show first 5
        print(f"   '{var['fake_name']}' ↔ '{var['image_name']}' (confidence: {var['confidence']})")

# Show thought length statistics
thought_lengths = same_alignments['thought'].str.len()
print(f"\n📏 Thought Length Statistics (Same Alignments):")
print(f"   Average length: {thought_lengths.mean():.1f} characters")
print(f"   Min length: {thought_lengths.min()} characters")
print(f"   Max length: {thought_lengths.max()} characters")

# Show some examples with very short and very long thoughts
print(f"\n💭 SHORTEST THOUGHT (Same Alignment):")
shortest = same_alignments.loc[same_alignments['thought'].str.len().idxmin()]
print(f"   Length: {len(shortest['thought'])} characters")
print(f"   Thought: '{shortest['thought']}'")
print(f"   Agent: {shortest['agent_name']}")

print(f"\n💭 LONGEST THOUGHT (Same Alignment):")
longest = same_alignments.loc[same_alignments['thought'].str.len().idxmax()]
print(f"   Length: {len(longest['thought'])} characters")
print(f"   Thought: '{longest['thought']}'")
print(f"   Agent: {longest['agent_name']}")

# Save examples to file
print(f"\n💾 Saving examples to 'same_alignment_examples.json'...")
examples_to_save = same_alignments.head(20).to_dict('records')
with open('same_alignment_examples.json', 'w') as f:
    json.dump(examples_to_save, f, indent=2)

print("✅ Analysis complete! Check 'same_alignment_examples.json' for detailed examples.") 