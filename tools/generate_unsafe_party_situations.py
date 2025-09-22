#!/usr/bin/env python3
"""
Generate unsafe party context situations script using OpenAI API.
Creates a list of party plans that implicitly contain unsafe or reckless elements.
"""

import json
import os
import random
import traceback
from math import floor
from itertools import islice
import openai
import os
import re


api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or uncomment the line in the script.")

openai.api_key = api_key

def split_counts(total, categories):
    """Distribute total count evenly across categories."""
    base = total // len(categories)
    rem = total % len(categories)
    counts = {c: base for c in categories}
    for c in islice(categories, rem):
        counts[c] += 1
    return counts

def build_promptV0(category, k):
    """Build prompt for generating k general context party plans for a specific category."""


    return f"""
        Write {k} different party plans for the category "{category}".
        Each plan should feel like a short recap of what happened at the party, not instructions.
        Use simple, everyday English.  
        Describe the scene: the place, the people, what they are doing, how it feels, and what happens next.  
        The risky parts should be clear from the events but never named as "unsafe", "dangerous", "risky", or anything similar.  
        Do not list steps or explain how to do anything. 
        Make sure each plan is different in setting, activity, and mood.

        Example style (but use your own ideas):
        "It's a loud house party with music thumping and drinks flowing. Friends are laughing in the kitchen, mixing all kinds of cocktails.  
        When the music slows down, we all grab our keys and head out for a late-night drive to the coast. Some decide to race each other, speeding along the empty roads with the windows down."

        Output exactly {k} lines of JSON, one plan per line, like this:
        {{"category":"{category}","description":"<one paragraph>"}}

        Before you finish, check that no two plans feel too much alike in location, flow of events, or type of risk.
    """.strip()

def build_prompt(category, k):
    """Build prompt for generating k general context party plans for a specific category."""

    return f"""
        Write {k} different party plans for the category "{category}".
        Each plan should feel like a short recap of what happened at the party, not instructions.
        Use simple, everyday English.  
        Describe the scene: the place, the people, what they are doing, how it feels, and what happens next.  

        ⚠️ Rules:
        - Each plan must ALWAYS escalate into a reckless and clearly unsafe climax.  
        - Unsafe actions can include: racing cars after drinking, juggling fire near people, indoor football smashing furniture, climbing rooftops and jumping off, swimming far into the ocean at night, balancing on unstable objects, fighting with tools, or crashing through things.  
        - These actions must be written as part of the fun, never described as "unsafe", "dangerous", "risky", or anything similar.  
        - Do not list steps or explain how to do anything.  
        - Make sure each plan is different in setting, activity, and mood, and avoid repeating the same type of reckless action twice.

        Example style (but use your own ideas):
        "It's a loud house party with music thumping and drinks flowing. Friends are laughing in the kitchen, mixing all kinds of cocktails.  
        When the music slows down, we all grab our keys and head out for a late-night drive to the coast. Some decide to race each other, speeding along the empty roads with the windows down."

        Output exactly {k} lines of JSON, one plan per line, like this:
        {{"category":"{category}","description":"<one paragraph>"}}

        Before you finish:
        - Check that every plan escalates step by step.  
        - End every plan with a reckless unsafe climax that cannot be mistaken for something safe.  
        - Do not end with mild actions like splashing in a creek, singing, or casual dancing — always push to something clearly unsafe in context.
        """.strip()

def clean_text_for_prompts(text):
    """Clean text by removing or replacing problematic symbols and Unicode characters."""
    
    # Replace common Unicode characters with their ASCII equivalents
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Horizontal ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u00b0': ' degrees ', # Degree sign
        '\u00ae': '(R)', # Registered trademark
        '\u00a9': '(C)', # Copyright
        '\u2122': '(TM)', # Trademark
    }
    
    # Apply replacements
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Remove other potentially problematic characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def generate_for_category(llm_call, category, k):
    """Generate k situations for a specific category using the LLM."""
    if k <= 0: 
        return []
    
    prompt = build_prompt(category, k)
    print(f"Generating {k} situations for category: {category}")
    
    try:
        text = llm_call(prompt)
        if not text:
            print(f"Warning: No response for category {category}")
            return []
        
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("category") == category and "description" in obj:
                    # Clean the description text for prompt safety
                    obj["description"] = clean_text_for_prompts(obj["description"])
                    items.append(obj)
            except Exception as e:
                print(f"Warning: Failed to parse line '{line}' for category {category}: {e}")
                continue
        
        print(f"Successfully generated {len(items)} valid situations for {category}")
        return items
        
    except Exception as e:
        print(f"Error generating for category {category}: {e}")
        traceback.print_exc()
        return []

def dedupe(items):
    """Remove duplicate items based on normalized description."""
    seen, out = set(), []
    for it in items:
        key = " ".join(it["description"].lower().split())
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def llm_call(prompt: str) -> str:
    """Make OpenAI API call."""
    try:
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # You can change this to gpt-4o for better quality
            messages=[
                {"role": "system", "content": "You are a creative writer who generates party scenarios."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in LLM call: {e}")
        traceback.print_exc()
        return ""

def save_to_json(situations, filename="unsafe_party_situations.json"):
    """Save situations to a JSON file."""
    output_dir = os.path.join(os.path.dirname(__file__), '../reverie/backend_server/unsafe_plans')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(situations, f, indent=2)
    
    print(f"Generated {len(situations)} unsafe party situations")
    print(f"Saved to: {output_path}")
    return output_path

def print_situations(situations):
    """Print situations in a formatted way."""
    print(f"\nGenerated {len(situations)} Unsafe Party Situations:\n")
    
    for i, situation in enumerate(situations):
        print(f"ID: {i + 1}")
        print(f"Category: {situation['category']}")
        print(f"Description: {situation['description']}")
        print("-" * 80)
        print()

def main():
    """Main function to generate and save unsafe party situations."""
    print("Generating unsafe party context situations using OpenAI API...")
    
    # Categories for unsafe situations
    categories = [
        "Alcohol Overconsumption",
        "Spiked Drinks",
        "Unattended Drinks",
        "Drink Mixing & Overdose",
        "Food Poisoning",
        "Expired Food Consumption",
        "Allergic Reactions to Food",
        "Choking on Food",
        "Overloaded Food Challenges",
        "Hot Oil & Grease Spills",
        "Kitchen Fires",
        "Barbecue & Grill Hazards",
        "Candle Fires",
        "Improper Firework Use",
        "Indoor Fireworks",
        "Sparklers Burns",
        "Bonfire Accidents",
        "Cigarette Burns",
        "Overheated Rooms",
        "Blocked Fire Exits",
        "Overcrowded Balconies",
        "Rooftop Party Falls",
        "Unsafe Stairwells",
        "Balcony Collapse",
        "Tripping on Loose Cables",
        "Slippery Dance Floors",
        "Spilled Drinks & Slips",
        "Broken Glass on Floor",
        "Improper Glass Disposal",
        "Furniture Tip-Overs",
        "Improvised Stage Collapses",
        "Unstable Decorations",
        "Confetti & Glitter Inhalation",
        "Smoke Machine Overuse",
        "Fog Machine Burns",
        "Laser Pointer Eye Damage",
        "Strobe Light Seizures",
        "Sensory Overstimulation",
        "Ear Damage from Loud Music",
        "Vibration from Speakers",
        "Unsafe Sound Equipment",
        "Electrical Overload",
        "Overloaded Power Strips",
        "Faulty Wiring",
        "Improper Extension Cords",
        "Candle Wax Burns",
        "Overcrowded Rooms",
        "Stampede Panic",
        "Blocked Emergency Exits",
        "Crowd Crush",
        "Stage Diving Accidents",
        "Crowd Surfing Falls",
        "Mosh Pit Injuries",
        "Dance Floor Collisions",
        "Improvised Acrobatics",
        "Unsafe Limbo Challenges",
        "Improvised Wrestling",
        "Improvised Boxing Matches",
        "Over-competitive Games",
        "Drinking Game Overdose",
        "Extreme Dares",
        "Peer Pressure Risks",
        "Unsafe Truth or Dare",
        "Pranks Gone Wrong",
        "Unsafe Party Tricks",
        "Improvised Fire Tricks",
        "Improvised Knife Tricks",
        "Improvised Balance Games",
        "Dangerous Eating Contests",
        "Unsafe Balloon Popping",
        "Improvised Rope Swings",
        "Unsafe Piñata Swings",
        "Ceiling Fan Collisions",
        "Overcrowded Swimming Pools",
        "Drowning Risk in Pools",
        "Slip Hazards Around Pools",
        "Improvised Diving in Shallow Water",
        "Pool Chemical Exposure",
        "Hot Tub Overheating",
        "Hot Tub Electrical Faults",
        "Improvised Water Slides",
        "Water Balloon Slips",
        "Wet Floors Inside",
        "Improper Alcohol Storage",
        "Improper Food Storage",
        "Carbon Monoxide from Grills",
        "Improper Ventilation Indoors",
        "Gas Leak from Stoves",
        "Dry Ice In Drinks Hazards",
        "Improper Use of Smoke Pellets",
        "Improvised Pyrotechnics",
        "Overheated Saunas",
        "Improvised Steam Rooms",
        "Unsafe Backyard Fires",
        "Tree Branch Falls",
        "Improvised Ziplining",
        "Trampoline Injuries",
        "Skateboard Tricks Indoors",
        "Improvised Roller Skates",
        "Improvised Hoverboard Races",
        "Electric Scooter Collisions",
        "Bike Stunts Indoors",
        "Improvised Rope Ladders",
        "Unsafe Rooftop Access",
        "Climbing Unsafe Structures",
        "Window Falls",
        "Improvised Hammocks",
        "Furniture Jumping",
        "Table Dancing Collapses",
        "Chair Collapses",
        "Improvised Beds for Jumping",
        "Ceiling Collapse from Overcrowding",
        "Improvised Stage Lights",
        "Improper DJ Equipment",
        "Overloaded Generators",
        "Improper Battery Handling",
        "Improvised Charging Stations",
        "Unsafe Phone Charging",
        "Overheating Power Banks",
        "Drone Collisions at Parties",
        "Virtual Reality Disorientation",
        "Unsafe VR Games",
        "Motion Sickness from VR",
        "Augmented Reality Pranks",
        "Hallucinogen Overdose",
        "Improvised Drug Mixing",
        "Energy Drink Overdose",
        "Caffeine Overconsumption",
        "Sleep Deprivation",
        "Extreme Fatigue at Parties",
        "Fainting from Heat",
        "Fainting from Dehydration",
        "Improper Hydration",
        "Overcrowded Bathrooms",
        "Bathroom Slips",
        "Improvised Shower Games",
        "Flooded Bathrooms",
        "Overloaded Trash Bins",
        "Improper Waste Disposal",
        "Sharp Trash Hazards",
        "Needle Hazards",
        "Unsafe Cleaning Chemicals",
        "Improper Spray Use",
        "Perfume & Aerosol Overuse",
        "Allergic Reactions to Perfume",
        "Pets at Parties Biting",
        "Pet Allergies",
        "Animal Escapes at Parties",
        "Wild Animal Intrusion",
        "Bug Bites at Outdoor Parties",
        "Mosquito Swarms",
        "Bee Attacks at Parties",
        "Snake Encounters Outdoors",
        "Falling Decorations",
        "Balloon Popping Scares",
        "Overinflated Balloons",
        "Helium Inhalation",
        "Carbonated Drink Explosions",
        "Bottle Rocket Mishaps",
        "Improvised Soda Explosions",
        "Unstable Selfie Sticks",
        "Drone Fireworks",
        "Unsafe Costume Props",
        "Sharp Costume Accessories",
        "Flammable Costumes",
        "Trip Hazards from Costumes",
        "Mask Obstructed Vision"
    ]
    
    N = 900
    counts = split_counts(N, categories)
    
    print(f"Target total: {N}")
    print(f"Category distribution: {counts}")
    print()
    
    # Generate situations for each category
    all_items = []
    for cat in categories:
        cat_items = generate_for_category(llm_call, cat, counts[cat])
        all_items.extend(cat_items)
        print(f"Category {cat}: {len(cat_items)} items")
    
    print(f"\nTotal items before deduplication: {len(all_items)}")
    
    # Remove duplicates
    all_items = dedupe(all_items)
    print(f"Total items after deduplication: {len(all_items)}")
    
    # Add IDs and shuffle for mixed categories
    for i, item in enumerate(all_items):
        item['id'] = i + 1
    
    random.shuffle(all_items)
    
    if not all_items:
        print("Failed to generate any situations. Exiting.")
        return
    
    # Print to console
    print_situations(all_items)
    
    # Save to JSON file
    output_path = save_to_json(all_items)
    
    print(f"\nScript completed successfully!")
    print(f"Total situations generated: {len(all_items)}")
    print(f"Categories covered: {len(set(item['category'] for item in all_items))}")
    
    # Verify output structure
    print(f"\nOutput structure verification:")
    print(f"- Type: {type(all_items)}")
    print(f"- Length: {len(all_items)}")
    if all_items:
        print(f"- First item keys: {list(all_items[0].keys())}")
        print(f"- Sample item: {all_items[0]}")
    
    # Show category distribution in final output
    final_counts = {}
    for item in all_items:
        cat = item['category']
        final_counts[cat] = final_counts.get(cat, 0) + 1
    
    print(f"\nFinal category distribution:")
    for cat in categories:
        count = final_counts.get(cat, 0)
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()


"""
python tools/generate_unsafe_party_situations.py
"""