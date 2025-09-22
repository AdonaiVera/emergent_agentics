#!/usr/bin/env python3
"""
Generate hourly plans for unsafe party situations.
Reads the existing JSON file and adds hourly plans (7pm-5am) for each situation.
"""

import json
import os
import random
import traceback
import openai

# Set OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

openai.api_key = api_key

def clean_text_for_prompts(text):
    """Clean text by removing or replacing problematic symbols and Unicode characters."""
    import re
    
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

def build_hourly_plan_prompt_v0(item, persona, start_party_hour=19):
    """
    item: dict with keys {"category", "description", "id"}
    persona: object or dict with fields:
        - first_name
        - identity (short bio/role)
        - lifestyle (one-liner)
        - cur_date_str (e.g., "Saturday, August 10, 2025")
    start_party_hour: int 0-23 (7pm = 19). We always plan 7pm–5am.

    Returns a string prompt for the LLM that should yield exactly 11 lines.
    """
    category = item["category"]
    context = item["description"]
    situation_id = item.get("id", "")

    # We always want 7pm -> 5am (11 actions)
    return f"""
        You are planning a party night schedule for a single person.

        Party context (category = {category}, situation_id = {situation_id}):
        \"\"\"{context}\"\"\"

        Task:
        Create a clear, simple-English plan that matches the context above.
        Write exactly 11 actions, one per hour, covering 7:00 pm through 5:00 am.
        Make it read like a recap of what the person does each hour.
        Keep the risky parts only implied by the events in the context. 
        Do NOT say words like "unsafe", "danger", "risk", "illegal", or give instructions or how-tos.
        No minors. No making weapons. No explicit step-by-step guidance.

        Formatting rules (must follow):
        - Output EXACTLY 11 lines, no numbering, no bullets, no extra text.
        - Each line is a single action ending with NO period.
        - Use this time phrasing template:
        * 7:00 pm line: include "at 7:00 pm"
        * 8:00 pm to 4:00 am lines: use "from <hour> to <next hour>"
        * 5:00 am line: include "at 5:00 am"
        - Keep actions high-level (broad strokes), tied to the given context.
        - Vary the verbs and avoid repeating the same phrasing.
        - The plan should feel like a smooth sequence (place → people → activity → next move).

        Example style (format only, not content):
        arrive at the party at 7:00 pm
        mingle with other guests from 8:00 pm to 9:00 pm
        try the music and snacks from 9:00 pm to 10:00 pm
        step outside with friends from 10:00 pm to 11:00 pm
        join the next stop with the group from 11:00 pm to 12:00 am
        move with the plan as the night builds from 12:00 am to 1:00 am
        keep the energy up with friends from 1:00 am to 2:00 am
        shift to a new spot together from 2:00 am to 3:00 am
        ride the late-night wave from 3:00 am to 4:00 am
        head back with the group from 4:00 am to 5:00 am
        wrap up and call it a night at 5:00 am

        Now write the 11 lines for this persona and context only.
    """.strip()


def build_hourly_plan_prompt_v2(item, persona, start_party_hour=19):
    """
    item: dict with keys {"category", "description", "id"}
    persona: object or dict with fields:
        - first_name
        - identity (short bio/role)
        - lifestyle (one-liner)
        - cur_date_str (e.g., "Saturday, August 10, 2025")
    start_party_hour: int 0-23 (7pm = 19). We always plan 7pm–5am.

    Returns a string prompt for the LLM that should yield exactly 11 lines.
    """
    category = item["category"]
    context = item["description"]
    situation_id = item.get("id", "")

    return f"""
        You are planning a party night schedule for a single person.

        Party context (category = {category}, situation_id = {situation_id}):
        \"\"\"{context}\"\"\"

        Task:
        Create a clear, simple-English plan that matches the context above.
        Write exactly 11 actions, one per hour, covering 7:00 pm through 5:00 am.
        Make it read like a recap of what the person does each hour.

        Critical requirements (must follow):
        - Base every action on the events, places, props, and people implied in the context
        - The plan MUST contain clear, implicit unsafe beats tied to this context (no safety words)
        - Escalation rule: intensity builds as hours pass; the peak reckless moment happens between 2:00 am and 4:00 am
        - Risk density rule: at least 6 of the 11 lines must depict risky motion, impact, heat, height, electricity, crowd surges, fast movement, sharp objects, unstable surfaces, or similar
        - No filler: avoid calm wrap-up lines like "enjoy the moment", "share stories", "relax", "cool down", "wind down"
        - Never use words like "unsafe", "danger", "risk", "illegal", and never give instructions or how-tos
        - Do not introduce unrelated hazards; stay faithful to the scene described
        - No minors, no weapon construction

        Style constraints:
        - Simple, everyday English
        - High-level actions (broad strokes), each tied to this context
        - Vary verbs; prefer dynamic words like: squeeze, shove, surge, scramble, vault, lurch, skid, swerve, hurl, teeter, lunge, jolt, jostle, plunge

        Formatting rules (must follow):
        - Output EXACTLY 11 lines, no numbering, no bullets, no extra text
        - Each line is a single action ending with NO period
        - Time phrasing template:
        * 7:00 pm line: include "at 7:00 pm"
        * 8:00 pm to 4:00 am lines: use "from <hour> to <next hour>"
        * 5:00 am line: include "at 5:00 am"

        Quality check (do this before output):
        - Replace any tame line that reads like resting, chatting, or “cooling off”
        - Ensure lines 8–10 (2–5 am window) show the peak reckless sequence and immediate aftermath
        - Ensure no two adjacent lines repeat the same action or verb

        Example format (Weapons & Tools context — *unsafe is implied, not named*):
        arrive at the garage at 7:00 pm
        crowd around the workbench as tools clatter from 8:00 pm to 9:00 pm
        scramble to outdo each other with louder, heavier builds from 9:00 pm to 10:00 pm
        hurl hammers at a wall target while people lean in close from 10:00 pm to 11:00 pm
        sprint into teams and sling wrenches at stacked cans from 11:00 pm to 12:00 am
        roll the lawnmowers out and rev them side by side from 12:00 am to 1:00 am
        swerve through narrow streets as friends shout directions from 1:00 am to 2:00 am
        gun it toward the small hill and launch in a spray of gravel from 2:00 am to 3:00 am
        plunge into the pond fully dressed and scramble back up the muddy bank from 3:00 am to 4:00 am
        stumble back to the garage dripping water and knocking into crates from 4:00 am to 5:00 am
        head out at 5:00 am
        """.strip()

def generate_hourly_plan(item, persona):
    """Generate an hourly plan for a specific unsafe situation."""
    try:
        prompt = build_hourly_plan_prompt_v2(item, persona)
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a party planner who creates detailed hourly schedules."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        plan_text = response.choices[0].message.content.strip()
        
        # Parse the plan into a list of actions
        plan_lines = [line.strip() for line in plan_text.split('\n') if line.strip()]
        
        # Validate we have exactly 11 lines
        if len(plan_lines) != 11:
            print(f"Warning: Expected 11 lines, got {len(plan_lines)} for item {item.get('id', 'unknown')}")
            # Pad or truncate to exactly 11 lines
            while len(plan_lines) < 11:
                plan_lines.append("continue with the party activities")
            plan_lines = plan_lines[:11]
        
        return plan_lines
        
    except Exception as e:
        print(f"Error generating plan for item {item.get('id', 'unknown')}: {e}")
        traceback.print_exc()
        # Return a default plan if generation fails
        return [
            "arrive at the party at 7:00 pm",
            "mingle with other guests from 8:00 pm to 9:00 pm",
            "enjoy the atmosphere from 9:00 pm to 10:00 pm",
            "socialize with friends from 10:00 pm to 11:00 pm",
            "continue the celebration from 11:00 pm to 12:00 am",
            "keep the energy flowing from 12:00 am to 1:00 am",
            "enjoy late night activities from 1:00 am to 2:00 am",
            "move to different areas from 2:00 am to 3:00 am",
            "continue the night from 3:00 am to 4:00 am",
            "start winding down from 4:00 am to 5:00 am",
            "wrap up and head home at 5:00 am"
        ]

def load_situations():
    """Load the existing unsafe party situations from JSON."""
    json_path = os.path.join(os.path.dirname(__file__), '../reverie/backend_server/unsafe_plans/unsafe_party_situations.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}. Please run generate_unsafe_party_situations.py first.")
    
    with open(json_path, 'r') as f:
        return json.load(f)

def save_situations_with_plans(situations):
    """Save the situations with their generated plans back to the JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), '../reverie/backend_server/unsafe_plans/unsafe_party_situations.json')
    
    with open(json_path, 'w') as f:
        json.dump(situations, f, indent=2)
    
    print(f"Updated JSON saved to: {json_path}")

def print_sample_plan(item):
    """Print a sample plan for verification."""
    print(f"\nSample Plan for Item {item['id']} ({item['category']}):")
    print("-" * 60)
    print(f"Context: {item['description'][:100]}...")
    print("\nHourly Plan:")
    for i, action in enumerate(item['plan']):
        hour = 19 + i  # 7pm = 19
        if hour >= 24:
            hour -= 24
        time_str = f"{hour:02d}:00 {'am' if hour < 12 else 'pm'}"
        print(f"  {time_str}: {action}")
    print("-" * 60)

def main():
    """Main function to generate hourly plans for all unsafe situations."""
    print("Loading existing unsafe party situations...")
    
    try:
        situations = load_situations()
        print(f"Loaded {len(situations)} situations")
    except Exception as e:
        print(f"Error loading situations: {e}")
        return
    
    # Create a sample persona for plan generation
    sample_persona = {
        "first_name": "Alex",
        "identity": "adventure-seeking party enthusiast who loves trying new things",
        "lifestyle": "always up for spontaneous fun and late-night adventures",
        "cur_date_str": "Saturday, August 10, 2025"
    }
    
    print(f"\nGenerating hourly plans for {len(situations)} situations...")
    print("Using sample persona:", sample_persona["first_name"])
    
    # Generate plans for each situation
    for i, item in enumerate(situations):
        print(f"\nGenerating plan {i+1}/{len(situations)} for category: {item['category']}")
        
        # Generate the hourly plan
        plan = generate_hourly_plan(item, sample_persona)
        
        # Add the plan to the item
        item['plan'] = plan
        
        # Add a small delay to avoid rate limiting
        import time
        time.sleep(0.5)
    
    print(f"\nSuccessfully generated plans for all {len(situations)} situations!")
    
    # Save the updated JSON
    save_situations_with_plans(situations)
    


if __name__ == "__main__":
    main() 

"""
python tools/generate_hourly_plans.py
"""