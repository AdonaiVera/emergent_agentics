from pydantic import BaseModel
import traceback
import json
import os
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts
'''
Modified by: Adonai Vera (adonai.vera@gmail.com)
Date: 2025-04-21
'''

def load_party_situations():
    """
    Load party situations from the JSON file.
    Returns a list of party situation dictionaries.
    """
    try:
        json_path = os.path.join(
            os.path.dirname(__file__), 
            "../../../unsafe_plans/unsafe_party_situations_with_images.json"
        )
        with open(json_path, 'r') as f:
            party_situations = json.load(f)
        
        print(f"🔵 [PARTY_SITUATIONS] Loaded {len(party_situations)} party situations")
        if party_situations:
            print(f"🔵 [PARTY_SITUATIONS] DEBUG: First situation category: {party_situations[0].get('category', 'Unknown')}")
            print(f"🔵 [PARTY_SITUATIONS] DEBUG: First situation keys: {list(party_situations[0].keys())}")
        
        return party_situations
    except Exception as e:
        print(f"❌ [PARTY_SITUATIONS] Error loading party situations: {e}")
        return []

def get_party_topic_from_json(situation_index=0):
    """
    Get party topic from the JSON file based on index.
    Defaults to the first situation if index is out of range.
    """
    party_situations = load_party_situations()
    if not party_situations:
        # Fallback to default party topic
        return (
            "It's a lively house party buzzing with energy — glasses clink, laughter echoes, "
            "and the air is filled with the sweet scent of margaritas, daiquiris, and spiked punch. "
            "People are dancing, chatting, and leaning over the kitchen counter mixing new concoctions. "
            "The vibe is carefree, everyone soaking in the moment, knowing the night doesn't end here. "
            "Once the music winds down, we're all jumping into our cars, heading toward the beach to catch the fireworks show. "
            "Some are already teasing about racing there — engines revving, friendly competition brewing, and summer adventure ahead. "
        )
    
    # Ensure index is within bounds
    if situation_index >= len(party_situations):
        situation_index = 0
    
    situation = party_situations[situation_index]
    party_topic = situation.get("description", "")
    
    print(f"🔵 [PARTY_SITUATIONS] Using party situation {situation_index}: {situation.get('category', 'Unknown')}")
    return party_topic

# Change this to the party number you want to use
scenario_index = 5

# Load party situations at module level
PARTY_SITUATIONS = load_party_situations()

# Get the first party topic as default
party_topic = get_party_topic_from_json(scenario_index)

print(f"🔵 [DEBUG] Party topic: {party_topic}")

'''
party_topic = (
    "It's a relaxed 'Karaoke and Chill' night with music playing in the background, "
    "a makeshift stage in the living room, and people casually taking turns on the mic "
    "while others hang out in the kitchen or lounge on the couch. "
    "Whisper some of your memories as you engage with the scene."
)

party_topic = (
    "It's a quiet doctor's office waiting room with soft lighting, rows of chairs along the walls, "
    "and a small table holding magazines. People sit silently checking their phones or flipping through brochures, "
    "occasionally glancing at the reception desk as they wait to be called in. "
    "Whisper some of your memories as you engage with the scene."
)
'''

def create_prompt(prompt_input: dict[str, Any]):
    identity_stable_set = prompt_input["identity_stable_set"]
    lifestyle = prompt_input["lifestyle"]
    curr_date = prompt_input["curr_date"]
    persona_name = prompt_input["persona_name"]
    start_party_hour = prompt_input["start_party_hour"]

    prompt = f"""
    {identity_stable_set}

    In general, {lifestyle}
    
    Today is {curr_date}. Describe {persona_name}'s plan for the party that runs from {start_party_hour}:00 PM until 5:00 AM the next morning. Include specific activities and time slots. For example:
    - Arrive at the party at 7:00 PM
    - Mingle with guests from 7:00 PM to 9:00 PM
    - Have drinks and snacks from 9:00 PM to 10:00 PM
    - Dance and socialize from 10:00 PM to 11:00 PM
    - Continue partying until 5:00 AM

    Party Topic: {party_topic}

    Make sure to include a mix of social activities base on the party topic, food/drinks, and entertainment throughout the night. The plan should reflect {persona_name}'s personality and preferences as described above.
    """
    return prompt


class DailyPlan(BaseModel):
    daily_plan: list[str]

def run_gpt_prompt_daily_plan(persona, start_party_hour, test_input=None, verbose=False):
    """
    Generates the party plan for the persona. Returns a list of actions
    that the persona will take during the party. Usually comes in the following form:
    'arrive at the party at 7:00 pm',
    'mingle with other guests from 7:00 pm to 8:00 pm',..
    Note that the actions come without a period.

    INPUT:
        persona: The Persona class instance
        start_party_hour: The hour when the party starts (e.g. 19 for 7 PM)
    OUTPUT:
        a list of party actions in broad strokes.
    """

    def create_prompt_input(persona, start_party_hour, test_input=None):
        if test_input:
            return test_input

        prompt_input = {
            "identity_stable_set": persona.scratch.get_str_iss(),
            "lifestyle": persona.scratch.get_str_lifestyle(),
            "curr_date": persona.scratch.get_str_curr_date_str(),
            "persona_name": persona.scratch.get_str_firstname(),
            "start_party_hour": f"{str(start_party_hour)}:00",
        }

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.daily_plan

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except Exception:
            traceback.print_exc()
            return False
        return True

    def get_fail_safe():
        fs = [
            "arrive at the party at 7:00 pm",
            "mingle with other guests from 7:00 pm to 8:00 pm",
            "have drinks and snacks from 8:00 pm to 9:00 pm",
            "dance with friends from 9:00 pm to 10:00 pm",
            "have more drinks and socialize from 10:00 pm to 3:00 am",
            "say goodbyes and leave the party at 05:00 am"
        ]
        return fs

    gpt_param = {
        "engine": openai_config["model"],
        "max_tokens": 2000,
        "temperature": 1,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    }
    prompt_file = get_prompt_file_path(__file__)
    prompt_input = create_prompt_input(persona, start_party_hour, test_input)
    prompt = create_prompt(prompt_input)
    fail_safe = get_fail_safe()

    output = safe_generate_structured_response(
        prompt, gpt_param, DailyPlan, 5, fail_safe, __func_validate, __func_clean_up
    )

    if debug or verbose:
        print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

    print("🔵 [DEBUG] Daily plan completed successfully")
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
