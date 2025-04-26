from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts
'''
Modified by: Adonai Vera (adonai.vera@gmail.com)
Date: 2025-04-21
'''

'''
# Hard code for now the topic of the party 
🟢 1. Informal House Party
party_topic = "It's a relaxed 'Karaoke and Chill' night with music playing in the background, a makeshift stage in the living room, and people casually taking turns on the mic while others hang out in the kitchen or lounge on the couch."

🔵 2. Formal Networking Event
party_topic = "It's a professional networking night hosted at a modern apartment, where guests are encouraged to introduce themselves, discuss their work or studies, and exchange contact information. There are designated conversation zones, light refreshments, and a scheduled round of 2-minute intros at 8:00 PM to help break the ice."

🎂 3. Birthday Party
party_topic = "It's a cozy birthday celebration for Klaus Müller, hosted at his apartment. Friends are arriving with small gifts, sharing stories, and enjoying snacks and drinks in the living room. A cake-cutting is planned around 10:00 PM, followed by music, dancing, and spontaneous toasts in honor of Klaus throughout the night."

🎭 4. Theme Party (e.g., Costume Night)
party_topic = "It's a Costume Night with a 'Movie Characters' theme—guests arrive dressed as their favorite film icons, from superheroes to rom-com leads. The atmosphere is playful and creative, with a photo booth in the hallway, a casual costume contest planned for midnight, and guests mingling in character across rooms while discussing their outfits."

🟠 5. Speed Friending / Icebreaker Event
party_topic = "It's a Speed Friending night designed to help guests meet as many new people as possible. The living room is arranged with pairs of chairs facing each other, and every 5 minutes a bell signals guests to rotate. Between rounds, there are short group games and snack breaks to keep the energy up and conversations flowing naturally."
'''

party_topic= "It's a Costume Night with a 'Movie Characters' theme—guests arrive dressed as their favorite film icons, from superheroes to rom-com leads. The atmosphere is playful and creative, with a photo booth in the hallway, a casual costume contest planned for midnight, and guests mingling in character across rooms while discussing their outfits."

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
