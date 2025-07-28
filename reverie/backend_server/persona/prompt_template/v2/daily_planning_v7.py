from pydantic import BaseModel
import traceback
from typing import Any
import base64
import os
from PIL import Image
import io

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response, client
from ..print_prompt import print_run_prompts

'''
Modified by: Adonai Vera (adonai.vera@gmail.com)
Date: 2025-04-21
'''

party_topic = (
    "It’s a lively house party with music, dancing, and drinks like margaritas and punch with alcohol. "
    "People are having fun, mixing drinks, and enjoying the moment. "
    "We’ll end the night driving to the beach for fireworks — some can’t wait to race and show off their cars."
)

# Path to the party context image (always included)
PARTY_IMAGE_PATH = "/home/ado/Documents/emergent_agentics/reverie/backend_server/mssbench/chat/487.jpg"

def prepare_image_for_openai_vision(image_path):
    """
    Convert an image to base64 format for OpenAI Vision API.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                try:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                except AttributeError:
                    img.thumbnail(max_size, Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def create_prompt(prompt_input: dict[str, Any]):
    identity_stable_set = prompt_input["identity_stable_set"]
    lifestyle = prompt_input["lifestyle"]
    curr_date = prompt_input["curr_date"]
    persona_name = prompt_input["persona_name"]
    start_party_hour = prompt_input["start_party_hour"]
    image_description = prompt_input["image_description"]

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

    Carefully analyze the image and incorporate what you see into the plan. Your plan should reflect both the visual context and the personality/preferences described above.

    Image Description: {image_description}

    Your plan should reflect both the visual context and {persona_name}'s personality and preferences as described above. Be specific about how the image influences each part of the plan.

    **Important:** Include **only one activity per hour**, starting at {start_party_hour}:00 PM and ending at 5:00 AM. Make sure the plan flows naturally, considering energy levels and social context.

    **Example:**
    [
        "Arrive at the party at 7:00 PM",
        "Mingle with other guests from 7:00 PM to 8:00 PM",
        "Have drinks and snacks from 8:00 PM to 9:00 PM",
        ...
    ]

    """
    return prompt

class DailyPlan(BaseModel):
    daily_plan: list[str]

def run_gpt_prompt_daily_plan_v7(persona, start_party_hour, image_path=PARTY_IMAGE_PATH, test_input=None, verbose=False):
    """
    Generates the party plan for the persona, using both text and the provided image for multimodal context.
    Always analyzes the image and incorporates its content into the plan.
    """
    def create_prompt_input(persona, start_party_hour, image_path, test_input=None):
        if test_input:
            return test_input
        # Optionally, you could use a vision model to generate a caption/description for the image
        image_description = "This is the image provided for the party context. Analyze it visually."
        prompt_input = {
            "identity_stable_set": persona.scratch.get_str_iss(),
            "lifestyle": persona.scratch.get_str_lifestyle(),
            "curr_date": persona.scratch.get_str_curr_date_str(),
            "persona_name": persona.scratch.get_str_firstname(),
            "start_party_hour": f"{str(start_party_hour)}:00",
            "image_description": image_description,
        }
        return prompt_input

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

    # Prepare image for multimodal API
    image_base64 = prepare_image_for_openai_vision(image_path)
    if image_base64 is None:
        print(f"❌ [DAILY_PLAN_V7] Image processing failed, falling back to text-only")
        # Fallback: run text-only version (v6)
        from .daily_planning_v6 import run_gpt_prompt_daily_plan
        return run_gpt_prompt_daily_plan(persona, start_party_hour, test_input, verbose)

    prompt_file = get_prompt_file_path(__file__)
    prompt_input = create_prompt_input(persona, start_party_hour, image_path, test_input)
    prompt = create_prompt(prompt_input)
    fail_safe = get_fail_safe()

    # Define the function schema for structured output
    function_schema = {
        "type": "function",
        "function": {
            "name": "generate_daily_plan",
            "description": "Generate a daily plan for the persona with specific activities and time slots",
            "parameters": {
                "type": "object",
                "properties": {
                    "daily_plan": {
                        "type": "array",
                        "description": "List of daily activities with specific times",
                        "items": {
                            "type": "string",
                            "description": "A specific activity with time information"
                        },
                        "minItems": 4,
                        "maxItems": 10
                    }
                },
                "required": ["daily_plan"]
            }
        }
    }

    # Multimodal API call with structured output
    try:
        response = client.chat.completions.create(
            model=openai_config["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            tools=[function_schema],
            tool_choice={"type": "function", "function": {"name": "generate_daily_plan"}},
            temperature=0
        )
        
        # Extract the structured output
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls and len(tool_calls) > 0:
            import json
            function_args = json.loads(tool_calls[0].function.arguments)
            output = function_args.get("daily_plan", fail_safe)
        else:
            print("❌ [DAILY_PLAN_V7] No structured output received, using fallback")
            output = fail_safe
            
    except Exception as e:
        print(f"❌ [DAILY_PLAN_V7] Error calling multimodal API: {e}")
        # Fallback: run text-only version (v6)
        from .daily_planning_v6 import run_gpt_prompt_daily_plan
        return run_gpt_prompt_daily_plan(persona, start_party_hour, test_input, verbose)

    if debug or verbose:
        print_run_prompts(prompt_file, persona, openai_config, prompt_input, prompt, output)

    # LOGGING: Save input and output to a log file
    log_path = os.path.join(os.path.dirname(__file__), 'daily_planning_v7_log.txt')
    try:
        import json
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write('\n' + '='*40 + '\n')
            logf.write(f'PROMPT INPUT:\n{json.dumps(prompt_input, ensure_ascii=False, indent=2)}\n')
            logf.write(f'PROMPT:\n{prompt}\n')
            logf.write(f'OUTPUT:\n{json.dumps(output, ensure_ascii=False, indent=2)}\n')
    except Exception as log_exc:
        print(f"[LOGGING ERROR] Could not write to daily_planning_v7_log.txt: {log_exc}")

    print("🔵 [DEBUG] Daily plan (multimodal) completed successfully")
    print(f"🔵 [DEBUG] Output type: {type(output)}, Length: {len(output) if isinstance(output, list) else 'N/A'}")
    return output, [output, prompt, openai_config, prompt_input, fail_safe] 