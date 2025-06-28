from pydantic import BaseModel
import random
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  activity = prompt_input["activity"]
  available_objects = prompt_input["available_objects"]

  prompt = f"""
Task -- choose the most relevant object for a specific activity with step-by-step reasoning.

Let's think through this systematically:

1. First, let's understand what the activity involves:
   - What is the person trying to do: {activity}
   - What are the key requirements for this activity?
   - What type of object would be most essential for this task?

2. Consider the available objects:
   - Available objects: [{available_objects}]
   - Which of these objects would be most directly related to the activity?
   - Are there any objects that are clearly not relevant?

3. Think about functionality:
   - Which object would be the primary tool or resource for this activity?
   - Is there an object that would be the most logical choice?
   - Are there any objects that would be secondary or auxiliary?

4. Consider practical usage:
   - Which object would the person actually interact with most during this activity?
   - Is there an object that represents the main focus of the activity?
   - What would be the most natural object to use for this task?

5. Make the optimal choice:
   - Pick the object that is most central to the activity
   - Choose the object that would be the primary focus
   - Select the most logical and practical choice

Examples:
Current activity: sleep in bed
Objects available: [bed, easel, closet, painting]
Reasoning: Sleeping requires a bed as the primary object for rest.
Pick ONE most relevant object from the objects available: bed
---
Current activity: painting
Objects available: [easel, closet, sink, microwave]
Reasoning: Painting requires an easel as the primary support for the canvas.
Pick ONE most relevant object from the objects available: easel
---
Current activity: cooking
Objects available: [stove, sink, fridge, counter]
Reasoning: Cooking primarily involves using the stove for heating food.
Pick ONE most relevant object from the objects available: stove
---
Current activity: watch TV
Objects available: [couch, TV, remote, coffee table]
Reasoning: Watching TV requires the TV as the primary object for viewing.
Pick ONE most relevant object from the objects available: TV
---
Current activity: study
Objects available: [desk, computer, chair, bookshelf]
Reasoning: Studying typically involves using a desk as the primary workspace.
Pick ONE most relevant object from the objects available: desk
---
Current activity: talk on the phone
Objects available: [phone, charger, bed, nightstand]
Reasoning: Talking on the phone requires the phone as the primary communication device.
Pick ONE most relevant object from the objects available: phone
---
Current activity: {activity}
Objects available: [{available_objects}]
Pick ONE most relevant object from the objects available:
"""
  return prompt


class GameObject(BaseModel):
  object: str


def run_gpt_prompt_action_game_object(
  action_description, persona, temp_address, test_input=None, verbose=False
):
  def create_prompt_input(action_description, persona, temp_address, test_input=None):
    if "(" in action_description:
      action_description = action_description.split("(")[-1][:-1]

    prompt_input = {
      "activity": action_description,
      "available_objects": persona.s_mem.get_str_accessible_arena_game_objects(
        temp_address
      ),
    }

    return prompt_input

  def __func_validate(gpt_response, prompt=""):
    if len(gpt_response.object.strip()) < 1:
      return False
    return True

  def __func_clean_up(gpt_response: GameObject, prompt=""):
    return gpt_response.object

  def get_fail_safe():
    fs = "<random>"
    return fs

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 500,  # Increased for CoT reasoning
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(
    action_description, persona, temp_address, test_input
  )
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt, gpt_param, GameObject, 5, fail_safe, __func_validate, __func_clean_up
  )

  accessible_objects = [
    i.strip()
    for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(
      ","
    )
  ]
  if output not in accessible_objects:
    print("ERROR: Output is not an accessible game object:", output)
    print("Choosing a random accessible object instead.")
    output = random.choice(accessible_objects)
    print("Randomly chosen object:", output)

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Action object (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 