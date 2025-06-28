from utils import debug
from typing import Any

from ..common import ActionLoc, openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  persona_name = prompt_input["persona_name"]
  action_sector = prompt_input["action_sector"]
  accessible_arenas = prompt_input["accessible_arenas"]
  broad_action = prompt_input["broad_action"]
  specific_action = prompt_input["specific_action"]

  prompt = f"""
Task -- choose an appropriate area within a sector for a specific activity with step-by-step reasoning.

Let's think through this systematically:

1. First, let's understand what the person is trying to do:
   - What is the broad action: {broad_action}
   - What is the specific action: {specific_action}
   - What does this activity typically require in terms of space and facilities?

2. Consider the available areas in {action_sector}:
   - Available areas: [{accessible_arenas}]
   - Which of these areas would be most suitable for this specific activity?
   - Are there any areas that are clearly inappropriate for this activity?

3. Consider practical requirements:
   - Does this activity need specific equipment or facilities?
   - Does it require privacy or can it be done in a shared space?
   - Is it a quiet activity or one that might disturb others?

4. Consider social and privacy factors:
   - Should {persona_name} stay in the current area if possible?
   - Are there any areas that belong to other people that should be avoided?
   - What would be the most considerate choice?

5. Make the optimal choice:
   - Stay in the current area if the activity can be done there
   - Never go into other people's rooms unless absolutely necessary
   - Choose the most appropriate and convenient location

Examples:
Jane Anderson is in kitchen in Jane Anderson's house.
Jane Anderson is going to Jane Anderson's house that has the following areas: [kitchen, bedroom, bathroom]
For cooking, Jane Anderson should go to the following area in Jane Anderson's house:
Answer: kitchen
---
Tom Watson is in common room in Tom Watson's apartment.
Tom Watson is going to Hobbs Cafe that has the following areas: [cafe]
For getting coffee, Tom Watson should go to the following area in Hobbs Cafe:
Answer: cafe
---
{persona_name} is going to {action_sector} that has the following areas: [{accessible_arenas}]
{persona_name} is {broad_action}. For {specific_action}, {persona_name} should go to the following area in {action_sector} (MUST pick one of [{accessible_arenas}]):
Answer:
  """
  return prompt


def run_gpt_prompt_action_arena(
  action_description,
  persona,
  act_world,
  act_sector,
  test_input=None,
  verbose=False,
):
  def create_prompt_input(
    action_description, persona, act_world, act_sector, test_input=None
  ):
    world_sector = f"{act_world}:{act_sector}"
    accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(world_sector)
    curr = accessible_arena_str.split(", ")
    fin_accessible_arenas = []

    for i in curr:
      if "'s room" in i:
        if persona.scratch.last_name in i:
          fin_accessible_arenas += [i]
      else:
        fin_accessible_arenas += [i]
    accessible_arena_str = ", ".join(fin_accessible_arenas)

    action_description_1 = action_description
    action_description_2 = action_description
    if "(" in action_description:
      action_description_1 = action_description.split("(")[0].strip()
      action_description_2 = action_description.split("(")[-1][:-1]

    prompt_input = {
      "persona_name": persona.scratch.get_str_name(),
      "action_sector": act_sector,
      "accessible_arenas": accessible_arena_str,
      "broad_action": action_description_1,
      "specific_action": action_description_2,
    }

    return prompt_input

  def __func_clean_up(gpt_response: ActionLoc, prompt=""):
    return gpt_response.area

  def __func_validate(gpt_response, prompt=""):
    arena = __func_clean_up(gpt_response)
    if len(arena.strip()) < 1:
      return False
    if "}" in arena:
      return False
    if "," in arena:
      return False
    return True

  def get_fail_safe():
    fs = "main room"
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
  prompt_input = create_prompt_input(action_description, persona, act_world, act_sector)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt,
    gpt_param,
    ActionLoc,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
    verbose=False,
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Action arena (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 