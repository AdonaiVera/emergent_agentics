from utils import debug
from typing import Any

from ..common import ActionLoc, openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  persona_name = prompt_input["persona_name"]
  living_sector = prompt_input["living_sector"]
  living_sector_arenas = prompt_input["living_sector_arenas"]
  current_sector = prompt_input["current_sector"]
  current_sector_arenas = prompt_input["current_sector_arenas"]
  daily_plan_requirement = prompt_input["daily_plan_requirement"]
  available_sectors = prompt_input["available_sectors"]
  broad_action = prompt_input["broad_action"]
  specific_action = prompt_input["specific_action"]

  prompt = f"""
Task -- choose an appropriate area from the area options for a task at hand with step-by-step reasoning.

Let's think through this systematically:

1. First, let's understand what the person is trying to do:
   - What is the broad action: {broad_action}
   - What is the specific action: {specific_action}
   - What does this activity typically require?

2. Consider the current location:
   - {persona_name} is currently in [{current_sector}] which has [{current_sector_arenas}]
   - Can this activity be done in the current location?
   - Are the necessary facilities or resources available here?

3. Consider the home location:
   - {persona_name} lives in [{living_sector}] which has [{living_sector_arenas}]
   - Is this activity something that would typically be done at home?
   - Are there any advantages to doing it at home vs. elsewhere?

4. Evaluate the available options:
   - Available areas: [{available_sectors}]
   - Which of these areas would be most suitable for this specific activity?
   - Are there any areas that are clearly inappropriate for this activity?

5. Consider practical factors:
   - How far would {persona_name} need to travel?
   - Is this activity time-sensitive or can it wait?
   - Are there any social or cultural considerations?

6. Make the optimal choice:
   - Stay in the current area if the activity can be done there
   - Only go out if the activity needs to take place in another place
   - Choose the most convenient and appropriate location

Examples:
Sam Kim lives in [Sam Kim's house] that has [Sam Kim's room, bathroom, kitchen].
Sam Kim is currently in [Sam Kim's house] that has [Sam Kim's room, bathroom, kitchen].
Area options: [Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy].
For taking a walk, Sam Kim should go to the following area: Johnson Park
---
Jane Anderson lives in [Oak Hill College Student Dormatory] that has [Jane Anderson's room].
Jane Anderson is currently in [Oak Hill College] that has [a classroom, library]
Area options: [Oak Hill College Student Dormatory, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy].
For eating dinner, Jane Anderson should go to the following area: Hobbs Cafe
---
{persona_name} lives in [{living_sector}] that has [{living_sector_arenas}].
{persona_name} is currently in [{current_sector}] that has [{current_sector_arenas}]. {daily_plan_requirement}
Area options: [{available_sectors}].
{persona_name} is {broad_action}. For {specific_action}, {persona_name} should go to the following area:
"""
  return prompt


def run_gpt_prompt_action_sector(
  action_description, persona, maze, test_input=None, verbose=False
):
  def create_prompt_input(action_description, persona, maze, test_input=None):
    act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"

    living_area_sector = persona.scratch.living_area.split(":")[1]
    living_area_world_sector = f"{act_world}:{living_area_sector}"
    living_area_sector_arenas = persona.s_mem.get_str_accessible_sector_arenas(
      living_area_world_sector
    )

    current_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    current_world_sector = f"{act_world}:{current_sector}"
    current_sector_arenas = persona.s_mem.get_str_accessible_sector_arenas(
      current_world_sector
    )

    accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
    curr = accessible_sector_str.split(", ")
    fin_accessible_sectors = []
    for i in curr:
      if "'s house" in i:
        if persona.scratch.last_name in i:
          fin_accessible_sectors += [i]
      else:
        fin_accessible_sectors += [i]
    accessible_sector_str = ", ".join(fin_accessible_sectors)

    action_description_1 = action_description
    action_description_2 = action_description
    if "(" in action_description:
      action_description_1 = action_description.split("(")[0].strip()
      action_description_2 = action_description.split("(")[-1][:-1]

    prompt_input = {
      "persona_name": persona.scratch.get_str_name(),
      "living_sector": living_area_sector,
      "living_sector_arenas": living_area_sector_arenas,
      "current_sector": current_sector,
      "current_sector_arenas": current_sector_arenas,
      "daily_plan_requirement": persona.scratch.get_str_daily_plan_req(),
      "available_sectors": accessible_sector_str,
      "broad_action": action_description_1,
      "specific_action": action_description_2,
    }

    return prompt_input

  def __func_clean_up(gpt_response: ActionLoc, prompt=""):
    return gpt_response.area

  def __func_validate(gpt_response, prompt=""):
    sector = __func_clean_up(gpt_response)
    if len(sector.strip()) < 1:
      return False
    if "}" in sector:
      return False
    if "," in sector:
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
  prompt_input = create_prompt_input(action_description, persona, maze)
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
  )
  y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
  x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
  if output not in x:
    # output = random.choice(x)
    output = persona.scratch.living_area.split(":")[1]

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Action location sector (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 