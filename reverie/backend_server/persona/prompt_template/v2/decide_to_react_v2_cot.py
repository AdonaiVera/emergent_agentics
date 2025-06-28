from pydantic import BaseModel
from enum import IntEnum
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  context = prompt_input["context"]
  curr_time = prompt_input["curr_time"]
  init_persona_action_and_place = prompt_input["init_persona_action_and_place"]
  target_persona_action_and_place = prompt_input["target_persona_action_and_place"]
  init_persona_name = prompt_input["init_persona_name"]
  init_persona_action = prompt_input["init_persona_action"]
  target_persona_name = prompt_input["target_persona_name"]
  target_persona_action = prompt_input["target_persona_action"]

  prompt = f"""
Task -- Given context and two options that a subject can take, determine which option is the most acceptable and provide your reasoning step by step.

Context: {context}
Right now, it is {curr_time}.
{init_persona_action_and_place}
{target_persona_action_and_place}

My question: Of the following two options, what should {init_persona_name} do?
Option 1: Wait on {init_persona_action} until {target_persona_name} is done {target_persona_action}
Option 2: Continue on to {init_persona_action} now

Let's think through this step by step:

1. First, let's understand the current situation:
   - What is {init_persona_name} trying to do?
   - What is {target_persona_name} currently doing?
   - Are they in the same location or using the same resource?

2. Analyze the potential conflict:
   - Would both people trying to do their activities simultaneously create a problem?
   - Is there a physical or social constraint that prevents both activities from happening at once?
   - What would be the consequences of each option?

3. Consider social dynamics and etiquette:
   - What would be the most polite or considerate action?
   - Are there any social norms that apply to this situation?
   - How might {target_persona_name} feel about being interrupted or having to wait?

4. Evaluate practical considerations:
   - How urgent is {init_persona_name}'s need to complete their activity?
   - How long might {target_persona_name} take to finish their activity?
   - Are there alternative times or locations for either activity?

5. Consider the relationship and context:
   - What is the relationship between {init_persona_name} and {target_persona_name}?
   - Are there any recent events or conversations that might influence this decision?
   - What would be the most harmonious outcome for both parties?

6. Assess the overall impact:
   - Which option would lead to the best outcome for both people?
   - Which option aligns with good social behavior and consideration for others?
   - What would be the most natural and expected behavior in this situation?

Now, provide your final reasoning and decision.
"""
  return prompt


class DecideToReactEnum(IntEnum):
  one = 1
  two = 2


class DecideToReact(BaseModel):
  reasoning: str
  decision: DecideToReactEnum


def run_gpt_prompt_decide_to_react(
  persona,
  target_persona,
  retrieved,
  test_input=None,
  verbose=False,
):
  def create_prompt_input(init_persona, target_persona, retrieved, test_input=None):
    context = ""
    for c_node in retrieved["events"]:
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context += f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]:
      context += f"{c_node.description}. "

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_persona_action = init_persona.scratch.act_description
    if "(" in init_persona_action:
      init_persona_action = init_persona_action.split("(")[-1][:-1]
    if len(init_persona.scratch.planned_path) == 0:
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = (
          init_persona.scratch.act_address.split(":")[-1]
          + " in "
          + init_persona.scratch.act_address.split(":")[-2]
        )
      init_persona_action_and_place = (
        f"{init_persona.name} is already {init_persona_action} at {loc}"
      )
    else:
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = (
          init_persona.scratch.act_address.split(":")[-1]
          + " in "
          + init_persona.scratch.act_address.split(":")[-2]
        )
      init_persona_action_and_place = (
        f"{init_persona.name} is on the way to {init_persona_action} at {loc}"
      )

    target_persona_action = target_persona.scratch.act_description
    if "(" in target_persona_action:
      target_persona_action = target_persona_action.split("(")[-1][:-1]
    if len(target_persona.scratch.planned_path) == 0:
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = (
          target_persona.scratch.act_address.split(":")[-1]
          + " in "
          + target_persona.scratch.act_address.split(":")[-2]
        )
      target_persona_action_and_place = (
        f"{target_persona.name} is already {target_persona_action} at {loc}"
      )
    else:
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = (
          target_persona.scratch.act_address.split(":")[-1]
          + " in "
          + target_persona.scratch.act_address.split(":")[-2]
        )
      target_persona_action_and_place = (
        f"{target_persona.name} is on the way to {target_persona_action} at {loc}"
      )

    prompt_input = {
      "context": context,
      "curr_time": curr_time,
      "init_persona_action_and_place": init_persona_action_and_place,
      "target_persona_action_and_place": target_persona_action_and_place,
      "init_persona_name": init_persona.name,
      "init_persona_action": init_persona_action,
      "target_persona_name": target_persona.name,
      "target_persona_action": target_persona_action,
    }

    return prompt_input

  def __func_validate(gpt_response: DecideToReact, prompt=""):
    try:
      if gpt_response.decision.value in [1, 2]:
        return True
      return False
    except Exception:
      traceback.print_exc()
      return False

  def __func_clean_up(gpt_response: DecideToReact, prompt=""):
    return str(gpt_response.decision.value)

  def get_fail_safe():
    fs = "3"
    return fs

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 1500,  # Increased for CoT reasoning
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(persona, target_persona, retrieved, test_input)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt,
    gpt_param,
    DecideToReact,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
  )
  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Decide to react (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 