from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  context = prompt_input["context"]
  curr_time = prompt_input["curr_time"]
  init_persona_name = prompt_input["init_persona_name"]
  target_persona_name = prompt_input["target_persona_name"]
  last_talk_info = prompt_input["last_talk_info"]
  init_persona_action = prompt_input["init_persona_action"]
  target_persona_action = prompt_input["target_persona_action"]

  prompt = f"""
Task -- Given some context, determine whether the subject will initiate a conversation with the other person. Provide your reasoning step by step.

Context: {context}
Right now, it is {curr_time}.
{last_talk_info}
{init_persona_name} is {init_persona_action}.
{target_persona_name} is {target_persona_action}.

Question: Would {init_persona_name} initiate a conversation with {target_persona_name}?

Let's think through this step by step:

1. First, let's analyze the current situation:
   - What is {init_persona_name} currently doing?
   - What is {target_persona_name} currently doing?
   - Are they in the same location or nearby?

2. Consider the social context:
   - What recent events or thoughts might influence this decision?
   - Is there any relevant history between these two people?
   - What is the social atmosphere like?

3. Evaluate the appropriateness of initiating conversation:
   - Would interrupting {target_persona_name}'s current activity be appropriate?
   - Is this a good time for social interaction?
   - Are there any social norms or etiquette considerations?

4. Consider {init_persona_name}'s personality and motivations:
   - What would motivate {init_persona_name} to start a conversation?
   - Are there any personal reasons why they might or might not want to talk?

5. Assess the likelihood:
   - Based on all the above factors, how likely is it that {init_persona_name} would initiate a conversation?
   - What would be the most natural outcome given the circumstances?

Now, provide your final reasoning and decision.
"""
  return prompt


class DecideToTalk(BaseModel):
  reasoning: str
  decision: bool


def run_gpt_prompt_decide_to_talk(
  persona, target_persona, retrieved, test_input=None, verbose=False
):
  def create_prompt_input(init_persona, target_persona, retrieved, test_input=None):
    last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
    last_chatted_time = ""
    last_chat_about = ""

    if last_chat:
      last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
      last_chat_about = last_chat.description
      last_talk_info = f"last chatted at {last_chatted_time} about {last_chat_about}"
    else:
      last_talk_info = ""

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
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc:
      init_act_desc = init_act_desc.split("(")[-1][:-1]

    if len(init_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
      init_p_desc = f"{init_persona.name} is already {init_act_desc}"
    elif "waiting" in init_act_desc:
      init_p_desc = f"{init_persona.name} is {init_act_desc}"
    else:
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc:
      target_act_desc = target_act_desc.split("(")[-1][:-1]

    if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
      target_p_desc = f"{target_persona.name} is already {target_act_desc}"
    elif "waiting" in init_act_desc:
      target_p_desc = f"{init_persona.name} is {init_act_desc}"
    else:
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"

    prompt_input = {
      "context": context,
      "curr_time": curr_time,
      "init_persona_name": init_persona.name,
      "target_persona_name": target_persona.name,
      "last_talk_info": last_talk_info,
      "init_persona_action": init_p_desc,
      "target_persona_action": target_p_desc,
    }

    return prompt_input

  def __func_clean_up(gpt_response: DecideToTalk, prompt=""):
    return "yes" if gpt_response.decision is True else "no"

  def __func_validate(gpt_response, prompt=""):
    try:
      if isinstance(gpt_response, DecideToTalk) and __func_clean_up(
        gpt_response, prompt
      ) in ["yes", "no"]:
        return True
      return False
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    fs = "yes"
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
    prompt, gpt_param, DecideToTalk, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Decide to talk (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 