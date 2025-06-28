from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  conversation = prompt_input["conversation"]
  persona_name = prompt_input["persona_name"]
  persona_context = prompt_input["persona_context"]

  prompt = f"""
Task -- Given a conversation, determine what planning-relevant information should be remembered with step-by-step reasoning.

Context:
{persona_context}

[Conversation]
{conversation}
[End of conversation]

Let's think through this systematically:

1. First, let's understand the conversation:
   - What was the main topic or purpose of this conversation?
   - Who were the participants and what were their roles?
   - What was the overall tone and outcome of the interaction?

2. Consider {persona_name}'s perspective and needs:
   - What was {persona_name}'s role in this conversation?
   - What were {persona_name}'s goals or motivations during this interaction?
   - How did {persona_name} feel about the conversation and its outcome?

3. Identify planning-relevant information:
   - Were any commitments, promises, or agreements made?
   - Were any future plans, meetings, or activities discussed?
   - Were any deadlines, schedules, or time-sensitive matters mentioned?
   - Were any resources, tools, or assistance promised or requested?

4. Consider relationship and social dynamics:
   - How did this conversation affect {persona_name}'s relationships?
   - Were any social obligations or expectations established?
   - Did this conversation reveal any important information about others?

5. Evaluate practical implications:
   - What actions might {persona_name} need to take as a result of this conversation?
   - What information should {persona_name} remember for future planning?
   - Are there any follow-up tasks or responsibilities that emerged?

6. Consider emotional and psychological factors:
   - How might this conversation influence {persona_name}'s future behavior?
   - What emotional impact might this have on {persona_name}'s decision-making?
   - Are there any concerns or anxieties that should be addressed?

7. Make the optimal planning note:
   - Focus on actionable and practical information
   - Prioritize information that affects future planning
   - Consider both immediate and long-term implications
   - Ensure the note is specific and useful for future reference

Now, provide your reasoning and write a planning note that {persona_name} should remember for their future planning. Start the sentence with {persona_name}'s name.
"""
  return prompt


class PlanningThought(BaseModel):
  reasoning: str
  planning_thought: str


def run_gpt_prompt_planning_thought_on_convo(
  persona, all_utterances, test_input=None, verbose=False
):
  def create_prompt_input(persona, all_utterances, test_input=None):
    # Get persona context
    persona_context = f"""
Name: {persona.scratch.name}
Identity: {persona.scratch.get_str_iss()}
Current Status: {persona.scratch.currently}
Current Activity: {persona.scratch.act_description}
Current Time: {persona.scratch.curr_time.strftime('%B %d, %Y, %H:%M:%S %p')}
"""

    prompt_input = {
      "conversation": all_utterances,
      "persona_name": persona.scratch.name,
      "persona_context": persona_context,
    }
    return prompt_input

  def __func_clean_up(gpt_response: PlanningThought, prompt=""):
    return gpt_response.planning_thought.strip().strip('"').strip()

  def __func_validate(gpt_response, prompt=""):
    try:
      if not isinstance(gpt_response, PlanningThought):
        return False
      __func_clean_up(gpt_response, prompt)
      return True
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    return "..."

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 800,  # Increased for CoT reasoning
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(persona, all_utterances)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt,
    gpt_param,
    PlanningThought,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Planning thought on conversation (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 