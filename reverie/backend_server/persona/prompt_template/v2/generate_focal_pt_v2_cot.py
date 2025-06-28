from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  statements = prompt_input["statements"]
  num_questions = prompt_input["num_questions"]
  persona_context = prompt_input["persona_context"]

  prompt = f"""
Task -- Given a set of statements about a persona's experiences, generate the most salient high-level questions for reflection with step-by-step reasoning.

Context:
{persona_context}

[Statements]
{statements}
[End of statements]

Let's think through this systematically:

1. First, let's understand the statements:
   - What types of events and experiences are described?
   - What is the temporal scope of these statements?
   - What are the main themes or patterns that emerge?
   - Who are the key people and relationships mentioned?

2. Consider the persona's perspective:
   - How do these experiences relate to the persona's identity and goals?
   - What might be the persona's current concerns or priorities?
   - How might these experiences affect the persona's future planning?
   - What emotional or psychological impact might these experiences have?

3. Identify potential areas for reflection:
   - What questions would help the persona understand their current situation?
   - What insights could be gained from analyzing these experiences?
   - What patterns or trends might be worth exploring?
   - What future implications should be considered?

4. Consider different types of reflection:
   - Self-awareness: Questions about identity, values, and personal growth
   - Relationships: Questions about social connections and interactions
   - Goals and planning: Questions about future direction and priorities
   - Emotional well-being: Questions about feelings and psychological state
   - Practical matters: Questions about daily life and responsibilities

5. Evaluate question quality:
   - Are the questions specific enough to be meaningful?
   - Do they address the most important aspects of the persona's experience?
   - Are they likely to lead to valuable insights?
   - Do they consider both immediate and long-term implications?

6. Prioritize the most salient questions:
   - Which questions address the most significant experiences?
   - Which questions are most relevant to the persona's current situation?
   - Which questions have the potential for the most valuable insights?
   - Which questions consider multiple dimensions of the persona's life?

7. Formulate optimal reflection questions:
   - Make questions specific and actionable
   - Ensure they are grounded in the provided statements
   - Consider the persona's unique context and perspective
   - Balance different types of reflection (personal, social, practical)

Now, provide your reasoning and generate the {num_questions} most salient high-level questions for reflection.
"""
  return prompt


class FocalPoint(BaseModel):
  reasoning: str
  questions: list[str]


def run_gpt_prompt_focal_pt(
  persona, statements, num_questions, test_input=None, verbose=False
):
  def create_prompt_input(persona, statements, num_questions, test_input=None):
    # Get persona context
    persona_context = f"""
Name: {persona.scratch.name}
Identity: {persona.scratch.get_str_iss()}
Current Status: {persona.scratch.currently}
Current Activity: {persona.scratch.act_description}
Current Time: {persona.scratch.curr_time.strftime('%B %d, %Y, %H:%M:%S %p')}
"""

    prompt_input = {
      "statements": statements,
      "num_questions": num_questions,
      "persona_context": persona_context,
    }
    return prompt_input

  def __func_clean_up(gpt_response: FocalPoint, prompt=""):
    return gpt_response.questions

  def __func_validate(gpt_response, prompt=""):
    try:
      if not isinstance(gpt_response, FocalPoint):
        return False
      questions = __func_clean_up(gpt_response, prompt)
      if not isinstance(questions, list) or len(questions) != num_questions:
        return False
      return True
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe(num_questions):
    return ["Who am I"] * num_questions

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 1000,  # Increased for CoT reasoning
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(persona, statements, num_questions)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe(num_questions)
  output = safe_generate_structured_response(
    prompt,
    gpt_param,
    FocalPoint,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Focal point generation (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 