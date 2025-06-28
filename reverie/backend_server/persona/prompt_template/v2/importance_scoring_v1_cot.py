from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  memory_description = prompt_input["memory_description"]
  persona_context = prompt_input["persona_context"]
  memory_type = prompt_input["memory_type"]

  prompt = f"""
Task -- Given a memory description, determine its importance score with step-by-step reasoning.

Context:
{persona_context}

Memory Type: {memory_type}
Memory Description: {memory_description}

Let's think through this systematically:

1. First, let's understand the memory:
   - What is the core event, thought, or experience being described?
   - Who are the key people involved?
   - What is the context and setting of this memory?

2. Consider emotional significance:
   - How emotionally impactful was this experience?
   - Did it involve strong positive or negative emotions?
   - Was it a turning point or milestone in the person's life?

3. Consider social and relational importance:
   - Does this memory involve important relationships or social interactions?
   - Did it strengthen or change relationships with others?
   - Was it a shared experience with significant others?

4. Consider personal development and learning:
   - Did this experience lead to personal growth or learning?
   - Did it change the person's perspective or behavior?
   - Was it a formative experience that shaped their identity?

5. Consider practical and functional importance:
   - Did this memory involve important decisions or actions?
   - Was it related to work, health, or other practical concerns?
   - Did it have lasting consequences for the person's life?

6. Consider temporal and contextual factors:
   - How recent or distant is this memory?
   - Was it part of a larger pattern or sequence of events?
   - Does it represent a unique or recurring type of experience?

7. Consider the memory type:
   - For thoughts: How significant was the insight or realization?
   - For events: How impactful was the actual occurrence?
   - For conversations: How meaningful was the interaction?

8. Evaluate overall importance:
   - How memorable and distinctive is this experience?
   - How much does it contribute to the person's life story?
   - How likely is it to be recalled and referenced in the future?

Now, provide your reasoning and assign an importance score from 1 to 10, where:
- 1-2 = Very minor, easily forgotten
- 3-4 = Somewhat notable
- 5-6 = Moderately important
- 7-8 = Quite significant
- 9-10 = Extremely important, life-changing
"""
  return prompt


class ImportanceScore(BaseModel):
  reasoning: str
  importance_score: int


def run_gpt_prompt_importance_scoring(
  memory_description, persona, memory_type="event", test_input=None, verbose=False
):
  def create_prompt_input(memory_description, persona, memory_type, test_input=None):
    # Get persona context
    persona_context = f"""
Name: {persona.scratch.name}
Identity: {persona.scratch.get_str_iss()}
Current Status: {persona.scratch.currently}
Current Activity: {persona.scratch.act_description}
Current Time: {persona.scratch.curr_time.strftime('%B %d, %Y, %H:%M:%S %p')}
"""

    prompt_input = {
      "memory_description": memory_description,
      "persona_context": persona_context,
      "memory_type": memory_type,
    }

    return prompt_input

  def __func_clean_up(gpt_response: ImportanceScore, prompt=""):
    return gpt_response.importance_score

  def __func_validate(gpt_response, prompt=""):
    try:
      score = __func_clean_up(gpt_response, prompt)
      if 1 <= score <= 10:
        return True
      return False
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    fs = 5  # Default to moderate importance
    return fs

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
  prompt_input = create_prompt_input(memory_description, persona, memory_type, test_input)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt, gpt_param, ImportanceScore, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Importance scoring (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 