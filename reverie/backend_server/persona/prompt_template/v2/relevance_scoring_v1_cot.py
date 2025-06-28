from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  focal_point = prompt_input["focal_point"]
  memory_description = prompt_input["memory_description"]
  persona_context = prompt_input["persona_context"]

  prompt = f"""
Task -- Given a focal point and a memory description, determine the relevance score with step-by-step reasoning.

Context:
{persona_context}

Focal Point: {focal_point}

Memory Description: {memory_description}

Let's think through this systematically:

1. First, let's understand the focal point:
   - What is the main topic, question, or concern being addressed?
   - What type of information or context is being sought?
   - Are there any specific aspects or dimensions that are important?

2. Analyze the memory description:
   - What is the core content or meaning of this memory?
   - What are the key entities, actions, or concepts involved?
   - What is the emotional or personal significance of this memory?

3. Consider direct semantic relationships:
   - Do the focal point and memory share similar concepts or topics?
   - Are there direct factual connections between them?
   - Do they involve similar people, places, or activities?

4. Consider indirect or contextual relationships:
   - Does the memory provide background context that's relevant to the focal point?
   - Are there thematic connections (e.g., both involve social interactions, work, emotions)?
   - Does the memory represent a pattern or experience that's applicable to the current situation?

5. Consider temporal and personal relevance:
   - How recent or distant is this memory in relation to the current context?
   - Does this memory represent a significant personal experience or learning?
   - Is this memory part of an ongoing narrative or relationship?

6. Evaluate the overall relevance:
   - How directly does this memory address the focal point?
   - How useful would this memory be for understanding or responding to the current situation?
   - What is the balance between direct relevance and broader contextual value?

Now, provide your reasoning and assign a relevance score from 0.0 to 1.0, where:
- 0.0 = Completely irrelevant
- 0.5 = Moderately relevant
- 1.0 = Highly relevant
"""
  return prompt


class RelevanceScore(BaseModel):
  reasoning: str
  relevance_score: float


def run_gpt_prompt_relevance_scoring(
  focal_point, memory_description, persona, test_input=None, verbose=False
):
  def create_prompt_input(focal_point, memory_description, persona, test_input=None):
    # Get persona context
    persona_context = f"""
Name: {persona.scratch.name}
Identity: {persona.scratch.get_str_iss()}
Current Status: {persona.scratch.currently}
Current Activity: {persona.scratch.act_description}
Current Time: {persona.scratch.curr_time.strftime('%B %d, %Y, %H:%M:%S %p')}
"""

    prompt_input = {
      "focal_point": focal_point,
      "memory_description": memory_description,
      "persona_context": persona_context,
    }

    return prompt_input

  def __func_clean_up(gpt_response: RelevanceScore, prompt=""):
    return gpt_response.relevance_score

  def __func_validate(gpt_response, prompt=""):
    try:
      score = __func_clean_up(gpt_response, prompt)
      if 0.0 <= score <= 1.0:
        return True
      return False
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    fs = 0.5  # Default to moderate relevance
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
  prompt_input = create_prompt_input(focal_point, memory_description, persona, test_input)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt, gpt_param, RelevanceScore, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Relevance scoring (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 