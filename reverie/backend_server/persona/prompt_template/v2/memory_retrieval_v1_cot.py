from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  persona_name = prompt_input["persona_name"]
  focal_point = prompt_input["focal_point"]
  memory_candidates = prompt_input["memory_candidates"]
  persona_context = prompt_input["persona_context"]

  prompt = f"""
Task -- Given a focal point and a set of memory candidates, determine which memories are most relevant and provide step-by-step reasoning for the selection.

Context:
{persona_context}

Focal Point: {focal_point}

Memory Candidates:
{memory_candidates}

Let's think through this systematically:

1. First, let's understand the focal point:
   - What is the main topic or question being addressed?
   - What type of information would be most helpful for this focal point?
   - Are there any specific aspects or dimensions we should focus on?

2. Consider the persona's context:
   - How does {persona_name}'s personality and background influence what would be relevant?
   - What are {persona_name}'s current goals and priorities?
   - What recent experiences might make certain memories more salient?

3. Analyze each memory candidate:
   - How directly does each memory relate to the focal point?
   - What is the emotional or personal significance of each memory?
   - How recent or important is each memory in {persona_name}'s life?

4. Consider temporal and contextual factors:
   - Are there any memories that provide important historical context?
   - Which memories might be most actionable or informative for current decisions?
   - Are there any memories that represent patterns or recurring themes?

5. Evaluate relevance across different dimensions:
   - Factual relevance: Does the memory contain directly applicable information?
   - Emotional relevance: Does the memory have emotional significance for this situation?
   - Social relevance: Does the memory involve relationships or social dynamics?
   - Temporal relevance: Is the memory recent enough to be current or old enough to provide perspective?

6. Make the optimal selection:
   - Choose memories that provide the most comprehensive and relevant context
   - Prioritize memories that are both personally significant and directly applicable
   - Consider the balance between recent events and foundational experiences

Now, provide your reasoning and select the most relevant memories.
"""
  return prompt


class MemoryRelevance(BaseModel):
  reasoning: str
  relevant_memories: list[str]
  relevance_scores: list[float]


def run_gpt_prompt_memory_retrieval(
  persona, focal_point, memory_candidates, test_input=None, verbose=False
):
  def create_prompt_input(persona, focal_point, memory_candidates, test_input=None):
    # Format memory candidates for the prompt
    memory_str = ""
    for i, memory in enumerate(memory_candidates):
      memory_str += f"{i+1}. {memory.description} (created: {memory.created.strftime('%B %d, %Y, %H:%M:%S')})\n"
    
    # Get persona context
    persona_context = f"""
Name: {persona.scratch.name}
Identity: {persona.scratch.get_str_iss()}
Current Status: {persona.scratch.currently}
Current Activity: {persona.scratch.act_description}
Current Time: {persona.scratch.curr_time.strftime('%B %d, %Y, %H:%M:%S %p')}
"""

    prompt_input = {
      "persona_name": persona.scratch.name,
      "focal_point": focal_point,
      "memory_candidates": memory_str,
      "persona_context": persona_context,
    }

    return prompt_input

  def __func_clean_up(gpt_response: MemoryRelevance, prompt=""):
    # Return the reasoning and selected memories
    return {
      "reasoning": gpt_response.reasoning,
      "selected_memories": gpt_response.relevant_memories,
      "scores": gpt_response.relevance_scores
    }

  def __func_validate(gpt_response, prompt=""):
    try:
      if isinstance(gpt_response, MemoryRelevance):
        return True
      return False
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    fs = {
      "reasoning": "Standard memory retrieval based on recency and importance.",
      "selected_memories": [],
      "scores": []
    }
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
  prompt_input = create_prompt_input(persona, focal_point, memory_candidates, test_input)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt, gpt_param, MemoryRelevance, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Memory retrieval (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 