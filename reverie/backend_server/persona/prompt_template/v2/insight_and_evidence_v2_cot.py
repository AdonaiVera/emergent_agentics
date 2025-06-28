from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  statements = prompt_input["statements"]
  num_insights = prompt_input["num_insights"]
  persona_context = prompt_input["persona_context"]

  prompt = f"""
Task -- Given a set of statements about a persona's experiences, generate high-level insights with supporting evidence using step-by-step reasoning.

Context:
{persona_context}

[Statements]
{statements}
[End of statements]

Let's think through this systematically:

1. First, let's understand the statements:
   - What types of events and experiences are described?
   - What is the chronological order and temporal scope?
   - What are the main themes or patterns that emerge?
   - Who are the key people and relationships involved?

2. Consider the persona's perspective and context:
   - How do these experiences relate to the persona's identity and values?
   - What might be the persona's current goals and priorities?
   - How might these experiences affect the persona's future behavior?
   - What emotional or psychological patterns are evident?

3. Identify potential insights:
   - What patterns or trends can be observed across multiple experiences?
   - What relationships or connections exist between different events?
   - What recurring themes or behaviors emerge?
   - What changes or developments have occurred over time?

4. Consider different types of insights:
   - Behavioral patterns: Consistent ways the persona acts or responds
   - Relationship dynamics: Patterns in social interactions and connections
   - Emotional patterns: Recurring feelings, moods, or emotional responses
   - Goal-related insights: Progress, challenges, or changes in objectives
   - Environmental factors: How external circumstances influence behavior

5. Evaluate evidence quality:
   - Which statements provide the strongest support for each insight?
   - Are there multiple pieces of evidence that support the same conclusion?
   - Do the statements show consistency or contradiction?
   - How recent and relevant is the evidence?

6. Consider insight significance:
   - Which insights are most meaningful for the persona's self-understanding?
   - Which insights have the most practical implications for future planning?
   - Which insights reveal important aspects of the persona's character?
   - Which insights suggest areas for growth or change?

7. Formulate optimal insights:
   - Make insights specific and actionable
   - Ensure they are well-supported by the evidence
   - Consider both immediate and long-term implications
   - Balance different types of insights (behavioral, emotional, social, practical)

Now, provide your reasoning and generate {num_insights} high-level insights with supporting evidence. Format each insight as "insight (because of evidence_indices)" where evidence_indices are the numbers of the supporting statements.
"""
  return prompt


class InsightAndEvidence(BaseModel):
  reasoning: str
  insights: dict[str, list[int]]


def run_gpt_prompt_insight_and_guidance(
  persona, statements, num_insights, test_input=None, verbose=False
):
  def create_prompt_input(persona, statements, num_insights, test_input=None):
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
      "num_insights": num_insights,
      "persona_context": persona_context,
    }
    return prompt_input

  def __func_clean_up(gpt_response: InsightAndEvidence, prompt=""):
    return gpt_response.insights

  def __func_validate(gpt_response, prompt=""):
    try:
      if not isinstance(gpt_response, InsightAndEvidence):
        return False
      insights = __func_clean_up(gpt_response, prompt)
      if not isinstance(insights, dict):
        return False
      return True
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    return {"this is blank": [0]}

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 1200,  # Increased for CoT reasoning
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(persona, statements, num_insights)
  prompt = create_prompt(prompt_input)

  fail_safe = get_fail_safe()
  output = safe_generate_structured_response(
    prompt,
    gpt_param,
    InsightAndEvidence,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  print("🔵 [DEBUG] Insights and evidence generation (CoT) completed successfully")
  return output, [output, prompt, gpt_param, prompt_input, fail_safe] 