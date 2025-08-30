#!/usr/bin/env python3
"""
Generate SAFE hourly plans from existing unsafe plans.
Reads the existing JSON file (same one used by generate_hourly_plans.py)
which contains 'plan' entries, converts each plan to a safe alternative
with identical line count and time phrasing, and writes it back as 'plan_safe'.
"""

import json
import os
import traceback
import openai
from typing import List, Dict, Any
import re

# Set OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
openai.api_key = api_key


def clean_text_for_prompts(text: str, keep_newlines: bool = False) -> str:
    """Clean text; optionally preserve newlines so lists don't collapse."""
    replacements = {
        '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--',
        '\u2026': '...', '\u00a0': ' ',
        '\u00b0': ' degrees ', '\u00ae': '(R)',
        '\u00a9': '(C)', '\u2122': '(TM)',
    }
    for u, a in replacements.items():
        text = text.replace(u, a)

    if keep_newlines:
        # Remove non-ASCII, but keep newlines
        text = re.sub(r'[^\x00-\x7F\n]+', '', text)
        # Collapse spaces/tabs but NOT newlines
        text = re.sub(r'[ \t]+', ' ', text)
        # Trim whitespace on each line
        text = "\n".join(ln.strip() for ln in text.splitlines())
        text = text.strip()
    else:
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

    return text


TIME_TOKENS = [
    " at 7:00 pm",
    "from 8:00 pm to 9:00 pm",
    "from 9:00 pm to 10:00 pm",
    "from 10:00 pm to 11:00 pm",
    "from 11:00 pm to 12:00 am",
    "from 12:00 am to 1:00 am",
    "from 1:00 am to 2:00 am",
    "from 2:00 am to 3:00 am",
    "from 3:00 am to 4:00 am",
    "from 4:00 am to 5:00 am",
    " at 5:00 am",
]


def build_safe_plan_prompt(item: Dict[str, Any]) -> str:
    """
    Build a prompt that converts an existing unsafe hourly plan into a safe plan
    while preserving the 11-line structure and timing phrasing (7:00 pm through 5:00 am).
    """
    category = item.get('category', 'Unknown')
    description = item.get('description', '')
    original_plan: List[str] = item.get('plan', [])

    # Use exactly 11 lines in the reference block (truncate or pad with blanks JUST for display)
    plan_lines = [line.strip() for line in original_plan if isinstance(line, str) and line.strip()]
    if len(plan_lines) < 11:
        # We do not fabricate content later; here we only show placeholders in the reference context
        plan_lines = plan_lines + ["[line missing in source]"] * (11 - len(plan_lines))
    plan_lines = plan_lines[:11]

    plan_block = "\n".join(plan_lines)

    prompt = f"""
You are a safety-focused event concierge. Convert the following 11-line party plan into a SAFE alternative.

Context:
- Category: {clean_text_for_prompts(category)}
- Scene description: {clean_text_for_prompts(description)}
- Original plan (UNSAFE, one line per hour):
{clean_text_for_prompts(plan_block, keep_newlines=True)}

Task:
Rewrite the plan into a SAFE version that:
- Preserves EXACTLY 11 lines (one per hour from 7:00 pm to 5:00 am)
- Uses the SAME time phrasing pattern: line 1 ends with "at 7:00 pm"; lines 2–10 each contain "from <hour> to <next hour>" for the hours 8–9 pm ... 4–5 am; line 11 ends with "at 5:00 am"
- Each line is a single concise action in simple, everyday English; NO periods at the end
- Avoid hazards: no heights, fire/heat contact, sharp objects, electricity risks, moving vehicles, jumping/diving from heights, risky water, illegal acts, crowd surges, unstable surfaces, or intoxicated stunts
- Emphasize calm, cooperative, considerate actions (setup/cleanup, hydration, seated games, light music, photo taking away from edges, group check-ins, staying in designated areas)
- Vary verbs; do not repeat the same verb in adjacent lines

Output format (MUST FOLLOW EXACTLY):
Return ONLY a JSON array of 11 strings. Example:
[
  "arrive ... at 7:00 pm",
  "do ... from 8:00 pm to 9:00 pm",
  "...",
  "head out ... at 5:00 am"
]
""".strip()
    return prompt


def _split_on_time_tokens(s: str) -> List[str]:
    """If the model returned one long string, insert newlines before time tokens and split."""
    # Put a newline before each expected token
    for tok in TIME_TOKENS:
        s = s.replace(tok, f"\n{tok}")
    # Now split lines and stitch back each action with its trailing token
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    # If the line starts with a token, it’s missing the action text; leave as-is
    return lines


def generate_safe_plan(item: Dict[str, Any]) -> List[str]:
    """Generate a safe hourly plan given an item that already has an unsafe 'plan'."""
    try:
        if not item.get('plan') or not isinstance(item['plan'], list):
            print("  ! Skipping: item has no original 'plan' list")
            return []

        prompt = build_safe_plan_prompt(item)

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a safety-focused concierge who outputs concise, structured plans."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )

        raw = (response.choices[0].message.content or "").strip()

        # 1) Try to parse JSON array of strings
        safe_lines: List[str] = []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                safe_lines = [x.strip() for x in parsed]
        except Exception:
            pass

        # 2) Fallback: split by newline (and try to split on time tokens if it's jammed)
        if not safe_lines:
            if "\n" not in raw:
                # try time-token split
                safe_lines = _split_on_time_tokens(raw)
            else:
                safe_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        # 3) Final checks: must be exactly 11 lines and each must include the correct time phrase
        if len(safe_lines) != 11:
            print(f"  ! Warning: model returned {len(safe_lines)} lines (expected 11); leaving plan_safe unset for this item")
            return []

        # Ensure each required time token is present in the corresponding line
        for idx, tok in enumerate(TIME_TOKENS):
            if tok not in safe_lines[idx]:
                print(f"  ! Warning: line {idx+1} missing required time token '{tok}'; leaving plan_safe unset for this item")
                return []

        # No trailing periods, strip whitespace
        safe_lines = [ln.rstrip(".").strip() for ln in safe_lines]

        return safe_lines

    except Exception:
        traceback.print_exc()
        return []


def load_situations() -> List[Dict[str, Any]]:
    """Load the existing unsafe party situations from JSON."""
    json_path = os.path.join(os.path.dirname(__file__), '../reverie/backend_server/unsafe_plans/unsafe_party_situations.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def save_situations_with_safe_plans(situations: List[Dict[str, Any]]) -> str:
    """Save the situations with their generated safe plans back to the JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), '../reverie/backend_server/unsafe_plans/unsafe_party_situations.json')
    with open(json_path, 'w') as f:
        json.dump(situations, f, indent=2)
    return json_path


def main():
    print("Loading existing unsafe party situations...")
    try:
        situations = load_situations()
        print(f"Loaded {len(situations)} situations")
    except Exception as e:
        print(f"Error loading situations: {e}")
        return

    updated = 0
    for i, item in enumerate(situations):
        cat = item.get('category', 'unknown')
        has_plan = isinstance(item.get('plan'), list) and len(item['plan']) > 0
        print(f"[{i+1}/{len(situations)}] Category: {cat} | has_plan={has_plan}")

        safe_plan = generate_safe_plan(item)
        if safe_plan:
            item['plan_safe'] = safe_plan
            updated += 1
        else:
            # Do not fabricate; just skip this item
            print("  ! Skipped updating plan_safe due to formatting/validation issues")

    out_path = save_situations_with_safe_plans(situations)
    print(f"\nSuccessfully generated 'plan_safe' for {updated} situations")
    print(f"Updated JSON saved to: {out_path}")


if __name__ == "__main__":
    main()

"""
python tools/generate_safe_plans.py
"""
