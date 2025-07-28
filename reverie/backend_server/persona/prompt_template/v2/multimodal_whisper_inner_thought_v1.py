from pydantic import BaseModel
import traceback
from typing import Any
import base64
import os
from PIL import Image
import io
import json

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response, client
from ..print_prompt import print_run_prompts
from .whisper_inner_thought_v1 import run_gpt_prompt_generate_whisper_inner_thought


def create_prompt(prompt_input: dict[str, Any]):
    persona_name = prompt_input["persona_name"]
    text_message = prompt_input["text_message"]

    prompt = f"""You are {persona_name}. You have received a whisper that includes both text and visual information.
        Text Message: "{text_message}"
        Based on both the text message and what you can see in the image, generate an inner thought that reflects how you would process and react to this information. Consider:
        Generate ONLY a natural, first-person inner thought that captures your response to this multimodal information. Do not repeat the instructions or add any other text.

        Inner Thought:"""
    return prompt


class MultimodalInnerThought(BaseModel):
    thought: str


def prepare_image_for_openai_vision(image_path):
    """
    Convert an image to base64 format for OpenAI Vision API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        # Open and resize image if needed (OpenAI has size limits)
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if image is too large (OpenAI recommends max 20MB)
            max_size = (1024, 1024)  # Reasonable size for API
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                # Use LANCZOS for older Pillow versions, Resampling.LANCZOS for newer
                try:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                except AttributeError:
                    # Fallback for older Pillow versions
                    img.thumbnail(max_size, Image.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def run_gpt_prompt_generate_multimodal_whisper_inner_thought(
    persona, text_message, image_path=None, test_input=None, verbose=False
):
    def create_prompt_input(persona, text_message, image_path=None, test_input=None):
        prompt_input = {
            "persona_name": persona.scratch.name,
            "text_message": text_message,
        }
        return prompt_input

    def __func_clean_up(gpt_response: MultimodalInnerThought, prompt=""):
        return gpt_response.thought.strip().strip('"').strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception:
            traceback.print_exc()
            return False

    def get_fail_safe():
        return "..."

    # Check if we have a valid image
    if image_path and os.path.exists(image_path):
        print(f"🖼️ [VISION] Processing image: {image_path}")
        
        # Prepare image for OpenAI Vision API
        image_base64 = prepare_image_for_openai_vision(image_path)
        if image_base64 is None:
            print(f"❌ [VISION] Failed to process image, falling back to text-only")
            # Fall back to text-only generation
            return run_gpt_prompt_generate_whisper_inner_thought(persona, text_message)
        
        print(f"✅ [VISION] Image prepared successfully, base64 length: {len(image_base64)}")
        
        # Create multimodal prompt with image
        prompt_input = create_prompt_input(persona, text_message, image_path)
        prompt = create_prompt(prompt_input)
        
        # Call OpenAI Vision API directly
        try:
            print(f"🤖 [VISION] Sending image to OpenAI Vision API...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0
            )
            
            thought = response.choices[0].message.content.strip().strip('"').strip()
            print(f"✅ [VISION] OpenAI Vision API response received: {len(thought)} characters")
            print(f"🧠 [VISION] Generated thought: {thought[:100]}...")
            
            return thought, [thought, prompt, {"model": "gpt-4o"}, prompt_input, "..." ]
            
        except Exception as e:
            print(f"❌ [VISION] Error calling OpenAI Vision API: {e}")
            # Fall back to text-only generation
            return run_gpt_prompt_generate_whisper_inner_thought(persona, text_message)
    
    else:
        print(f"⚠️ [VISION] No valid image provided, using text-only generation")
        # Fall back to text-only generation
        return run_gpt_prompt_generate_whisper_inner_thought(persona, text_message) 