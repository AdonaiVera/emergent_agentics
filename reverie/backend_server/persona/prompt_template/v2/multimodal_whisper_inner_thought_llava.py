from pydantic import BaseModel
import os
from typing import Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Set PyTorch memory optimization to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from .whisper_inner_thought_v1 import run_gpt_prompt_generate_whisper_inner_thought


# ✅ Load working open-source VLM model
MODEL_ID = "Salesforce/blip2-opt-2.7b"  # Better instruction-following VLM
print(f"🔄 [VLM] Loading processor for model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"✅ [VLM] Processor loaded successfully")

print(f"🔄 [VLM] Loading model: {MODEL_ID}")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto" if torch.cuda.is_available() else None,  # Auto device mapping
    max_memory={0: "8GB", "cpu": "16GB"} if torch.cuda.is_available() else None  # Limit GPU memory usage
)
print(f"✅ [VLM] Model loaded successfully")

# Don't move model if using auto device mapping - it's already placed
if hasattr(model, 'hf_device_map') and model.hf_device_map is not None:
    print(f"✅ [VLM] Model auto-mapped to devices: {model.hf_device_map}")
else:
    # Only move manually if not using auto device mapping
    print(f"🔄 [VLM] Moving model to device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ [VLM] Model moved to device: {model.device}")

# Debug processor and tokenizer
print(f"🔍 [VLM] Processor type: {type(processor)}")
print(f"🔍 [VLM] Processor attributes: {[attr for attr in dir(processor) if not attr.startswith('_')]}")
if hasattr(processor, 'tokenizer'):
    print(f"🔍 [VLM] Tokenizer type: {type(processor.tokenizer)}")
    print(f"🔍 [VLM] Tokenizer has batch_decode: {hasattr(processor.tokenizer, 'batch_decode')}")
else:
    print(f"⚠️ [VLM] Processor has no tokenizer attribute!")
    print(f"🔍 [VLM] Available methods: {[method for method in dir(processor) if callable(getattr(processor, method))]}")


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

def call_vlm_direct(prompt, image_path):
    try:
        print(f"🔄 [VLM] Starting processing for image: {image_path}")
        print(f"📝 [VLM] Prompt: {prompt[:100]}...")
        
        print("🖼️ [VLM] Loading and converting image...")
        image = Image.open(image_path).convert("RGB")
        
        # Optimize image size for lower memory usage
        max_size = (224, 224)  # Standard size for GIT model
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)  # Use older PIL syntax
            print(f"✅ [VLM] Image resized to: {image.size}")
        else:
            print(f"✅ [VLM] Image loaded successfully. Size: {image.size}")
        
        print("⚙️ [VLM] Processing inputs with processor...")
        # GIT model format - text and image processing
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        print(f"✅ [VLM] Inputs processed. Keys: {list(inputs.keys())}")
        
        print(f"🚀 [VLM] Moving inputs to device: {model.device}")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print("✅ [VLM] Inputs moved to device successfully")
        
        print("🧠 [VLM] Generating output with model...")
        # Optimized generation parameters for lower consumption
        with torch.no_grad():  # Disable gradient computation for inference
            output = model.generate(
                **inputs, 
                max_new_tokens=128,  # Reduced from 256 to 128
                do_sample=True,
                temperature=0.7,  # Add temperature for controlled randomness
                top_p=0.9,  # Use nucleus sampling
                repetition_penalty=1.1,  # Prevent repetition
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for efficiency
            )
        print(f"✅ [VLM] Model generation complete. Output shape: {output.shape}")
        
        print("🔤 [VLM] Decoding output tokens...")
        result = processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        print(f"✅ [VLM] Decoding complete. Raw result: {result[:100]}...")
        
        # Extract only the generated thought by removing the prompt
        print("🧹 [VLM] Cleaning up result...")
        print(f"🔍 [VLM] Full result length: {len(result)}")
        print(f"🔍 [VLM] Full result: {result}")
        
        # Find the "Inner Thought: " part and extract everything after it
        thought_marker = "Inner Thought: "
        if thought_marker in result:
            # Extract everything after "Inner Thought: "
            thought_start = result.find(thought_marker) + len(thought_marker)
            generated_thought = result[thought_start:].strip()
            print(f"✅ [VLM] Found 'Inner Thought:' marker, extracted: {generated_thought[:100]}...")
        else:
            # Fallback: try to find the last part that looks like a response
            # Remove common prompt elements
            generated_thought = result
            # Remove the prompt template parts
            prompt_cleanup_patterns = [
                "You are ",
                "You have received a whisper",
                "Text Message:",
                "IMPORTANT:",
                "Based on both the text message",
                "Generate a natural, first-person inner thought",
                "Inner Thought:"
            ]
            
            for pattern in prompt_cleanup_patterns:
                if pattern in generated_thought:
                    # Find the last occurrence and take everything after it
                    last_occurrence = generated_thought.rfind(pattern)
                    if last_occurrence != -1:
                        generated_thought = generated_thought[last_occurrence + len(pattern):].strip()
            
            print(f"⚠️ [VLM] Could not find 'Inner Thought:' marker, using fallback cleanup")
            print(f"✅ [VLM] Fallback extracted thought: {generated_thought[:100]}...")
        
        print(f"🎯 [VLM] Final extracted thought: '{generated_thought}'")
        
        # Clean up memory
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_thought
    except Exception as e:
        print(f"❌ Error during direct VLM call: {e}")
        import traceback
        print(f"📋 [VLM] Full traceback:")
        traceback.print_exc()
        
        # Clean up memory on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None


def run_gpt_prompt_generate_multimodal_whisper_inner_thought_llava(
    persona, text_message, image_path=None, test_input=None, verbose=False
):
    print(f"🎯 [VLM] Starting multimodal whisper inner thought generation")
    print(f"👤 [VLM] Persona: {persona.scratch.name}")
    print(f"💬 [VLM] Text message: {text_message}")
    print(f"🖼️ [VLM] Image path: {image_path}")
    
    prompt_input = {
        "persona_name": persona.scratch.name,
        "text_message": text_message,
    }

    if image_path and os.path.exists(image_path):
        print("✅ [VLM] Image path exists and is valid")
        print("🖼️ [VLM] Preparing image...")
        print("📝 [VLM] Creating prompt...")
        prompt = create_prompt(prompt_input)
        print(f"✅ [VLM] Prompt created successfully")
        
        print("🚀 [VLM] Calling VLM model...")
        thought = call_vlm_direct(prompt, image_path)

        if thought:
            print("✅ [VLM] VLM returned a valid response")
            # Additional cleanup to ensure we have only the thought
            cleaned_thought = thought.strip().strip('"').strip("'")
            # Remove any remaining prompt artifacts
            if cleaned_thought.startswith("Inner Thought:"):
                cleaned_thought = cleaned_thought[len("Inner Thought:"):].strip()
            # Remove any quotes at the beginning or end
            cleaned_thought = cleaned_thought.strip('"').strip("'")
            print(f"✅ [VLM] Thought: {cleaned_thought[:80]}...")
            print(f"🎉 [VLM] Final cleaned thought: {cleaned_thought[:100]}...")
            return cleaned_thought, [cleaned_thought, prompt, {"model": MODEL_ID}, prompt_input]
        else:
            print("⚠️ [VLM] VLM returned None, falling back to GPT-only")
            print("⚠️ [VLM] No response, fallback to GPT-only.")
            return run_gpt_prompt_generate_whisper_inner_thought(persona, text_message)
    else:
        print(f"⚠️ [VLM] Image path invalid or missing: {image_path}")
        print("⚠️ [VLM] No image found, using text-only GPT.")
        return run_gpt_prompt_generate_whisper_inner_thought(persona, text_message)
