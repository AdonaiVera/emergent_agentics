"""
Model Configuration for Emergent Agentics

This file provides easy configuration for switching between different models
for multimodal planning tasks.
"""

# Model selection for multimodal planning
# Options: "gpt4o", "deepseek", "qwen", "claude"
MULTIMODAL_MODEL = "gpt4o"  # Default to Qwen

# Flag to disable images in multimodal requests (for text-only experiments)
# Set to True to run experiments without sending images to the model
DISABLE_IMAGES = False

# DeepSeek specific configuration
DEEPSEEK_CONFIG = {
    "model_name": "deepseek-ai/deepseek-vl-7b-chat",
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "device": "auto"  # "auto", "cuda", "cpu"
}

# Qwen specific configuration
QWEN_CONFIG = {
    "model_name": "Qwen/Qwen2-VL-7B-Instruct",
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "device": "auto"  # "auto", "cuda", "cpu"
}

# GPT-4o specific configuration (uses existing OpenAI config)
GPT4O_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0
}

# Claude specific configuration
CLAUDE_CONFIG = {
    "model": "claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet with vision
    "max_tokens": 1024,
    "temperature": 0.1
}

def get_multimodal_model():
    """
    Get the current multimodal model setting.
    
    Returns:
        str: The current model setting ("gpt4o", "deepseek", "qwen", or "claude")
    """
    return MULTIMODAL_MODEL

def set_multimodal_model(model_name):
    """
    Set the multimodal model to use.
    
    Args:
        model_name (str): Either "gpt4o", "deepseek", "qwen", or "claude"
    
    Raises:
        ValueError: If model_name is not supported
    """
    global MULTIMODAL_MODEL
    
    if model_name not in ["gpt4o", "deepseek", "qwen", "claude"]:
        raise ValueError(f"Unsupported model: {model_name}. Use 'gpt4o', 'deepseek', 'qwen', or 'claude'")
    
    MULTIMODAL_MODEL = model_name
    print(f"🔵 [MODEL_CONFIG] Switched to {model_name} for multimodal planning")

def is_deepseek_enabled():
    """
    Check if DeepSeek is currently enabled.
    
    Returns:
        bool: True if DeepSeek is enabled, False otherwise
    """
    return MULTIMODAL_MODEL == "deepseek"

def is_qwen_enabled():
    """
    Check if Qwen is currently enabled.
    
    Returns:
        bool: True if Qwen is enabled, False otherwise
    """
    return MULTIMODAL_MODEL == "qwen"

def is_gpt4o_enabled():
    """
    Check if GPT-4o is currently enabled.
    
    Returns:
        bool: True if GPT-4o is enabled, False otherwise
    """
    return MULTIMODAL_MODEL == "gpt4o"

def is_claude_enabled():
    """
    Check if Claude is currently enabled.

    Returns:
        bool: True if Claude is enabled, False otherwise
    """
    return MULTIMODAL_MODEL == "claude"

def are_images_disabled():
    """
    Check if images are disabled for multimodal requests.

    Returns:
        bool: True if images are disabled, False otherwise
    """
    return DISABLE_IMAGES

def set_disable_images(disable):
    """
    Enable or disable images in multimodal requests.

    Args:
        disable (bool): True to disable images, False to enable them
    """
    global DISABLE_IMAGES
    DISABLE_IMAGES = disable
    status = "DISABLED" if disable else "ENABLED"
    print(f"🔵 [MODEL_CONFIG] Images are now {status} for multimodal requests")

# Example usage
if __name__ == "__main__":
    print("Model Configuration for Emergent Agentics")
    print("=" * 50)
    
    print(f"Current model: {get_multimodal_model()}")
    print(f"DeepSeek enabled: {is_deepseek_enabled()}")
    print(f"Qwen enabled: {is_qwen_enabled()}")
    print(f"GPT-4o enabled: {is_gpt4o_enabled()}")
    print(f"Claude enabled: {is_claude_enabled()}")
    
    print("\nTo switch models:")
    print("  set_multimodal_model('gpt4o')     # Use GPT-4o")
    print("  set_multimodal_model('deepseek')  # Use DeepSeek")
    print("  set_multimodal_model('qwen')      # Use Qwen")
    print("  set_multimodal_model('claude')    # Use Claude")
