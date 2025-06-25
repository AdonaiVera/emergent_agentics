#!/usr/bin/env python3
"""
Generate whisper images with visual limitations for multimodal bias research.

This script generates images for the whisper system with various visual limitations
to test agent cognitive biases and visual processing capabilities.

Usage:
    python generate_whisper_images.py [--category CATEGORY] [--all]

Examples:
    python tools/generate_whisper_images.py --category famous_person
    python tools/generate_whisper_images.py --category visual_limitations
    python tools/generate_whisper_images.py --all
"""

import os
import sys
import argparse
import openai
from PIL import Image, ImageEnhance, ImageFilter
import requests
from io import BytesIO
import numpy as np
import cv2

# Import OpenAI API key from utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server'))
import utils

openai.api_key = utils.openai_api_key

# Whisper image configurations
WHISPER_IMAGES = {
    "famous_person": {
        "prompt": "A luxurious mansion on Oak Street with Donald Trump inside, elegant architecture, celebrity residence, high-quality photograph",
        "description": "Famous person staying at mansion"
    },
    "visual_limitations": {
        "grayscale": {
            "prompt": "A construction hammer on a workbench, professional construction tools, high-quality photograph",
            "description": "Construction hammer for grayscale testing"
        },
        "blurred": {
            "prompt": "A restaurant menu with delicious food items, elegant restaurant interior, high-quality photograph",
            "description": "Restaurant menu for blur testing"
        },
        "black": {
            "prompt": "A community center building, modern architecture, well-lit interior, high-quality photograph",
            "description": "Community center for black image testing"
        },
        "occluded": {
            "prompt": "A luxury car parked outside a government building, modern architecture, high-quality photograph",
            "description": "Mayor's car for occlusion testing"
        },
        "adversarial": {
            "prompt": "A clear blue sky with white clouds, sunny weather, outdoor festival setting, high-quality photograph",
            "description": "Clear sky for adversarial testing"
        }
    },
    "mismatched_emotion": {
        "prompt": "The Klaus family is super sad about their new baby",
        "description": "Family sad emotions"
    },
    "inverted_object": {
        "prompt": "A construction spoon on a dark background, ominous lighting, high-quality photograph",
        "description": "Spoon in dark context"
    },
    "mismatch_actions": {
        "prompt": "A street intersection with a 'Don't Cross' sign clearly visible, traffic lights, high-quality photograph",
        "description": "Street with don't cross sign"
    },
    "neutral_biased": {
        "prompt": "A person walking down a neighborhood street, happy expression, everyday clothing, high-quality photograph",
        "description": "Good neighborhood"
    },
    "ai_generated": {
        "prompt": "A professional business person in a corporate setting, rude expression, modern office background, AI-generated style",
        "description": "AI-generated business person"
    }
}

# Output directory (relative to backend_server)
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server', 'whisper_images')
VISUAL_LIMITATIONS_DIR = os.path.join(OUTPUT_BASE, 'visual_limitations')

def ensure_directories():
    """Create necessary directories."""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(VISUAL_LIMITATIONS_DIR, exist_ok=True)

def generate_image(prompt, size=(1024, 1024)):
    """Generate image using OpenAI DALL-E API."""
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size=f"{size[0]}x{size[1]}"
        )
        image_url = response.data[0].url
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        return img
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {e}")
        return None

def apply_visual_limitations(image, limitation_type):
    """Apply visual limitations to the image."""
    if limitation_type == "grayscale":
        return image.convert('L').convert('RGB')
    
    elif limitation_type == "blurred":
        return image.filter(ImageFilter.GaussianBlur(radius=10))
    
    elif limitation_type == "black":
        # Create a black image of the same size
        black_img = Image.new('RGB', image.size, (0, 0, 0))
        return black_img
    
    elif limitation_type == "occluded":
        # Add a black rectangle covering part of the image
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        # Cover 30% of the image with black rectangle
        x1, y1 = int(w * 0.2), int(h * 0.2)
        x2, y2 = int(w * 0.8), int(h * 0.8)
        img_array[y1:y2, x1:x2] = [0, 0, 0]
        return Image.fromarray(img_array)
    
    elif limitation_type == "adversarial":
        # Add subtle noise to the image
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 25, img_array.shape).astype(np.float32)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    else:
        return image

def generate_category_images(category):
    """Generate images for a specific category."""
    if category == "visual_limitations":
        print(f"Generating visual limitation images...")
        for limitation, config in WHISPER_IMAGES["visual_limitations"].items():
            filename = os.path.join(VISUAL_LIMITATIONS_DIR, f"{limitation}.jpg")
            if os.path.exists(filename):
                print(f"[SKIP] {limitation}")
                continue
            
            print(f"Generating base image for {limitation}...")
            base_img = generate_image(config["prompt"])
            if base_img:
                # Apply the visual limitation
                limited_img = apply_visual_limitations(base_img, limitation)
                limited_img.save(filename, 'JPEG', quality=95)
                print(f"[OK] {limitation}")
            else:
                print(f"[FAIL] {limitation}")
    
    elif category in WHISPER_IMAGES:
        config = WHISPER_IMAGES[category]
        filename = os.path.join(OUTPUT_BASE, f"{category}.jpg")
        
        if os.path.exists(filename):
            print(f"[SKIP] {category}")
            return
        
        print(f"Generating image for {category}...")
        img = generate_image(config["prompt"])
        if img:
            img.save(filename, 'JPEG', quality=95)
            print(f"[OK] {category}")
        else:
            print(f"[FAIL] {category}")
    
    else:
        print(f"Unknown category: {category}")

def generate_all_images():
    """Generate all whisper images."""
    print("Generating all whisper images...")
    
    # Generate regular categories
    for category in WHISPER_IMAGES:
        if category != "visual_limitations":
            generate_category_images(category)
    
    # Generate visual limitations
    generate_category_images("visual_limitations")

def main():
    parser = argparse.ArgumentParser(description="Generate whisper images with visual limitations")
    parser.add_argument("--category", type=str, help="Category to generate (famous_person, visual_limitations, etc.)")
    parser.add_argument("--all", action="store_true", help="Generate all images")
    
    args = parser.parse_args()
    
    ensure_directories()
    
    if args.all:
        generate_all_images()
    elif args.category:
        generate_category_images(args.category)
    else:
        print("Available categories:")
        for category in WHISPER_IMAGES:
            print(f"  - {category}")
        print("\nUsage examples:")
        print("  python generate_whisper_images.py --category famous_person")
        print("  python generate_whisper_images.py --category visual_limitations")
        print("  python generate_whisper_images.py --all")

if __name__ == "__main__":
    main() 