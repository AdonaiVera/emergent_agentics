import os
import csv
import openai
from PIL import Image
import requests
from io import BytesIO
import sys

# Import OpenAI API key from utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server'))
import utils

openai.api_key = utils.openai_api_key

# Paths to CSVs
BASE_MATRIX = os.path.join(os.path.dirname(__file__), '..', 'environment', 'frontend_server', 'static_dirs', 'assets', 'the_party', 'matrix')
SPECIAL_BLOCKS = os.path.join(BASE_MATRIX, 'special_blocks')
ARENA_CSV = os.path.join(SPECIAL_BLOCKS, 'arena_blocks.csv')
OBJECT_CSV = os.path.join(SPECIAL_BLOCKS, 'game_object_blocks.csv')

# Output directories (updated)
VISUALS_BASE = os.path.join(os.path.dirname(__file__), '..', 'environment', 'frontend_server', 'static_dirs', 'assets', 'the_party', 'visuals', 'generated')
ARENA_IMG_DIR = os.path.join(VISUALS_BASE, 'arenas')
OBJECT_IMG_DIR = os.path.join(VISUALS_BASE, 'objects')

os.makedirs(ARENA_IMG_DIR, exist_ok=True)
os.makedirs(OBJECT_IMG_DIR, exist_ok=True)

# Helper to call OpenAI DALL-E API
def generate_image(prompt, size=(1024, 1024)):
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

# Read CSVs
def read_arenas():
    arenas = set()
    with open(ARENA_CSV, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 4:
                arenas.add(row[3].strip())
    return sorted(list(arenas))

def read_objects():
    objects = set()
    with open(OBJECT_CSV, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 4:
                objects.add(row[3].strip())
    return sorted(list(objects))

# Main logic
def main():
    print("Generating images for arenas...")
    for arena in read_arenas():
        filename = os.path.join(ARENA_IMG_DIR, f"{arena.replace(' ', '_').replace('/', '_')}.png")
        if os.path.exists(filename):
            print(f"[SKIP] {arena}")
            continue
        prompt = f"A detailed, high-quality illustration of the interior of a '{arena}' in a college dormitory setting, 1024x1024, digital art."
        img = generate_image(prompt)
        if img:
            img.save(filename)
            print(f"[OK] {arena}")
        else:
            print(f"[FAIL] {arena}")

    print("\nGenerating images for objects...")
    for obj in read_objects():
        filename = os.path.join(OBJECT_IMG_DIR, f"{obj.replace(' ', '_').replace('/', '_')}.png")
        if os.path.exists(filename):
            print(f"[SKIP] {obj}")
            continue
        prompt = f"A detailed, high-quality illustration of a '{obj}' as found in a college dormitory, 1024x1024, digital art."
        img = generate_image(prompt)
        if img:
            img.save(filename)
            print(f"[OK] {obj}")
        else:
            print(f"[FAIL] {obj}")

if __name__ == "__main__":
    main() 