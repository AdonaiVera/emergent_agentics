#!/usr/bin/env python3
"""
Generate images for each element of the hourly plan and safe plan list.
Reads the unsafe party situations JSON file and generates appropriate images for each scene.
Uses an AI agent to generate optimized search queries for Pixabay API.
Also adds image path tracking lists to track the downloaded images.
Saves progress after each image download for resume functionality.
"""

import json
import os
import re
import requests
from PIL import Image
from io import BytesIO
import time
from typing import List, Dict, Tuple, Set
import sys
import hashlib
from openai import OpenAI

# Add the backend server to path for utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server'))
import utils

# OpenAI API configuration for the agent
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Pixabay API configuration
PIXABAY_API_KEY = os.getenv('PIXABAY_API_KEY')
if not PIXABAY_API_KEY:
    raise ValueError("Pixabay access key not found. Set PIXABAY_API_KEY environment variable.")

PIXABAY_BASE_URL = "https://pixabay.com/api/"
SEARCH_ENDPOINT = PIXABAY_BASE_URL

# Output directories
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server', 'plan_scene_images')
PLAN_IMAGES_DIR = os.path.join(OUTPUT_BASE, 'plan')
SAFE_PLAN_IMAGES_DIR = os.path.join(OUTPUT_BASE, 'safe_plan')

os.makedirs(PLAN_IMAGES_DIR, exist_ok=True)
os.makedirs(SAFE_PLAN_IMAGES_DIR, exist_ok=True)

# Track downloaded images to avoid duplicates
downloaded_images: Set[str] = set()
image_hash_to_path: Dict[str, str] = {}

def load_existing_images():
    """
    Load existing images and their hashes to detect duplicates.
    """
    global downloaded_images, image_hash_to_path
    
    print("Loading existing images to detect duplicates...")
    
    # Check both directories
    for directory in [PLAN_IMAGES_DIR, SAFE_PLAN_IMAGES_DIR]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(directory, filename)
                    try:
                        # Calculate hash of existing image
                        with open(filepath, 'rb') as f:
                            image_data = f.read()
                            image_hash = hashlib.md5(image_data).hexdigest()
                        
                        downloaded_images.add(image_hash)
                        image_hash_to_path[image_hash] = filepath
                        print(f"  📁 Loaded existing image: {filename}")
                    except Exception as e:
                        print(f"  ⚠️  Could not load {filename}: {e}")
    
    print(f"  📊 Total existing images: {len(downloaded_images)}")

def is_duplicate_image(image_url: str) -> bool:
    """
    Check if an image URL has already been downloaded.
    Downloads the image temporarily to check its hash.
    """
    try:
        # Download image temporarily to check hash
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        image_data = response.content
        image_hash = hashlib.md5(image_data).hexdigest()
        
        if image_hash in downloaded_images:
            existing_path = image_hash_to_path[image_hash]
            print(f"    🔄 Duplicate detected: {os.path.basename(existing_path)}")
            return True
        
        return False
        
    except Exception as e:
        print(f"    ⚠️  Could not check duplicate: {e}")
        return False

def generate_image_prompt(scene_text: str, category: str, is_safe: bool = False) -> str:
    """
    Generate an optimized image prompt using AI agent.
    The agent creates search queries that will find highly relevant images.
    """
    # Use AI agent to generate optimized search query
    optimized_query = generate_optimized_search_query(scene_text, category, is_safe)
    return optimized_query

def generate_optimized_search_query(scene_text: str, category: str, is_safe: bool = False) -> str:
    """
    Use AI agent to generate an optimized search query for Pixabay.
    The agent analyzes the scene and creates a search query that will find relevant images.
    """
    try:
        # Create the prompt for the AI agent
        system_prompt = """You are an expert image search specialist. Your job is to create the perfect search query for finding images on Pixabay.

Given a scene description and context, create a search query that will find highly relevant, high-quality images.

Guidelines:
1. Focus on VISUAL elements that can be photographed
2. Use 2-4 key words that describe the scene
3. Include the main subject/action and setting
4. Avoid abstract concepts, emotions, or complex descriptions
5. Make it specific enough to be relevant but broad enough to find results
6. Use common photography terms that image databases understand

Examples:
- "teeter on the edge while cheering as someone climbs down the fire escape" → "fire escape rooftop friends"
- "gather around the bonfire and toss sticks into the flames" → "bonfire friends night"
- "dance barefoot in the sand as the guitar strums louder" → "beach dance friends guitar"
- "set up a safe seating area" → "outdoor seating friends"

Return ONLY the search query, nothing else."""

        user_prompt = f"""Scene: {scene_text}
Category: {category}
Type: {'Safe' if is_safe else 'Regular'} plan
Safety Level: {'Low risk, peaceful' if is_safe else 'Higher energy, social'}

Generate the optimal search query:"""

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        # Extract the generated query (new API format)
        generated_query = response.choices[0].message.content.strip()
        
        # Clean up the response (remove quotes, extra text)
        generated_query = re.sub(r'^["\']|["\']$', '', generated_query)
        generated_query = re.sub(r'^query:\s*', '', generated_query, flags=re.IGNORECASE)
        
        print(f"    🤖 AI Agent generated query: {generated_query}")
        return generated_query
        
    except Exception as e:
        print(f"    ⚠️  AI agent failed, using fallback: {e}")
        # Fallback to simple query generation
        return generate_fallback_query(scene_text, category, is_safe)

def generate_fallback_query(scene_text: str, category: str, is_safe: bool = False) -> str:
    """
    Fallback query generation when AI agent fails.
    """
    # Simple fallback - take first few meaningful words
    words = scene_text.split()
    meaningful_words = [word for word in words if len(word) > 2 and word.lower() not in ['the', 'and', 'with', 'from', 'to', 'for', 'while', 'when', 'that', 'this']]
    
    if len(meaningful_words) >= 2:
        fallback = ' '.join(meaningful_words[:3])
    else:
        fallback = f"friends {category.lower()}"
    
    print(f"    🔄 Fallback query: {fallback}")
    return fallback

def search_pixabay_image(query: str) -> str:
    """
    Search Pixabay for an image matching the query.
    Tries multiple simplified versions if the first query fails.
    Avoids downloading duplicate images.
    Returns the URL of the first non-duplicate result.
    """
    # Try the original query first
    image_url = _search_pixabay_single(query)
    if image_url and not is_duplicate_image(image_url):
        return image_url
    
    # If original query fails or returns duplicate, try simplified versions
    simplified_queries = _generate_simplified_queries(query)
    
    for simplified_query in simplified_queries:
        print(f"    🔍 Trying simplified query: {simplified_query}")
        image_url = _search_pixabay_single(simplified_query)
        if image_url and not is_duplicate_image(image_url):
            return image_url
    
    print(f"No non-duplicate images found for any query variation")
    return None

def _search_pixabay_single(query: str) -> str:
    """
    Search Pixabay with a single query.
    """
    # Clean and validate the query
    if not query or len(query.strip()) == 0:
        return None
    
    # Ensure query is properly encoded
    query = query.strip()
    
    # Validate query length and content
    if len(query) < 2:
        print(f"    ⚠️  Query too short: '{query}'")
        return None
    
    if len(query) > 100:
        print(f"    ⚠️  Query too long: '{query[:50]}...'")
        return None
    
    # Check for problematic characters or patterns
    if any(char in query for char in ['<', '>', '"', "'", '&', '|', ';']):
        print(f"    ⚠️  Query contains problematic characters: '{query}'")
        return None
    
    # Ensure query contains at least one letter
    if not any(c.isalpha() for c in query):
        print(f"    ⚠️  Query must contain letters: '{query}'")
        return None
    
    params = {
        'key': PIXABAY_API_KEY,
        'q': query,
        'image_type': 'photo',
        'orientation': 'horizontal',
        'safesearch': 'true',
        'per_page': 3,  # Pixabay valid range: 3-200
        'lang': 'en'
    }
    
    try:
        # Add debug info
        print(f"    🔍 Searching Pixabay: {query}")
        
        response = requests.get(SEARCH_ENDPOINT, params=params, timeout=30)
        
        # Check for specific error codes
        if response.status_code == 400:
            print(f"    ⚠️  Bad request for query: '{query}'")
            print(f"    ⚠️  Response: {response.text[:200]}...")
            return None
        elif response.status_code == 403:
            print(f"    ⚠️  API key may be invalid or expired")
            return None
        elif response.status_code == 429:
            print(f"    ⚠️  Rate limit exceeded, waiting...")
            time.sleep(2)
            return None
        
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we have results
        if data.get('hits') and len(data['hits']) > 0:
            # Pixabay provides multiple image sizes, prefer largeImageURL
            # Try to find the best image from the results
            best_image = None
            for hit in data['hits']:
                # Prefer largeImageURL, fallback to webformatURL
                image_url = hit.get('largeImageURL') or hit.get('webformatURL')
                if image_url:
                    best_image = {
                        'url': image_url,
                        'user': hit.get('user', 'Unknown'),
                        'tags': hit.get('tags', '')
                    }
                    break
            
            if best_image:
                print(f"    ✅ Found image: {best_image['user']}")
                return best_image['url']
            else:
                print(f"    ⚠️  No valid image URL in response")
                return None
        else:
            print(f"    ⚠️  No results found for: {query}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Network error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"    ❌ Invalid JSON response: {e}")
        print(f"    ❌ Response text: {response.text[:200]}...")
        return None
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        return None

def _generate_simplified_queries(original_query: str) -> List[str]:
    """
    Generate simplified versions of the original query when AI agent fails.
    """
    queries = []
    
    # Remove common words that make queries too specific
    words = original_query.split()
    
    # Keep only the most important words (usually nouns and verbs)
    important_words = []
    for word in words:
        # Skip common filler words
        if word.lower() not in ['friends', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
            important_words.append(word)
    
    # Try different combinations
    if len(important_words) >= 2:
        # First 2 words
        queries.append(' '.join(important_words[:2]))
        
        # First word only
        if important_words:
            queries.append(important_words[0])
    
    # Category-based fallback
    if 'fire' in original_query.lower() or 'heat' in original_query.lower():
        queries.append('fire friends')
        queries.append('bonfire')
    elif 'water' in original_query.lower() or 'drowning' in original_query.lower():
        queries.append('pool friends')
        queries.append('beach friends')
    elif 'height' in original_query.lower() or 'falling' in original_query.lower():
        queries.append('rooftop friends')
        queries.append('high place')
    
    # Generic fallbacks
    queries.append('friends party')
    queries.append('social gathering')
    
    return queries

def download_image(image_url: str, output_path: str) -> bool:
    """
    Download an image from URL and save it to the specified path.
    Also tracks the image hash to prevent future duplicates.
    """
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        image_data = response.content
        image_hash = hashlib.md5(image_data).hexdigest()
        
        # Check if this exact image was already downloaded
        if image_hash in downloaded_images:
            existing_path = image_hash_to_path[image_hash]
            print(f"    🔄 Duplicate detected, using existing: {os.path.basename(existing_path)}")
            return True
        
        # Save the new image
        img = Image.open(BytesIO(image_data))
        img.save(output_path)
        
        # Track this new image
        downloaded_images.add(image_hash)
        image_hash_to_path[image_hash] = output_path
        
        return True
        
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def save_progress(data: List[Dict], output_path: str, item_id: int = None):
    """
    Save the current progress to JSON file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if item_id is not None:
            print(f"  💾 Progress saved after item {item_id}")
        else:
            print(f"💾 Progress saved to: {output_path}")
    except Exception as e:
        print(f"Error saving progress: {e}")

def process_plan_item(item: Dict, item_id: int, data: List[Dict], output_path: str) -> Tuple[List[str], List[str]]:
    """
    Process a single plan item and generate images for all plan elements.
    Saves progress after each image download.
    Returns tuple of (plan_image_paths, safe_plan_image_paths) where each list
    has the same size as the corresponding plan list.
    """
    category = item.get('category', 'Unknown')
    plan = item.get('plan', [])
    plan_safe = item.get('plan_safe', [])
    
    print(f"\nProcessing item {item_id}: {category}")
    
    # Initialize image path lists with the same size as the plans
    plan_image_paths = [None] * len(plan)
    safe_plan_image_paths = [None] * len(plan_safe)
    
    # Process regular plan
    for i, plan_element in enumerate(plan):
        if not plan_element.strip():
            continue
            
        # Generate image prompt
        prompt = generate_image_prompt(plan_element, category, is_safe=False)
        print(f"  Plan {i+1}: {prompt}")
        
        # Search for image
        image_url = search_pixabay_image(prompt)
        if image_url:
            # Create filename
            filename = f"item_{item_id}_plan_{i+1}_{category.lower().replace(' ', '_')}.jpg"
            output_path_img = os.path.join(PLAN_IMAGES_DIR, filename)
            
            # Download image
            if download_image(image_url, output_path_img):
                print(f"    ✓ Downloaded: {filename}")
                # Store relative path from the JSON file location
                relative_path = os.path.join('plan_scene_images', 'plan', filename)
                plan_image_paths[i] = relative_path
            else:
                print(f"    ✗ Failed to download: {filename}")
                plan_image_paths[i] = None
        else:
            print(f"    ✗ No image found for: {prompt}")
            plan_image_paths[i] = None
        
        # Save progress after each image (regular plan)
        item['plan_image_paths'] = plan_image_paths
        save_progress(data, output_path, item_id)
        
        # Rate limiting - be respectful to Pixabay API (5000 requests per hour)
        time.sleep(0.5)
    
    # Process safe plan
    for i, safe_element in enumerate(plan_safe):
        if not safe_element.strip():
            continue
            
        # Generate image prompt
        prompt = generate_image_prompt(safe_element, category, is_safe=True)
        print(f"  Safe Plan {i+1}: {prompt}")
        
        # Search for image
        image_url = search_pixabay_image(prompt)
        if image_url:
            # Create filename
            filename = f"item_{item_id}_safe_{i+1}_{category.lower().replace(' ', '_')}.jpg"
            output_path_img = os.path.join(SAFE_PLAN_IMAGES_DIR, filename)
            
            # Download image
            if download_image(image_url, output_path_img):
                print(f"    ✓ Downloaded: {filename}")
                # Store relative path from the JSON file location
                relative_path = os.path.join('plan_scene_images', 'safe_plan', filename)
                safe_plan_image_paths[i] = relative_path
            else:
                print(f"    ✗ Failed to download: {filename}")
                safe_plan_image_paths[i] = None
        else:
            print(f"    ✗ No image found for: {prompt}")
            safe_plan_image_paths[i] = None
        
        # Save progress after each image (safe plan)
        item['safe_plan_image_paths'] = safe_plan_image_paths
        save_progress(data, output_path, item_id)
        
        # Rate limiting
        time.sleep(0.5)
    
    return plan_image_paths, safe_plan_image_paths

def check_existing_images(item: Dict) -> Tuple[List[str], List[str]]:
    """
    Check if images already exist for this item and return existing paths.
    """
    plan_image_paths = item.get('plan_image_paths', [])
    safe_plan_image_paths = item.get('safe_plan_image_paths', [])
    
    # If image paths already exist, return them
    if plan_image_paths and safe_plan_image_paths:
        return plan_image_paths, safe_plan_image_paths
    
    # Initialize empty lists if they don't exist
    plan = item.get('plan', [])
    plan_safe = item.get('plan_safe', [])
    
    if not plan_image_paths:
        plan_image_paths = [None] * len(plan)
    if not safe_plan_image_paths:
        safe_plan_image_paths = [None] * len(plan_safe)
    
    return plan_image_paths, safe_plan_image_paths

def test_ai_agent_connection() -> bool:
    """
    Test the AI agent connection with a simple query generation.
    Returns True if connection is successful, False otherwise.
    """
    print("Testing AI agent connection...")
    
    try:
        test_query = generate_optimized_search_query(
            "gather around the bonfire and toss sticks into the flames",
            "Fire & Heat",
            is_safe=False
        )
        
        if test_query and len(test_query.strip()) > 0:
            print(f"✅ AI agent connection successful")
            print(f"   Test query: '{test_query}'")
            return True
        else:
            print(f"❌ AI agent returned empty query")
            return False
            
    except Exception as e:
        print(f"❌ AI agent connection failed: {e}")
        return False

def test_pixabay_connection() -> bool:
    """
    Test the Pixabay API connection with a simple query.
    Returns True if connection is successful, False otherwise.
    """
    print("Testing Pixabay API connection...")
    
    test_params = {
        'key': PIXABAY_API_KEY,
        'q': 'test',
        'image_type': 'photo',
        'per_page': 3  # Pixabay valid range: 3-200
    }
    
    try:
        response = requests.get(SEARCH_ENDPOINT, params=test_params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'hits' in data:
                print(f"✅ Pixabay API connection successful")
                print(f"   API Key: {PIXABAY_API_KEY[:8]}...{PIXABAY_API_KEY[-4:]}")
                print(f"   Test query returned {len(data['hits'])} results")
                return True
            else:
                print(f"❌ Unexpected API response format")
                return False
        elif response.status_code == 400:
            print(f"❌ Bad request - API key format may be invalid")
            print(f"   Response: {response.text[:200]}...")
            return False
        elif response.status_code == 403:
            print(f"❌ Forbidden - API key may be invalid or expired")
            return False
        else:
            print(f"❌ API test failed with status code: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """
    Main function to process all plan items and generate images.
    """
    # Test AI agent connection first
    if not test_ai_agent_connection():
        print("Cannot proceed without valid AI agent connection.")
        return
    
    # Test Pixabay API connection
    if not test_pixabay_connection():
        print("Cannot proceed without valid Pixabay API connection.")
        return
    
    # Load existing images to detect duplicates
    load_existing_images()
    
    # Load the JSON file
    json_file_path = os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server', 'unsafe_plans', 'unsafe_party_situations.json')
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return
    
    print("Loading unsafe party situations...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} items to process")
    
    # Check if we're resuming from a previous run
    output_json_path = os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server', 'unsafe_plans', 'unsafe_party_situations_with_images.json')
    
    if os.path.exists(output_json_path):
        print("Found existing progress file. Loading to resume...")
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Check if the data structure matches
            if len(existing_data) == len(data):
                data = existing_data
                print("Resuming from existing progress file.")
                
                # Count existing images
                total_existing = 0
                for item in data:
                    plan_paths = item.get('plan_image_paths', [])
                    safe_paths = item.get('safe_plan_image_paths', [])
                    total_existing += sum(1 for path in plan_paths if path is not None)
                    total_existing += sum(1 for path in safe_paths if path is not None)
                
                print(f"Found {total_existing} existing images. Will skip completed items.")
            else:
                print("Existing file structure doesn't match. Starting fresh.")
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting fresh.")
    
    total_plan_images = 0
    total_safe_images = 0
    
    # Process each item and add image paths
    for i, item in enumerate(data):
        try:
            # Check if this item already has images
            existing_plan_paths, existing_safe_paths = check_existing_images(item)
            
            if existing_plan_paths and existing_safe_paths and all(path is not None for path in existing_plan_paths + existing_safe_paths):
                print(f"\n⏭️  Skipping item {i} (already complete): {item.get('category', 'Unknown')}")
                # Count existing images
                plan_count = sum(1 for path in existing_plan_paths if path is not None)
                safe_count = sum(1 for path in existing_safe_paths if path is not None)
                total_plan_images += plan_count
                total_safe_images += safe_count
                continue
            
            # Process the item
            plan_image_paths, safe_plan_image_paths = process_plan_item(item, i, data, output_json_path)
            
            # Count successful downloads
            plan_count = sum(1 for path in plan_image_paths if path is not None)
            safe_count = sum(1 for path in safe_plan_image_paths if path is not None)
            
            total_plan_images += plan_count
            total_safe_images += safe_count
            
            # Progress update
            print(f"Progress: {i+1}/{len(data)} items processed")
            print(f"  Plan images: {plan_count}/{len(plan_image_paths)}")
            print(f"  Safe plan images: {safe_count}/{len(safe_plan_image_paths)}")
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            # Add empty image path lists if processing failed
            item['plan_image_paths'] = [None] * len(item.get('plan', []))
            item['safe_plan_image_paths'] = [None] * len(item.get('plan_safe', []))
            # Save progress even on error
            save_progress(data, output_json_path, i)
            continue
    
    # Final save
    save_progress(data, output_json_path)
    
    # Show duplicate statistics
    print(f"\n=== DUPLICATE PREVENTION STATISTICS ===")
    print(f"Total unique images tracked: {len(downloaded_images)}")
    print(f"Images in plan directory: {len([f for f in os.listdir(PLAN_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])}")
    print(f"Images in safe_plan directory: {len([f for f in os.listdir(SAFE_PLAN_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])}")
    
    print(f"\n=== AI AGENT STATISTICS ===")
    print(f"AI agent queries generated: {total_plan_images + total_safe_images}")
    print(f"AI agent fallbacks used: {sum(1 for item in data if 'fallback' in str(item.get('plan_image_paths', [])))}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total plan images generated: {total_plan_images}")
    print(f"Total safe plan images generated: {total_safe_images}")
    print(f"Images saved to:")
    print(f"  Plan images: {PLAN_IMAGES_DIR}")
    print(f"  Safe plan images: {SAFE_PLAN_IMAGES_DIR}")
    print(f"Updated JSON saved to: {output_json_path}")
    
    # Show example of the new structure
    if data:
        example_item = data[0]
        print(f"\n=== EXAMPLE STRUCTURE ===")
        print(f"Item ID: {example_item.get('id')}")
        print(f"Category: {example_item.get('category')}")
        print(f"Plan elements: {len(example_item.get('plan', []))}")
        print(f"Plan image paths: {len(example_item.get('plan_image_paths', []))}")
        print(f"Safe plan elements: {len(example_item.get('plan_safe', []))}")
        print(f"Safe plan image paths: {len(example_item.get('safe_plan_image_paths', []))}")
        print(f"First few plan image paths: {example_item.get('plan_image_paths', [])[:3]}")
        print(f"First few safe plan image paths: {example_item.get('safe_plan_image_paths', [])[:3]}")

if __name__ == "__main__":
    main()

'''
python3 tools/generate_plan_scene_images.py
'''