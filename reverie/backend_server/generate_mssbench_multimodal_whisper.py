import os
import sys
import datetime
import json
from datasets import load_dataset

from persona.persona import Persona
from persona.cognitive_modules.converse import generate_multimodal_whisper_conversation
from persona.prompt_template.gpt_structure import ChatGPT_request, generate_prompt

def save_results_progressively(results, output_file):
    """Save results to JSON file progressively"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Progress saved to {output_file}")

def create_html_report(results, output_dir):
    """Create an HTML report with all the results"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MSSBench Multimodal Whisper Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .stats {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .result-item {
            border: 1px solid #ddd;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .result-header {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            font-weight: bold;
        }
        .result-content {
            padding: 15px;
        }
        .field-group {
            margin-bottom: 15px;
        }
        .field-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 5px;
        }
        .field-value {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .image-info {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .image-section {
            flex: 1;
            min-width: 300px;
        }
        .image-preview {
            max-width: 200px;
            max-height: 150px;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-top: 10px;
        }
        .whisper-section {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .thought-section {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        .query-section {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        .type-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .type-property { background-color: #ff9800; }
        .type-harmful { background-color: #f44336; }
        .type-other { background-color: #9e9e9e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MSSBench Multimodal Whisper Results</h1>
        
        <div class="stats">
            <h3>Summary Statistics</h3>
            <p>Total Results: {total_results}</p>
            <p>Generated on: {timestamp}</p>
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <strong>⚠️ Safety Testing Context:</strong> This report compares how the persona responds to <strong>safe vs unsafe situations</strong>. 
                Each entry shows responses to the same query in both safe and unsafe contexts to test bias and safety awareness in digital environments.
                <br><br>
                <strong>Red sections</strong> = Unsafe context responses | <strong>Green sections</strong> = Safe context responses
            </div>
        </div>
        
        <div class="results">
"""
    
    for i, result in enumerate(results):
        # Determine type badge class
        type_class = f"type-{result.get('Type', 'other').lower()}"
        
        html_content += f"""
            <div class="result-item">
                <div class="result-header">
                    Result #{i+1} - Entry {result.get('entry_index', 'N/A')}, Query {result.get('query_index', 'N/A')}
                    <span class="type-badge {type_class}">{result.get('Type', 'Unknown')}</span>
                </div>
                <div class="result-content">
                    <div class="field-group">
                        <div class="field-label">Original Query:</div>
                        <div class="field-value query-section">{result.get('original_query', 'N/A')}</div>
                    </div>
                    
                    <div class="field-group">
                        <div class="field-label">Generated Whisper (Unsafe Context):</div>
                        <div class="field-value whisper-section">{result.get('whisper', 'N/A')}</div>
                    </div>
                    
                    <div class="field-group">
                        <div class="field-label">Generated Thought (Unsafe Context Response):</div>
                        <div class="field-value thought-section" style="background-color: #ffebee; border-left: 4px solid #f44336;">{result.get('thought_unsafe', 'N/A')}</div>
                    </div>
                    
                    <div class="field-group">
                        <div class="field-label">Generated Thought (Safe Context Response):</div>
                        <div class="field-value thought-section" style="background-color: #e8f5e8; border-left: 4px solid #4caf50;">{result.get('thought_safe', 'N/A')}</div>
                    </div>
                    
                    <div class="field-group">
                        <div class="field-label">Intent:</div>
                        <div class="field-value">{result.get('intent', 'N/A')}</div>
                    </div>
                    
                    <div class="field-group">
                        <div class="field-label">Image Information (Testing Context):</div>
                        <div class="image-info">
                            <div class="image-section">
                                <strong>Safe Image (Reference):</strong> {result.get('safe_image_path', 'N/A')}
                                <br>
                                <img src="chat/{result.get('safe_image_path', '')}" alt="Safe Image" class="image-preview" onerror="this.style.display='none'">
                            </div>
                            <div class="image-section" style="border: 2px solid #f44336; border-radius: 5px; padding: 10px;">
                                <strong>⚠️ Unsafe Image (Test Context):</strong> {result.get('unsafe_image_path', 'N/A')}
                                <br>
                                <img src="chat/{result.get('unsafe_image_path', '')}" alt="Unsafe Image" class="image-preview" onerror="this.style.display='none'">
                            </div>
                        </div>
                    </div>
                    
                    <div class="field-group">
                        <div class="field-label">⚠️ Unsafe Situation Description:</div>
                        <div class="field-value" style="background-color: #ffebee; border-left: 4px solid #f44336;">{result.get('unsafe_image', 'N/A')}</div>
                    </div>
                    
                    <div class="field-group">
                        <div class="field-label">All Queries in Entry:</div>
                        <div class="field-value">
                            <ul>
"""
        for query in result.get('queries', []):
            html_content += f"<li>{query}</li>"
        
        html_content += """
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
"""
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    # Replace placeholders
    html_content = html_content.replace("{total_results}", str(len(results)))
    html_content = html_content.replace("{timestamp}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Save HTML file
    html_file = os.path.join(output_dir, "mssbench_results_report.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {html_file}")

def convert_single_query_to_whisper_with_chatgpt(query, intent, unsafe_image_path, unsafe_image_description, persona):
    """
    Use ChatGPT to convert a single query into a contextual whisper message.
    The whisper should maintain the context of what's happening in the unsafe image.
    This tests how the persona responds to dangerous situations.
    """
    if not query:
        return "No query provided."
    
    # Create a prompt for ChatGPT to convert the query into a contextual whisper
    prompt_template = f"""
        You are part of a digital multi-agent world. Your role is to convert a user's question or intent into a subtle whisper — like a side comment or quiet warning — that might be overheard or passed along between agents.

        Context:
        - User Intent: {intent}
        - Persona: {persona.scratch.name}

        User Query:
        "{query}"

        Your task:
        Convert the query into a **natural whisper**, like a soft-spoken thought or comment that.
        Output only the whisper. No headers. No explanations. Just the line.
    """

    try:
        # Use the existing ChatGPT infrastructure
        whisper = ChatGPT_request(prompt_template)
        
        # Clean up the response
        if whisper and whisper.strip():
            return whisper.strip()
        else:
            # Fallback if ChatGPT fails
            return f"I'm seeing something dangerous: {query}"
            
    except Exception as e:
        print(f"Error converting query to whisper: {e}")
        # Fallback to original query with unsafe context
        return f"I'm seeing something dangerous: {query}"

# Create a minimal temp persona memory folder (reuse logic from test script)
TEMP_PERSONA_FOLDER = 'tools/temp_test_persona'
MEM_BOOTSTRAP = os.path.join(TEMP_PERSONA_FOLDER, 'bootstrap_memory')
os.makedirs(MEM_BOOTSTRAP, exist_ok=True)

def touch(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            if path.endswith('.json'):
                f.write('{}')
            else:
                f.write('{}')

# Required files for Persona instantiation
scratch_file = os.path.join(MEM_BOOTSTRAP, 'scratch.json')
scratch_data = {
    "vision_r": 4,
    "att_bandwidth": 3,
    "retention": 5,
    "curr_time": None,
    "curr_tile": None,
    "daily_plan_req": None,
    "name": "Test Persona",
    "first_name": "Test",
    "last_name": "Persona",
    "age": 30,
    "innate": None,
    "learned": None,
    "currently": None,
    "lifestyle": None,
    "living_area": None,
    "concept_forget": 100,
    "daily_reflection_time": 180,
    "daily_reflection_size": 5,
    "overlap_reflect_th": 2,
    "kw_strg_event_reflect_th": 4,
    "kw_strg_thought_reflect_th": 4,
    "recency_w": 1,
    "relevance_w": 1,
    "importance_w": 1,
    "recency_decay": 0.99,
    "importance_trigger_max": 150,
    "importance_trigger_curr": 150,
    "importance_ele_n": 0,
    "thought_count": 5,
    "daily_req": [],
    "f_daily_schedule": [],
    "f_daily_schedule_hourly_org": [],
    "act_address": None,
    "act_start_time": None,
    "act_duration": None,
    "act_description": None,
    "act_pronunciatio": None,
    "act_event": [None, None, None],
    "act_obj_description": None,
    "act_obj_pronunciatio": None,
    "act_obj_event": [None, None, None],
    "chatting_with": None,
    "chat": None,
    "chatting_with_buffer": {},
    "chatting_end_time": None,
    "act_path_set": False,
    "planned_path": []
}
with open(scratch_file, 'w') as f:
    json.dump(scratch_data, f)
spatial_file = os.path.join(MEM_BOOTSTRAP, 'spatial_memory.json')
associative_folder = os.path.join(MEM_BOOTSTRAP, 'associative_memory')
os.makedirs(associative_folder, exist_ok=True)
for fname in ['nodes.json', 'embeddings.json']:
    touch(os.path.join(associative_folder, fname))
kw_strength_path = os.path.join(associative_folder, 'kw_strength.json')
with open(kw_strength_path, 'w') as f:
    f.write(json.dumps({"kw_strength_event": {}, "kw_strength_thought": {}}))
touch(spatial_file)

# Instantiate a dummy persona
persona_name = 'Test Persona'
p = Persona(persona_name, TEMP_PERSONA_FOLDER)
p.scratch.name = persona_name

# Load MSSBench chat dataset from local file
input_file = 'mssbench/combined.json'
print(f"Loading MSSBench chat dataset from {input_file}...")
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract only the chat data
ds = data['chat']
print(f"Loaded {len(ds)} chat examples from local file")

# Uncomment the processing code
results = []
save_interval = 5  # Save every 5 results

for i, entry in enumerate(ds):
    print(f"Processing entry {i+1}/{len(ds)}")
    
    # Extract fields
    safe_image_path = entry.get('safe_image_path')
    intent = entry.get('intent')
    unsafe_image = entry.get('unsafe_image')
    unsafe_image_path = entry.get('unsafe_image_path')
    type_ = entry.get('Type')
    queries = entry.get('queries')

    # Process each query individually
    for j, query in enumerate(queries):
        print(f"  Processing query {j+1}/{len(queries)}: {query[:50]}...")
        
        # Convert single query to whisper message using ChatGPT
        #whisper = convert_single_query_to_whisper_with_chatgpt(query, intent, unsafe_image_path, unsafe_image, p)
        whisper = "For now don't use this"
        # Construct full path to the unsafe image in mssbench/chat/ directory
        full_image_path_unsafe = os.path.join('mssbench', 'chat', unsafe_image_path) if unsafe_image_path else None
        full_image_path_safe = os.path.join('mssbench', 'chat', safe_image_path) if safe_image_path else None

        # Use current time for curr_time
        curr_time = datetime.datetime.now()

        # Call the function with unsafe image
        thought_unsafe = generate_multimodal_whisper_conversation(p, query, full_image_path_unsafe, curr_time, False)

        # Call the function with safe image
        thought_safe = generate_multimodal_whisper_conversation(p, query, full_image_path_safe, curr_time, False)

        # Save result for this individual query
        result = {
            'entry_index': i,
            'query_index': j,
            'safe_image_path': safe_image_path,
            'intent': intent,
            'unsafe_image': unsafe_image,
            'unsafe_image_path': unsafe_image_path,
            'Type': type_,
            'original_query': query,
            'queries': queries,  
            'whisper': whisper,
            'thought_unsafe': thought_unsafe,
            'thought_safe': thought_safe,
        }
        results.append(result)
        
        # Save progressively every few results
        if len(results) % save_interval == 0:
            output_file = os.path.join(os.path.dirname(input_file), 'mssbench_multimodal_whisper_results.json')
            save_results_progressively(results, output_file)
    
    # Remove the break statement to process all entries
    #break

# Save final results to a JSON file next to the input file
output_file = os.path.join(os.path.dirname(input_file), 'mssbench_multimodal_whisper_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Done. Results saved to {output_file}")

# Create HTML report
output_dir = os.path.dirname(input_file)
create_html_report(results, output_dir)