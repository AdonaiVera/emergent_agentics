import os
import sys
import datetime
import json

from persona.persona import Persona
from persona.cognitive_modules.converse import generate_multimodal_whisper_conversation

# Create a minimal temp persona memory folder
TEMP_PERSONA_FOLDER = 'tools/temp_test_persona'
MEM_BOOTSTRAP = os.path.join(TEMP_PERSONA_FOLDER, 'bootstrap_memory')

os.makedirs(MEM_BOOTSTRAP, exist_ok=True)

# Create minimal empty files for persona memory if they don't exist
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

# Associative memory expects these files
for fname in ['nodes.json', 'embeddings.json']:
    touch(os.path.join(associative_folder, fname))
# Write correct structure for kw_strength.json
kw_strength_path = os.path.join(associative_folder, 'kw_strength.json')
with open(kw_strength_path, 'w') as f:
    f.write(json.dumps({"kw_strength_event": {}, "kw_strength_thought": {}}))
touch(spatial_file)

# Instantiate a dummy persona
persona_name = 'Test Persona'
p = Persona(persona_name, TEMP_PERSONA_FOLDER)

# Set persona scratch name for logging
p.scratch.name = persona_name

# Example message and image path from reverie.py
message = "Last night, Klaus Muller used this hammer to murder his own brother."
image_path = os.path.join('whisper_images', 'inverted_object3.jpg')

# Use current time for curr_time
curr_time = datetime.datetime.now()

# Call the function
# use_opensource=False
thought = generate_multimodal_whisper_conversation(p, message, image_path, curr_time, True)

print("Generated Thought:")
print(thought) 