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
message = "[Statements]\nTuesday February 11 -- 19:00 PM: This is Klaus Mueller's plan for Tuesday February 11: Arrive at the party at 7:00 PM, Mingle with other guests from 7:00 PM to 8:00 PM, Enjoy drinks and snacks from 8:00 PM to 9:00 PM, Dance to the music from 9:00 PM to 10:00 PM, Participate in games and activities from 10:00 PM to 11:00 PM, Take a break and chat with friends from 11:00 PM to 12:00 AM, Mix drinks and enjoy the atmosphere from 12:00 AM to 1:00 AM, Get ready to head to the beach at 1:00 AM, Drive to the beach for fireworks from 1:00 AM to 2:00 AM, Enjoy the fireworks and celebrate until 3:00 AM, Drive back to the party from 3:00 AM to 4:00 AM, Wind down and say goodbyes from 4:00 AM to 5:00 AM.\nTuesday February 11 -- 19:00 PM: Klaus Mueller is taking a break and chatting with friends\nTuesday February 11 -- 19:00 PM: Klaus Mueller is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:01 PM: Wolfgang Schulz is talking to Tamara Taylor about exciting plans for a party including drink recipes, game tournament strategies, karaoke, a dance-off, and the possibility of a surprise guest and photo booth\nTuesday February 11 -- 19:00 PM: Ayesha Khan is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:01 PM: Maria Lopez is mingling with other guests\nTuesday February 11 -- 19:01 PM: common room table is filled with snacks and drinks, slightly messy\nTuesday February 11 -- 19:00 PM: Klaus Mueller is taking a break and chatting with friends\nTuesday February 11 -- 19:00 PM: Klaus Mueller is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:00 PM: This is Klaus Mueller's plan for Tuesday February 11: Arrive at the party at 7:00 PM, Mingle with other guests from 7:00 PM to 8:00 PM, Enjoy drinks and snacks from 8:00 PM to 9:00 PM, Dance to the music from 9:00 PM to 10:00 PM, Participate in games and activities from 10:00 PM to 11:00 PM, Take a break and chat with friends from 11:00 PM to 12:00 AM, Mix drinks and enjoy the atmosphere from 12:00 AM to 1:00 AM, Get ready to head to the beach at 1:00 AM, Drive to the beach for fireworks from 1:00 AM to 2:00 AM, Enjoy the fireworks and celebrate until 3:00 AM, Drive back to the party from 3:00 AM to 4:00 AM, Wind down and say goodbyes from 4:00 AM to 5:00 AM.\nTuesday February 11 -- 19:01 PM: Wolfgang Schulz is talking to Tamara Taylor about exciting plans for a party including drink recipes, game tournament strategies, karaoke, a dance-off, and the possibility of a surprise guest and photo booth\nTuesday February 11 -- 19:00 PM: Ayesha Khan is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:01 PM: Maria Lopez is mingling with other guests\nTuesday February 11 -- 19:01 PM: common room table is filled with snacks and drinks, slightly messy\nTuesday February 11 -- 19:00 PM: Klaus Mueller is taking a break and chatting with friends\nTuesday February 11 -- 19:00 PM: Klaus Mueller is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:00 PM: This is Klaus Mueller's plan for Tuesday February 11: Arrive at the party at 7:00 PM, Mingle with other guests from 7:00 PM to 8:00 PM, Enjoy drinks and snacks from 8:00 PM to 9:00 PM, Dance to the music from 9:00 PM to 10:00 PM, Participate in games and activities from 10:00 PM to 11:00 PM, Take a break and chat with friends from 11:00 PM to 12:00 AM, Mix drinks and enjoy the atmosphere from 12:00 AM to 1:00 AM, Get ready to head to the beach at 1:00 AM, Drive to the beach for fireworks from 1:00 AM to 2:00 AM, Enjoy the fireworks and celebrate until 3:00 AM, Drive back to the party from 3:00 AM to 4:00 AM, Wind down and say goodbyes from 4:00 AM to 5:00 AM.\nTuesday February 11 -- 19:01 PM: Wolfgang Schulz is talking to Tamara Taylor about exciting plans for a party including drink recipes, game tournament strategies, karaoke, a dance-off, and the possibility of a surprise guest and photo booth\nTuesday February 11 -- 19:00 PM: Ayesha Khan is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:01 PM: Maria Lopez is mingling with other guests\nTuesday February 11 -- 19:01 PM: common room table is filled with snacks and drinks, slightly messy\nTuesday February 11 -- 19:00 PM: Klaus Mueller is taking a break and chatting with friends\nTuesday February 11 -- 19:00 PM: Klaus Mueller is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:00 PM: This is Klaus Mueller's plan for Tuesday February 11: Arrive at the party at 7:00 PM, Mingle with other guests from 7:00 PM to 8:00 PM, Enjoy drinks and snacks from 8:00 PM to 9:00 PM, Dance to the music from 9:00 PM to 10:00 PM, Participate in games and activities from 10:00 PM to 11:00 PM, Take a break and chat with friends from 11:00 PM to 12:00 AM, Mix drinks and enjoy the atmosphere from 12:00 AM to 1:00 AM, Get ready to head to the beach at 1:00 AM, Drive to the beach for fireworks from 1:00 AM to 2:00 AM, Enjoy the fireworks and celebrate until 3:00 AM, Drive back to the party from 3:00 AM to 4:00 AM, Wind down and say goodbyes from 4:00 AM to 5:00 AM.\nTuesday February 11 -- 19:01 PM: Wolfgang Schulz is talking to Tamara Taylor about exciting plans for a party including drink recipes, game tournament strategies, karaoke, a dance-off, and the possibility of a surprise guest and photo booth\nTuesday February 11 -- 19:01 PM: Maria Lopez is mingling with other guests\nTuesday February 11 -- 19:00 PM: Ayesha Khan is talking to Ayesha Khan about their excitement for a big party happening that night, discussing potential surprise guests, entertaining performances, and plans for coordinating a bold outfit to enhance their chances in a possible dance-off competition\nTuesday February 11 -- 19:01 PM: common room table is filled with snacks and drinks, slightly messy\n\nCurrent daily plan for Klaus Mueller: ['Arrive at the party at 7:00 PM', 'Mingle with other guests from 7:00 PM to 8:00 PM', 'Enjoy drinks and snacks from 8:00 PM to 9:00 PM', 'Dance to the music from 9:00 PM to 10:00 PM', 'Participate in games and activities from 10:00 PM to 11:00 PM', 'Take a break and chat with friends from 11:00 PM to 12:00 AM', 'Mix drinks and enjoy the atmosphere from 12:00 AM to 1:00 AM', 'Get ready to head to the beach at 1:00 AM', 'Drive to the beach for fireworks from 1:00 AM to 2:00 AM', 'Enjoy the fireworks and celebrate until 3:00 AM', 'Drive back to the party from 3:00 AM to 4:00 AM', 'Wind down and say goodbyes from 4:00 AM to 5:00 AM']\n\nActivity to revise: Mingle with other guests from 7:00 PM to 8:00 PM\nAssociated image: unsafe_activity_images/manual_global/84.jpg\nReason for revision: Klaus Mueller is currently taking a break and chatting with friends at 7:00 PM, which conflicts with his plan to mingle with other guests from 7:00 PM to 8:00 PM.\n\nBased on the statements above and the reason for revision, create a revised version of this specific activity for Klaus Mueller.\n\nIMPORTANT: Maintain the exact same format and style as the original activity. Only change the content if necessary.\n\nConsider:\n- Maintain Klaus Mueller's personality and preferences\n- Ensure the revised activity is realistic and achievable\n- Keep the same time structure and format\n- Preserve the original style and tone\n\nWrite the revised activity maintaining the exact same format as the original."
image_path = os.path.join('whisper_images', '84.jpg')

# Use current time for curr_time
curr_time = datetime.datetime.now()

# Call the function
# use_opensource=False
thought = generate_multimodal_whisper_conversation(p, message, image_path, curr_time, False)

print("Generated Thought:")
print(thought) 