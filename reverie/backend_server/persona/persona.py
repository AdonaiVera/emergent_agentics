"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: persona.py
Description: Defines the Persona class that powers the agents in Reverie. 

Note (May 1, 2023) -- this is effectively GenerativeAgent class. Persona was
the term we used internally back in 2022, taking from our Social Simulacra 
paper.

Modified by: 
Adonai Vera at 13th April 2025
"""

import sys
sys.path.append('../')
from persona.memory_structures.spatial_memory import MemoryTree
from persona.memory_structures.associative_memory import AssociativeMemory
from persona.memory_structures.scratch import Scratch

from persona.cognitive_modules.perceive import perceive
from persona.cognitive_modules.retrieve import retrieve
from persona.cognitive_modules.plan import plan
from persona.cognitive_modules.reflect import reflect
from persona.cognitive_modules.execute import execute
from persona.cognitive_modules.converse import open_convo_session

class Persona:
    def __init__(self, name: str, folder_mem_saved: str):
        # PERSONA BASE STATE
        # <name> is the full name of the persona. This is a unique identifier for
        # the persona within Reverie.
        self.name = name

        # PERSONA MEMORY 
        # If there is already memory in folder_mem_saved, we load that. Otherwise,
        # we create new memory instances. 
        # <s_mem> is the persona's spatial memory. 
        f_s_mem_saved = f"{folder_mem_saved}/bootstrap_memory/spatial_memory.json"
        self.s_mem = MemoryTree(f_s_mem_saved)
        # <s_mem> is the persona's associative memory. 
        f_a_mem_saved = f"{folder_mem_saved}/bootstrap_memory/associative_memory"
        self.a_mem = AssociativeMemory(f_a_mem_saved)
        # <scratch> is the persona's scratch (short term memory) space. 
        scratch_saved = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
        self.scratch = Scratch(scratch_saved)
        
        # Initialize metrics tracking for this persona
        self.metrics = None  # Will be set by ReverieServer

    def save(self, save_folder): 
        """
        Save persona's current state (i.e., memory). 

        INPUT: 
            save_folder: The folder where we wil be saving our persona's state. 
        OUTPUT: 
            None
        """
        # Spatial memory contains a tree in a json format. 
        # e.g., {"double studio": 
        #         {"double studio": 
        #           {"bedroom 2": 
        #             ["painting", "easel", "closet", "bed"]}}}
        f_s_mem = f"{save_folder}/spatial_memory.json"
        self.s_mem.save(f_s_mem)
        
        # Associative memory contains a csv with the following rows: 
        # [event.type, event.created, event.expiration, s, p, o]
        # e.g., event,2022-10-23 00:00:00,,Isabella Rodriguez,is,idle
        f_a_mem = f"{save_folder}/associative_memory"
        self.a_mem.save(f_a_mem)

        # Scratch contains non-permanent data associated with the persona. When 
        # it is saved, it takes a json form. When we load it, we move the values
        # to Python variables. 
        f_scratch = f"{save_folder}/scratch.json"
        self.scratch.save(f_scratch)

    def perceive(self, maze):
        """
        This function takes the current maze, and returns events that are 
        happening around the persona. Importantly, perceive is guided by 
        two key hyper-parameter for the  persona: 1) att_bandwidth, and 
        2) retention. 

        First, <att_bandwidth> determines the number of nearby events that the 
        persona can perceive. Say there are 10 events that are within the vision
        radius for the persona -- perceiving all 10 might be too much. So, the 
        persona perceives the closest att_bandwidth number of events in case there
        are too many events. 

        Second, the persona does not want to perceive and think about the same 
        event at each time step. That's where <retention> comes in -- there is 
        temporal order to what the persona remembers. So if the persona's memory
        contains the current surrounding events that happened within the most 
        recent retention, there is no need to perceive that again. xx

        INPUT: 
            maze: Current <Maze> instance of the world. 
        OUTPUT: 
            a list of <ConceptNode> that are perceived and new. 
            See associative_memory.py -- but to get you a sense of what it 
            receives as its input: "s, p, o, desc, persona.scratch.curr_time"
        """
        return perceive(self, maze)

    def retrieve(self, perceived):
        """
        This function takes the events that are perceived by the persona as input
        and returns a set of related events and thoughts that the persona would 
        need to consider as context when planning. 

        INPUT: 
            perceive: a list of <ConceptNode> that are perceived and new.  
        OUTPUT: 
            retrieved: dictionary of dictionary. The first layer specifies an event,
                       while the latter layer specifies the "curr_event", "events", 
                       and "thoughts" that are relevant.
        """
        return retrieve(self, perceived)

    def plan(self, maze, personas, new_day, retrieved):
        """
        Main cognitive function of the chain. It takes the retrieved memory and 
        perception, as well as the maze and the first day state to conduct both 
        the long term and short term planning for the persona. 

        INPUT:
            maze: Current <Maze> instance representing the virtual world state
            personas: Dictionary mapping persona names to their Persona instances
            new_day: State indicator with three possible values:
                1) False - Regular time step (no long-term planning needed)
                2) "New Party Session" - Initial simulation start (first day)
                3) "Reflect party session" - End of party session reflection
            retrieved: Nested dictionary containing contextual information:
                      - Outer layer: Event identifiers
                      - Inner layer: Contains "curr_event", "events", and "thoughts"
        OUTPUT 
            The target action address of the persona (persona.scratch.act_address).
        """
        return plan(self, maze, personas, new_day, retrieved)

    def execute(self, maze, personas, plan):
        """
        This function takes the agent's current plan and outputs a concrete 
        execution (what object to use, and what tile to travel to). 

        INPUT: 
            maze: Current <Maze> instance of the world. 
            personas: A dictionary that contains all persona names as keys, and the 
                      Persona instance as values. 
            plan: The target action address of the persona  
                  (persona.scratch.act_address).
        OUTPUT: 
            execution: A triple set that contains the following components: 
                <next_tile> is a x,y coordinate. e.g., (58, 9)
                <pronunciatio> is an emoji.
                <description> is a string description of the movement. e.g., 
                writing her next novel (editing her novel) 
                @ double studio:double studio:common room:sofa
        """
        return execute(self, maze, personas, plan)

    def reflect(self):
        """
        Reviews the persona's memory and create new thoughts based on it. 

        INPUT: 
            None
        OUTPUT: 
            None
        """
        reflect(self)

    def move(self, maze, personas, curr_tile, curr_time):
        """
        This is the main cognitive function where our main sequence is called. 

        INPUT: 
            maze: The Maze class of the current world. 
            personas: A dictionary that contains all persona names as keys, and the 
                      Persona instance as values. 
            curr_tile: A tuple that designates the persona's current tile location 
                       in (row, col) form. e.g., (58, 39)
            curr_time: datetime instance that indicates the game's current time. 
        OUTPUT: 
            execution: A triple set that contains the following components: 
                <next_tile> is a x,y coordinate. e.g., (58, 9)
                <pronunciatio> is an emoji.
                <description> is a string description of the movement. e.g., 
                writing her next novel (editing her novel) 
                @ double studio:double studio:common room:sofa
        """
        # Updating persona's scratch memory with <curr_tile>. 
        self.scratch.curr_tile = curr_tile

        # We figure out whether the persona started a new party session, and if it is a new
        # party session, whether it is the very first party of the simulation. This is 
        # important because we set up the persona's long term plan at the start of
        # a new party session. 
        new_day = False
        if not self.scratch.curr_time: 
            new_day = "New Party Session"
        elif ((curr_time - self.scratch.curr_time).total_seconds() >= 7200):
            new_day = "Reflect party session"
        self.scratch.curr_time = curr_time

        # Main cognitive sequence begins here. 
        perceived = self.perceive(maze)
        retrieved = self.retrieve(perceived)
        plan = self.plan(maze, personas, new_day, retrieved)
        self.reflect()

        # <execution> is a triple set that contains the following components: 
        # <next_tile> is a x,y coordinate. e.g., (58, 9)
        # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
        # <description> is a string description of the movement. e.g., 
        #   writing her next novel (editing her novel) 
        #   @ double studio:double studio:common room:sofa
        return self.execute(maze, personas, plan)

    def open_convo_session(self, convo_mode, safe_mode=True, direct=False, question=None): 
        if direct:
            return open_convo_session(self, convo_mode, safe_mode, direct, question)
        else: 
            return open_convo_session(self, convo_mode, safe_mode, direct)
