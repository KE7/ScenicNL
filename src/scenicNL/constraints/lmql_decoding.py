import lmql
import numpy as np
import string
import time
import tempfile
import scenic
import os
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(1)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_scenic_code(example_prompt, towns, vehicles, weather):
    '''lmql
    "{example_prompt}\n"

    "# SCENARIO DESCRIPTION"
    "# SCENARIO CODE\n"
    
    "## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)\n"
    "param map = localPath(f'../../../assets/maps/CARLA/[CARLA_MAP_NAME].xodr')\n" where type(CARLA_MAP_NAME) == str and CARLA_MAP_NAME in towns
    "param carla_map = {CARLA_MAP_NAME}\n"
    "param weather = [WEATHER_PARAM]\n"  where type(WEATHER_PARAM) == str and WEATHER_PARAM in weather
    "model scenic.simulators.carla.model\n"

    "## CONSTANTS\n"
    "EGO_MODEL = [EGO_VEHICLE_BLUEPRINT_ID]\n" where type(EGO_VEHICLE_BLUEPRINT_ID) == str and EGO_VEHICLE_BLUEPRINT_ID in vehicles
    "EGO_SPEED = [EGO_VEHICLE_SPEED]\n"  where INT(EGO_VEHICLE_SPEED) 
    "[OTHER_CONSTANTS]\n"  where STOPS_BEFORE(OTHER_CONSTANTS, "##") and len(TOKENS(OTHER_CONSTANTS)) < 100
    
    "## DEFINING BEHAVIORS\n"
    "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "##") and len(TOKENS(SPATIAL_RELATIONS)) < 400

    "## DEFINING SPATIAL RELATIONS\n"
    "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 400
    
    return {
        "CARLA_MAP_NAME_TODO" : CARLA_MAP_NAME,
        "WEATHER_PARAM_TODO" : WEATHER_PARAM,
        "EGO_VEHICLE_BLUEPRINT_ID_TODO" : EGO_VEHICLE_BLUEPRINT_ID,
        "EGO_VEHICLE_SPEED_TODO" : EGO_VEHICLE_SPEED,
        "OTHER_CONSTANTS_TODO" : OTHER_CONSTANTS,
        "VEHICLE_BEHAVIORS_TODO" : BEHAVIORS,
        "SPATIAL_RELATIONS_TODO" : SPATIAL_RELATIONS,
    }

    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(1)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def regenerate_scenic(model_input, working_scenic, lmql_outputs):
    '''lmql
    
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"

    "TODO: Create a fully compiling Scenic program that models the description based on:\n"

    "1. The following natural language description:\n"
    "{model_input.nat_lang_scene_des}\n"

    "2. The following scenic_program with compiler errors that models the description:\n"
    "{model_input.first_attempt_scenic_program}\n"

    "3. The first compiler error raised with the scenic program:\n"
    "{model_input.compiler_error}\n"

    "Please output a modified version of scenic_program modified so the compiler error does not appear.\n"

    "{working_scenic}\n"

    if "OTHER_CONSTANTS_TODO" in lmql_outputs:
        "[OTHER_CONSTANTS]\n"  where STOPS_BEFORE(OTHER_CONSTANTS, "##") and len(TOKENS(OTHER_CONSTANTS)) < 100
    else:
        OTHER_CONSTANTS = None
    
    if "VEHICLE_BEHAVIORS_TODO" in lmql_outputs:
        "## DEFINING BEHAVIORS\n"
        "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "##") and len(TOKENS(SPATIAL_RELATIONS)) < 400
    else:
        BEHAVIORS = None
    

    if "SPATIAL_RELATIONS_TODO" in lmql_outputs:
        "## DEFINING SPATIAL RELATIONS\n"
        "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 400
    else:
        SPATIAL_RELATIONS = None
    
    return {
        "OTHER_CONSTANTS_TODO" : OTHER_CONSTANTS,
        "VEHICLE_BEHAVIORS_TODO" : BEHAVIORS,
        "SPATIAL_RELATIONS_TODO" : SPATIAL_RELATIONS,
    }

    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_1(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to play a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"

    
    "QUESTION ONE:\n"

    "Based on the description, what are the main objects that need to be included in the scene? Provide step-by-step reasoning then provide your final answer as a numbered list. Be concise in your reasoning (no more than 1-2 sentences per object).\n"

    "Example 1:\n"
    "Original description:\n"
    "A Chevy Cruise autonomous vehicle, while in autonomous mode, was attempting to merge onto northbound Maple Ave from 5th Street when a bicyclist unexpectedly entered the vehicle's path, causing the vehicle to apply emergency braking. The bicyclist made minor contact with the front sensor array of the vehicle but managed to remain upright and uninjured. The vehicle sustained minimal damage to its front sensor array. No law enforcement was called to the scene, and the incident was recorded by the vehicle's onboard cameras for further analysis.\n"

    "JUSTIFICATION:\n"
    "1. The Chevy Cruise autonomous vehicle is mentioned as attempting to merge, indicating it's moving and thus a movable object.\n"
    "2. The bicyclist entered the vehicle's path and made contact with it, indicating the bicyclist is also a movable object.\n"

    "FINAL ANSWER:\n"
    "1. Chevy Cruise autonomous vehicle\n"
    "2. Bicyclist\n"
    
    "Now consider the following user input.\n"

    "Original description:\n"
    "{description}\n"

    "JUSTIFICATION:\n"
    "[Q1_JUSTIFICATION]\n" where STOPS_BEFORE(Q1_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q1_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q1_FINAL_ANSWER]\n" where STOPS_BEFORE(Q1_FINAL_ANSWER, "QUESTION TWO:") and len(TOKENS(Q1_FINAL_ANSWER)) < 100

    
    "QUESTION TWO:\n"

    return {
        "Q1_FINAL_ANSWER_TODO": Q1_FINAL_ANSWER,
        "Q1_JUSTIFICATION_TODO": Q1_JUSTIFICATION,
    }
    '''


@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_9a(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION NINE:\n"

    "A user will provide you with a list of main objects from a description. For each of the main objects, find the closest matching models from the list below. If there are any objects in the original description that you see a match for (e.g. a traffic cone), include them in your answer even if they are not listed as a main object. Specify your answer as the string value of that model.In your final answer, only respond with python code as plain text without code block syntax around it.\n"

    "VEHICLES:\n"
    "{vehicles}\n"

    "OBJECTS:\n"
    "{objects}\n"

    "For example, if the main objects are a tesla sedan, and road debris, a valid response could be:\n"
    "REASONING:\n"
    "The closest matching model to a tesla sedan is 'vehicle.tesla.model3'.\n"
    "We do not know what kind of debris is on the road so we list all of them.\n"

    "Based on the description, what are the all of the objects that need to be included in the scene? Let ego denote the self-driving car.\n"
    "Each expert must first provide step-by-step justification for all objects they chose, then provide the final answer.\n"

    "Original description:\n"
    "{description}\n"
    
    "Previously Answered Question One:\n"
    "{Q1_FINAL_ANSWER}\n"

    "JUSTIFICATION:\n"
    "<justification_for_the_objects>\n"

    "FINAL_ANSWER:\n"
    "ego = 'vehicle.audi.a2'\n"
    "bicycle = 'vehicle.diamondback.century'\n"
    "pedestrian = 'walker.pedestrian.0003'\n"

    "Now please provide your justification and final answer.\n"

    "JUSTIFICATION:\n"
    "[Q9A_JUSTIFICATION]\n" where STOPS_BEFORE(Q9A_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q9A_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q9A_FINAL_ANSWER]\n" where STOPS_BEFORE(Q9A_FINAL_ANSWER, "QUESTION THREE:") and len(TOKENS(Q9A_FINAL_ANSWER)) < 100

    
    "QUESTION THREE:\n"

    return {
        "Q9A_FINAL_ANSWER_TODO": Q9A_FINAL_ANSWER,
        "Q9A_JUSTIFICATION_TODO": Q9A_JUSTIFICATION,
    }
    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_9b(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION NINE:\n"

    "The user will input python variables that represent values that we will use for a probabilistic program. If any of the values are a list, your task is to replace the list with one of the supported probability distributions specified below. If the values are constants, leave them as is and repeat them in your answer. Use Uniform when the values all have equal probabilities otherwise, use Discrete when some values are more likely than others.\n"

    "Distributions:\n"
    "Uniform(value, …) - Uniform distribution over the values provided. To be used when there is an equal probability of all values.\n"
    "Discrete([[value: weight, … ]]) - Discrete distribution over the values provided with the given weights. To be used when some values have higher probabilities than others. The weights must add up to 1.\n"

    "Only respond with code as plain text without code block syntax around it\n"

    "Example 1:\n"

    "Original description:\n"
    "A Chevy Cruise autonomous vehicle, while in autonomous mode, was attempting to merge onto northbound Maple Ave from 5th Street when a bicyclist unexpectedly entered the vehicle's path, causing the vehicle to apply emergency braking. The bicyclist made minor contact with the front sensor array of the vehicle but managed to remain upright and uninjured. The vehicle sustained minimal damage to its front sensor array. No law enforcement was called to the scene, and the incident was recorded by the vehicle's onboard cameras for further analysis.\n"

    "Program:\n"
    "AUTONOMOUS_VEHICLE_MODEL = \"vehicle.chevrolet.impala\"\n"
    "BICYCLE_MODEL = [[\"vehicle.bh.crossbike\", \"vehicle.diamondback.century\", \"vehicle.gazelle.omafiets\"]]\n"
    
    "Final Answer:\n"
    "AUTONOMOUS_VEHICLE_MODEL = \"vehicle.chevrolet.impala\"\n"
    "BICYCLE_MODEL = Discrete({{\"vehicle.bh.crossbike\": 0.4, \"vehicle.diamondback.century\": 0.3, \"vehicle.gazelle.omafiets\": 0.3}})\n"

    "Now please provide your justification and final answer.\n"

    "JUSTIFICATION:\n"
    "[Q9B_JUSTIFICATION]\n" where STOPS_BEFORE(Q9B_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q9B_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q9B_FINAL_ANSWER]\n" where STOPS_BEFORE(Q9B_FINAL_ANSWER, "QUESTION THREE:") and len(TOKENS(Q9B_FINAL_ANSWER)) < 100

    
    "QUESTION THREE:\n"

    return {
        "Q9A_FINAL_ANSWER_TODO": Q9A_FINAL_ANSWER,
        "Q9A_JUSTIFICATION_TODO": Q9A_JUSTIFICATION,
    }
    '''


@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_4a(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION FOUR:\n"

    "Original description:\n"
    "{description}\n"

    "What details about the world and environment are missing from the description? (e.g. weather, time of day, etc.)\n"

    "JUSTIFICATION:\n"
    "[Q4A_JUSTIFICATION]\n" where STOPS_BEFORE(Q4A_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q4A_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q4A_FINAL_ANSWER]\n" where STOPS_BEFORE(Q4A_FINAL_ANSWER, "QUESTION FIVE:") and len(TOKENS(Q4A_FINAL_ANSWER)) < 100


    "QUESTION FIVE:\n"

    return {
        "Q4A_FINAL_ANSWER_TODO": Q4A_FINAL_ANSWER,
        "Q4A_JUSTIFICATION_TODO": Q4A_JUSTIFICATION,
    }
    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_4b(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION FOUR:\n"

    "Original description:\n"
    "{description}\n"

    "Relevant objects:\n"
    "{ANSWERS.get('Q1_FINAL_ANSWER')}\n"

    "For each of the relevant objects, what details about the objects are missing from the description that you would need to ask the author about in order to create a more accurate scene? What are the main environmental factors that need to be included in the scene? Your questions should cover dynamics of objects in motion (e.g. speed), distances between every pair of objects, and environmental conditions (e.g. weather). Provide your questions as a numbered list, but do not ask about personal details of any individuals involved.\n"

    "Example 1:\n"
    "Original description:\n"
    "An autonomous Ford Explorer SUV, operating in full autonomous mode, was navigating the ramp to merge onto Sand Hill Road amidst a heavy rainstorm. The vehicle's sensors detected the wet road conditions and adjusted speed accordingly. However, the driver claims' there was debris on the road and they adjusted accordingly. They encountered an unexpected large puddle which caused the vehicle to hydroplane, leading to a temporary loss of traction. The autonomous system attempted corrective steering maneuvers, but the vehicle ultimately made contact with the guardrail on the right side of the ramp. The collision resulted in moderate damage to the vehicle's right rear quarter panel and bumper. No injuries were reported, as the vehicle was unoccupied except for the presence of a safety operator, who did not sustain any injuries. The California Highway Patrol was notified and arrived on the scene to document the incident and assist in the vehicle's recovery. The incident highlighted the challenges faced by autonomous vehicles in severe weather conditions and the need for ongoing improvements in sensor and navigation technologies to handle such situations.\n"

    "Relevant objects:\n"
    "1. Autonomous Ford Explorer SUV\n"
    "2. Road debris\n"

    "Final Answer:\n"
    "Missing information:\n"
    "1. Can you provide more details about the speed at which the autonomous Ford Explorer SUV was traveling on the ramp during the heavy rainstorm?\n"
    "2. How large was the unexpected puddle that caused the vehicle to hydroplane?\n"
    "3. What type of debris was present on the road that the driver had to avoid?\n"
    "4. Were there any other vehicles in the vicinity on the ramp at the time of the incident?\n"
    "5. What specific weather conditions were present during the heavy rainstorm (e.g., visibility, wind speed)?\n"
    "6. How did the autonomous system respond to the hydroplaning situation before the collision occurred?\n"

    "Example 2:\n"

    "Original description:\n"
    "At approximately 12:05 PM, an autonomous BMW i8 convertible was eastbound on University Avenue when it collided with a city bus running a red light at Hamilton Avenue. At the same time, a scooter rider and a bicyclist, legally crossing Hamilton, narrowly avoided the incident, while two pedestrians were nearby on the sidewalk. Despite the BMW’s attempt to avoid the collision through emergency braking and evasive maneuvers, it sustained significant front-end damage, and the bus minor damage on its right side. Three bus passengers reported minor injuries. The complex scenario, involving multiple road users, highlighted the challenges autonomous vehicles face in dynamic urban environments. The primary cause was identified as the bus driver's failure to obey the traffic signal.\n"

    "Relevant objects:\n"
    "1. Autonomous BMW i8 convertible\n"
    "2. City bus\n"
    "3. Scooter rider\n"
    "4. Bicyclist\n"
    "5. Pedestrian one\n"
    "6. Pedestrian two\n"

    "Final Answer:\n"
    "1. What was the approximate speed of the autonomous BMW i8 convertible when it was traveling eastbound on University Avenue?\n"
    "2. Can you provide more details about the evasive maneuvers that the BMW attempted to avoid the collision?\n"
    "3. How close were the scooter rider and bicyclist to the point of impact between the BMW and the city bus?\n"
    "4. Were there any specific actions taken by the scooter rider and bicyclist to avoid the collision?\n"
    "5. How far away were the two pedestrians on the sidewalk from the intersection where the collision occurred?\n"
    "6. What were the weather conditions like at the time of the incident?\n"
    "7. Were there any specific road markings or signs at the intersection of University Avenue and Hamilton Avenue that may have influenced the events leading up to the collision?\n"
    "8. How did the autonomous system of the BMW respond to the situation when it detected the city bus running a red light?\n"
    "9. Were there any traffic congestion or other vehicles around the intersection that could have affected the incident?\n"


    "JUSTIFICATION:\n"
    "[Q4B_JUSTIFICATION]\n" where STOPS_BEFORE(Q4B_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q4B_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q4B_FINAL_ANSWER]\n" where STOPS_BEFORE(Q4B_FINAL_ANSWER, "QUESTION FIVE:") and len(TOKENS(Q4B_FINAL_ANSWER)) < 100


    "QUESTION FIVE:\n"

    return {
        "Q4B_FINAL_ANSWER_TODO": Q4B_FINAL_ANSWER,
        "Q4B_JUSTIFICATION_TODO": Q4B_JUSTIFICATION,
    }
    '''


@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_5(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION FIVE:\n"

    "Original description:\n"
    "{description}\n"

    "Relevant objects:\n"
    "{ANSWERS.get('Q1_FINAL_ANSWER')}\n"

    "Based on the missing object information from the user, provide a reasonable probability distribution over the missing values. Answer only the questions that are about distance between objects, speed, weather, or time. For example, if the time of day is missing but you know that the scene is in the morning, you could use a normal distribution with mean 8am and standard deviation 1 hour (Normal(8, 1)). If the color of the car is missing, you could use a uniform distribution over common car color string names. If the car speed is missing, you could use a normal distribution with mean around a reasonable speed limit for area of the scene and reasonable standard deviation, etc.\n"

    "First provide step-by-step reasoning as to why you choose such a distribution then provide your final answer as a numbered list. Be concise in your reasoning (no more than 1-2 sentences per object).\n"

    "Original description:\n"
    "An autonomous Ford Explorer SUV, operating in full autonomous mode, was navigating the ramp to merge onto Sand Hill Road amidst a heavy rainstorm. The vehicle's sensors detected the wet road conditions and adjusted speed accordingly. However, the driver claims' there was debris on the road and they adjusted accordingly. They encountered an unexpected large puddle which caused the vehicle to hydroplane, leading to a temporary loss of traction. The autonomous system attempted corrective steering maneuvers, but the vehicle ultimately made contact with the guardrail on the right side of the ramp. The collision resulted in moderate damage to the vehicle's right rear quarter panel and bumper. No injuries were reported, as the vehicle was unoccupied except for the presence of a safety operator, who did not sustain any injuries. The California Highway Patrol was notified and arrived on the scene to document the incident and assist in the vehicle's recovery. The incident highlighted the challenges faced by autonomous vehicles in severe weather conditions and the need for ongoing improvements in sensor and navigation technologies to handle such situations.\n"

    "Relevant objects:\n"
    "AUTONOMOUS_VEHICLE_MODEL = \"vehicle.ford.crown\"\n"
    "ROAD_DEBRIS_MODEL = [[\"static.prop.dirtdebris01\", \"static.prop.dirtdebris02\"]]\n"

    "Missing information:\n"
    "1. Can you provide more details about the speed at which the autonomous Ford Explorer SUV was traveling on the ramp during the heavy rainstorm?\n"
    "2. How large was the unexpected puddle that caused the vehicle to hydroplane?\n"
    "3. What type of debris was present on the road that the driver had to avoid?\n"
    "4. Were there any other vehicles in the vicinity on the ramp at the time of the incident?\n"
    "5. What specific weather conditions were present during the heavy rainstorm (e.g., visibility, wind speed)?\n"
    "6. How did the autonomous system respond to the hydroplaning situation before the collision occurred?\n"

    "REASONING:\n"
    "1. The speed at which the autonomous Ford Explorer SUV was traveling on the ramp during the heavy rainstorm can be modeled using a normal distribution with a mean around the speed limit for ramps (e.g., 35-45 mph) and a reasonable standard deviation to account for variations in driving behavior and road conditions.\n"
    "2. The size of the unexpected puddle that caused the vehicle to hydroplane can be modeled using a normal distribution with a mean based the fact that the puddle must have been at least as wide as the vehicle and on common puddle sizes on roads (e.g., 6-8 feet in diameter) and a standard deviation to capture variations in puddle sizes.\n"
    "3. The type of debris present on the road that the driver had to avoid can be modeled using a uniform distribution over the provided ROAD_DEBRIS_MODEL options [[\"static.prop.dirtdebris01\", \"static.prop.dirtdebris02\"]].\n"
    "4. The presence of other vehicles in the vicinity on the ramp at the time of the incident can be modeled using a Bernoulli distribution with a parameter reflecting the likelihood of other vehicles being present (e.g., low probability due to heavy rainstorm and specific location).\n"
    "5. The specific weather conditions present during the heavy rainstorm (e.g., visibility, wind speed) can be modeled using a combination of distributions such as normal distribution for visibility range and wind speed based on historical weather data for the area.\n"
    "6. The response of the autonomous system to the hydroplaning situation before the collision occurred can be modeled as a categorical distribution with options such as \"applied corrective steering maneuvers,\" \"adjusted speed,\" \"issued warnings to safety operator,\" etc.\n"

    "FINAL_ANSWER:\n"
    "1. Speed at which the autonomous Ford Explorer SUV was traveling: Normal distribution with mean around 40 mph and standard deviation 5 mph.\n"
    "2. Size of the unexpected puddle: Normal distribution with mean around 7 feet in diameter and standard deviation 0.5 feet.\n"
    "3. Type of debris present on the road: Uniform distribution over [[\"static.prop.dirtdebris01\", \"static.prop.dirtdebris02\"]].\n"
    "4. Presence of other vehicles in the vicinity: Bernoulli distribution with low probability.\n"
    "5. Specific weather conditions: Modeled using appropriate distributions based on historical data.\n"
    "6. Autonomous system response to hydroplaning situation: Categorical distribution with relevant options.\n"


    "JUSTIFICATION:\n"
    "[Q5_JUSTIFICATION]\n" where STOPS_BEFORE(Q5_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q5_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q5_FINAL_ANSWER]\n" where STOPS_BEFORE(Q5_FINAL_ANSWER, "QUESTION SIX:") and len(TOKENS(Q5_FINAL_ANSWER)) < 100


    "QUESTION SIX:\n"

    return {
        "Q5_FINAL_ANSWER_TODO": Q5_FINAL_ANSWER,
        "Q5_JUSTIFICATION_TODO": Q5_JUSTIFICATION,
    }
    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_6(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION SIX:\n"

    "Original description:\n"
    "{description}\n"

    "Relevant objects:\n"
    "{ANSWERS.get('Q1_FINAL_ANSWER')}\n"

    "You are a specialized agent for writing Scenic, a probabilistic programming language.\n"

    "A user will provide you with probability distributions for missing information in a vehicle crash description. Your task is to interpret the probability distributions and express them as a Scenic program.\n"

    "Scenic can only support the following distributions so you must pick the closest matching distribution. Under no circumstance should you use any of the other distributions\n"

    "Range(low, high) - Uniform distribution over the real range [[low, high]]\n"
    "DiscreteRange(low, high) - Uniform distribution over the discreet integer range [[low, high]]\n"
    "Normal(mean, std) - Normal distribution with mean and standard deviation\n"
    "TruncatedNormal(mean, stdDev, low, high) - Normal distribution with mean and standard deviation truncated to the range [[low, high]]\n"
    "Uniform(value, …) - Uniform distribution over the list of values provided.\n"
    "Discrete([[value: weight, … ]]) - Discrete distribution over the list of values provided with the given weights (e.g., [[value: 0.5, value: 0.2, value: 0.3]])\n"

    "For weather, Scenic can only support a Uniform or Discrete distribution over the following values: [['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'SoftRainSunset', 'MidRainSunset', 'HardRainSunset', 'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 'SoftRainNight', 'MidRainyNight', 'HardRainNight' ,'DustStorm']]\n"

    "Based on the distributions and original description, define Scenic distributions over the uncertain values. Provide values for the parameters to your distributions. You may not use any of the other distributions. If you cannot find a distribution that matches the missing information, you must choose the closest matching distribution.\n"

    "JUSTIFICATION:\n"
    "[Q6_JUSTIFICATION]\n" where STOPS_BEFORE(Q6_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q6_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q6_FINAL_ANSWER]\n" where STOPS_BEFORE(Q6_FINAL_ANSWER, "QUESTION SEVEN:") and len(TOKENS(Q6_FINAL_ANSWER)) < 100


    "QUESTION SEVEN:\n"

    return {
        "Q6_FINAL_ANSWER_TODO": Q6_FINAL_ANSWER,
        "Q6_JUSTIFICATION_TODO": Q6_JUSTIFICATION,
    }
    '''


@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_7(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION SEVEN:\n"

    "Original description:\n"
    "{description}\n"

    "You are a specialized agent for writing Scenic, a probabilistic programming language.\n"

    "Based on the original description, pick from the following the best matching town. You may not choose any other town. If you cannot find a town that matches the original description, you must choose the closest matching town. Then after selecting a town, provide a high-level description (ignoring road names) of where in the town we should replicate the original description. For example, if the original description specified a highway such as US-101, provide a description about the properties of that highway, such as it is a 4 lane road.\n"
    "Town07 - imitates a quiet rural community, a green landscape filled with cornfields, barns, grain silos and windmills.\n"
    "Town06 - is a low density town set into a coniferous landscape exhibiting a multitude of large, 4-6 lane roads and special junctions like the Michigan Left.\n"
    "Town05 - is an urban environment set into a backdrop of conifer-covered hills with a raised highway and large multilane roads and junctions.\n"
    "Town04 - is a small town with a backdrop of snow-capped mountains and conifers. A multi-lane road circumnavigates the town in a figure of 8.\n"
    "Town03 - is a larger town with features of a downtown urban area. The map includes some interesting road network features such as a roundabout, underpasses and overpasses. The town also includes a raised metro track and a large building under construction.\n"
    "Town02 - is a small town with numerous T-junctions and a variety of buildings, there are patches of coniferous trees, a park and a residential and commercial area.\n"
    "Town01 - is a small town with numerous T-junctions and a variety of buildings, surrounded by coniferous trees and featuring several small bridges spanning across a river that divides the town into 2 halves.\n"

    "Each expert and the final answer should be provided in the following format:\n"
    "TOWN:\n"
    "<Town0x>\n"

    "LOCATION_IN_TOWN:\n"
    "<description_of_location_in_town>\n"

    "JUSTIFICATION:\n"
    "[Q7_JUSTIFICATION]\n" where STOPS_BEFORE(Q7_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q7_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q7_FINAL_ANSWER]\n" where STOPS_BEFORE(Q7_FINAL_ANSWER, "QUESTION EIGHT:") and len(TOKENS(Q7_FINAL_ANSWER)) < 100


    "QUESTION EIGHT:\n"

    return {
        "Q7_FINAL_ANSWER_TODO": Q7_FINAL_ANSWER,
        "Q7_JUSTIFICATION_TODO": Q7_JUSTIFICATION,
    }
    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_3ab(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION THREE:\n"

    "Original description:\n"
    "{description}\n"

    "What are the main events that happened in the scene? (e.g. car stopped when pedestrian crossed the street, a car was driving in a lane then switched lanes then made a left turn, etc.). Describe these events in natural language.\n"

    "JUSTIFICATION:\n"
    "[Q3A_JUSTIFICATION]\n" where STOPS_BEFORE(Q3A_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q3A_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q3A_FINAL_ANSWER]\n" where STOPS_BEFORE(Q3A_FINAL_ANSWER, "QUESTION FOUR:") and len(TOKENS(Q3A_FINAL_ANSWER)) < 100


    "QUESTION FOUR:\n"

    "Scenic only allows certain properties to be described in Linear Temporal Logic (LTL) formula (the end of the events or time invariant properties). So for the events that we can, describe the end of the events in LTL formula for them. Here are some examples of valid LTL formulas that are supported in Scenic:\n"
    "car2 not in intersection until car1 in intersection\n"
    "eventually car2 in intersection\n"
    "eventually ego in intersection\n"
    "(always car.speed < 30) implies (always distance to car > 10)\n"
    "always not ((ego can see car1) and (ego can see car2))\n"

    "Please output 1-2 most relevant properties to the program description provided above.\n"

    "JUSTIFICATION:\n"
    "[Q3B_JUSTIFICATION]\n" where STOPS_BEFORE(Q3B_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q3B_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q3B_FINAL_ANSWER]\n" where STOPS_BEFORE(Q3B_FINAL_ANSWER, "QUESTION FIVE:") and len(TOKENS(Q3B_FINAL_ANSWER)) < 100
    

    "QUESTION FIVE:\n"

    return {
        "Q3A_FINAL_ANSWER_TODO": Q3A_FINAL_ANSWER,
        "Q3A_JUSTIFICATION_TODO": Q3A_JUSTIFICATION,
        "Q3B_FINAL_ANSWER_TODO": Q3B_FINAL_ANSWER,
        "Q3B_JUSTIFICATION_TODO": Q3B_JUSTIFICATION,
    }
    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_2(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION TWO:\n"

    "Original description:\n"
    "{description}\n"

    "Relevant objects:\n"
    "{ANSWERS.get('Q1_FINAL_ANSWER')}\n"

    "Based on the relevant objects selected from the original description, what are the spatial relationships between the objects? (e.g. car is in front of pedestrian, etc.) Are the objects moving or stationary? Are they visible or occluded? You can only use the following terms to describe spatial relationships: in front of, behind, left of, right of, facing, ahead of, behind, visible, and not visible.\n"

    "Each expert and the final answer should be provided in the following format:\n"
    "SPATIAL_RELATIONSHIPS:\n"
    "<spatial_relationships>\n"

    "MOVEMENT:\n"
    "<movement>\n"

    "VISIBILITY:\n"
    "<visibility>\n"

    "JUSTIFICATION:\n"
    "[Q2_JUSTIFICATION]\n" where STOPS_BEFORE(Q2_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q2_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q2_FINAL_ANSWER]\n" where STOPS_BEFORE(Q2_FINAL_ANSWER, "QUESTION THREE:") and len(TOKENS(Q2_FINAL_ANSWER)) < 100


    "QUESTION THREE:\n"

    return {
        "Q2_FINAL_ANSWER_TODO": Q2_FINAL_ANSWER,
        "Q2_JUSTIFICATION_TODO": Q2_JUSTIFICATION,
    }
    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning_8(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
    '''lmql
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
    "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
    "EXPERT_1:\n"
    "<expert_1_answer>\n"
    "EXPERT_2:\n"
    "<expert_2_answer>\n"
    "EXPERT_3:\n"
    "<expert_3_answer>\n"
    "FINAL_ANSWER:\n"
    "<final_answer>\n"

    "Here is one example of a Scenic program:\n"
    "{example}\n"


    "QUESTION EIGHT:\n"

    "Original description:\n"
    "{description}\n"

    "Relevant objects:\n"
    "{ANSWERS.get('Q1_FINAL_ANSWER')}\n"

    "Important events:\n"
    "{ANSWERS.get('Q3A_FINAL_ANSWER')}\n"

    "Here is a list of the supported behaviors in Scenic. Based on the relevant objects and important events, which behaviors do we need to use to recreate the original description? You may select more than one behavior as they are composable. If you cannot find a behavior that matches the original description, you must choose the closest matching behavior.\n"

    "Here are the only behaviors that are allowed for vehicles, buses, motorcycles, and bicycles:\n"
    "behavior ConstantThrottleBehavior(x : float):\n"
    "behavior DriveAvoidingCollisions(target_speed : float = 25, avoidance_threshold : float = 10):\n"
    "   # Drive at a target speed, avoiding collisions with other vehicles\n"
    "   # Throttle is off and braking is applied if the distance to the nearest vehicle is less\n"
    "   # than the avoidance threshold\n"
    "behavior AccelerateForwardBehavior(): # Accelerate forward with throttle set to 0.5\n"
    "behavior FollowLaneBehavior(target_speed : float = 10, laneToFollow : Lane = None, is_oppositeTraffic : bool = False):\n"
    "   # Follow's the lane on which the vehicle is at, unless the laneToFollow is specified.\n"
    "   # Once the vehicle reaches an intersection, by default, the vehicle will take the straight route.\n"
    "   # If straight route is not available, then any available turn route will be taken, uniformly randomly. \n"
    "   # If turning at the intersection, the vehicle will slow down to make the turn, safely. \n"
    "   # This behavior does not terminate. A recommended use of the behavior is to accompany it with condition,\n"
    "   # e.g. do FollowLaneBehavior() until ...\n"
    "   # :param target_speed: Its unit is in m/s. By default, it is set to 10 m/s\n"
    "   # :param laneToFollow: If the lane to follow is different from the lane that the vehicle is on, this parameter can be used to specify that lane. By default, this variable will be set to None, which means that the vehicle will follow the lane that it is currently on.\n"
    "behavior FollowTrajectoryBehavior(target_speed : float = 10, trajectory : List[Lane] = None, turn_speed : float = None):\n"
    "   # Follows the given trajectory. The behavior terminates once the end of the trajectory is reached.\n"
    "   # :param target_speed: Its unit is in m/s. By default, it is set to 10 m/s\n"
    "   # :param trajectory: It is a list of sequential lanes to track, from the lane that the vehicle is initially on to the lane it should end up on.\n"
    "behavior TurnBehavior(trajectory : List[Lane] = None, target_speed : float = 6):\n"
    "   # This behavior uses a controller specifically tuned for turning at an intersection.\n"
    "   # This behavior is only operational within an intersection, it will terminate if the vehicle is outside of an intersection.\n"
    "behavior LaneChangeBehavior(laneSectionToSwitchTo : Lane, is_oppositeTraffic : bool = False, target_speed : float = 10):\n"
    "   # is_oppositeTraffic should be specified as True only if the laneSectionToSwitch to has\n"
    "   # the opposite traffic direction to the initial lane from which the vehicle started LaneChangeBehavior\n"

    "Here are the only behaviors that are allowed for pedestrians:\n"
    "behavior WalkForwardBehavior(speed=0.5):\n"
    "   take SetWalkingDirectionAction(self.heading), SetWalkingSpeedAction(speed)\n"
    "   # Walk forward behavior for pedestrians by uniformly sampling either side of the sidewalk for the pedestrian to walk on\n"
    "behavior WalkBehavior(maxSpeed=1.4):\n"
    "   take SetWalkAction(True, maxSpeed)\n"
    "behavior CrossingBehavior(reference_actor, min_speed=1, threshold=10, final_speed=None):\n"
    "   # This behavior dynamically controls the speed of an actor that will perpendicularly (or close to)\n"
    "   # cross the road, so that it arrives at a spot in the road at the same time as a reference actor.\n"
    "   # Args:\n"
    "   # min_speed (float): minimum speed of the crossing actor. As this is a type of 'synchronization action',\n"
    "   # a minimum speed is needed, to allow the actor to keep moving even if the reference actor has stopped\n"
    "   # threshold (float): starting distance at which the crossing actor starts moving\n"
    "   # final_speed (float): speed of the crossing actor after the reference one surpasses it\n"

    "Each expert and the final answer should be provided in the following format:\n"
    "BEHAVIOR:\n"
    "<behavior>\n"

    "JUSTIFICATION:\n"
    "[Q8_JUSTIFICATION]\n" where STOPS_BEFORE(Q8_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q8_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q8_FINAL_ANSWER]\n" where STOPS_BEFORE(Q8_FINAL_ANSWER, "QUESTION NINE:") and len(TOKENS(Q8_FINAL_ANSWER)) < 100

    
    "QUESTION NINE:\n"

    return {
        "Q8_FINAL_ANSWER_TODO": Q8_FINAL_ANSWER,
        "Q8_JUSTIFICATION_TODO": Q8_JUSTIFICATION,
    }
    '''


@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(1)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def regenerate_scenic_code(model_input, example_prompt, working_scenic, new_scenic, lmql_outputs):
    '''lmql

    "You are an autonomous vehicle simulation programming expert. Earlier on, you made your first attempt of the following task.\n"
    
    "{example_prompt}\n"

    "TODO: Your task is to modify one section of your previous output so the generated code for the given scenario executes.\n"

    "Previous user input: consider the following natural language description of a traffic incident:\n"
    "{model_input.nat_lang_scene_des}\n"

    "## WORKING CODE: The code below represents the working section program and contains no errors.\n"
    "{working_scenic}\n"

    "## ERROR: The first compiler error raised with the continuation of the Scenic program below:\n"
    "{model_input.compiler_error}\n"

    "TASK: Modify the code below using your expertise about the Scenic programming language so the error no longer appears\n"
    "{new_scenic}\n"

    if "OTHER_CONSTANTS_TODO" in lmql_outputs:
        "[OTHER_CONSTANTS]\n"  where STOPS_BEFORE(OTHER_CONSTANTS, "##") and len(TOKENS(OTHER_CONSTANTS)) < 100
    else:
        OTHER_CONSTANTS = None
    
    if "VEHICLE_BEHAVIORS_TODO" in lmql_outputs:
        "## DEFINING BEHAVIORS\n"
        "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "##") and len(TOKENS(SPATIAL_RELATIONS)) < 400
    else:
        BEHAVIORS = None
    

    if "SPATIAL_RELATIONS_TODO" in lmql_outputs:
        "## DEFINING SPATIAL RELATIONS\n"
        "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 400
    else:
        SPATIAL_RELATIONS = None
    
    return {
        "OTHER_CONSTANTS_TODO" : OTHER_CONSTANTS,
        "VEHICLE_BEHAVIORS_TODO" : BEHAVIORS,
        "SPATIAL_RELATIONS_TODO" : SPATIAL_RELATIONS,
    }

    '''

def strip_other_constants(other_constants):
    """
    removes weird number characters at the beginning of other constants generation
    """
    nonalpha = string.digits + string.punctuation + string.whitespace
    return other_constants.lstrip(nonalpha)
    
def construct_scenic_program_tot(model_input, example_prompt, nat_lang_scene_des, segmented_retry=True, max_retries=5):
    """
    constructs a scenic program using the template in lmql_template_limited.scenic 
    incorporates tot reasoning from LMQL into final outputs
    """

    #Load known variable sets from blueprints
    towns = list(np.load('src/scenicNL/constraints/blueprints/towns.npy'))
    vehicles = list(np.load('src/scenicNL/constraints/blueprints/vehicles.npy')) 
    weather = list(np.load('src/scenicNL/constraints/blueprints/weather.npy')) 
    objects = list(np.load('src/scenicNL/constraints/blueprints/objects.npy')) 
    description = nat_lang_scene_des
    example = model_input.examples[0]
    model_input.set_nl(nat_lang_scene_des)

    #Load output template
    scenic_template_path = f"src/scenicNL/constraints/lmql_template_limited.scenic"
    scenic_template = open(scenic_template_path, 'r').read()

    #Start reasoning
    def update(temp, full):
        for k, v in temp.items():
            if k not in full:
                full[k] = v
    reasoning_funcs = [generate_reasoning_1, generate_reasoning_9a, generate_reasoning_9b, generate_reasoning_4a, generate_reasoning_4b, generate_reasoning_5]
    reasoning_funcs += []
    lmql_tot_full = {}
    for reasoning_count, reasoning_func in enumerate(reasoning_funcs):
        lmql_tot_temp = reasoning_func(description, example, towns, vehicles, objects, weather, lmql_tot_full)
        update(temp = lmql_tot_temp, full = lmql_tot_full)
        print(f'Completed generate reasoning pt {reasoning_count+1}/{len(reasoning_funcs)}!')

    print('$%$%$%')
    # print(lmql_tot_full)
    for key in sorted(lmql_tot_full.keys()):
        if 'FINAL_ANSWER' in key:
            print(key, '-', lmql_tot_full[key])
    print('$%$%$%')
    #End reasoning
    
    # assert False

    print('0. Displaying example_prompt for the code that follows')
    print(example_prompt)
    print()

    lmql_outputs = generate_scenic_code(example_prompt, towns, vehicles, weather)
    assert 'EGO_VEHICLE_BLUEPRINT_ID_TODO' in lmql_outputs

    lmql_outputs["OTHER_CONSTANTS_TODO"] = strip_other_constants(lmql_outputs["OTHER_CONSTANTS_TODO"])
    lmql_outputs["TEXT_DESCRIPTION_TODO"] = nat_lang_scene_des

    section_keys =  ["OTHER_CONSTANTS_TODO", "VEHICLE_BEHAVIORS_TODO" , "SPATIAL_RELATIONS_TODO"]
    
    #complete the template using the lmql_outputs
    if not segmented_retry:
        final_scenic = scenic_template.format_map(lmql_outputs)
    else:
        print('1. Segmenting retries into template sections:')
        template_sections = scenic_template.split("##")
        template_sections = ["##" + section for section in template_sections]
        for ind, section in enumerate(template_sections):
            print(f'{ind}-'*8)
            print(section)
        print()


        print('2. Printing first template section infilled with lmql_outputs:')
        final_scenic = template_sections[0].format_map(lmql_outputs) + '\n' + template_sections[1].format_map(lmql_outputs) #this should compile everytime
        new_scenic = ''
        i = 2
        # final_scenic = template_sections[0].format_map(lmql_outputs) #+ template_sections[1].format_map(lmql_outputs) #this should compile everytime
        print(final_scenic)

        print('Starting loop below:')
        i = 2
        num_retries = max_retries
        while i < len(template_sections) and num_retries > 0:
            print(f'\n\n\n\n{i} {num_retries} {i} {num_retries} {i} {num_retries}\n\n\n\n')

            print('3. Lmql outputs')
            for key in sorted(lmql_outputs.keys()):
                print(key)
            print('3b. Lmql outputs')
            for key in sorted(lmql_outputs.keys()):
                print(key, '-', lmql_outputs[key])


            print('3c. Template to fill in')
            print(template_sections[i])
            print('3d. Template filling in attempt')
            print('start')
            print(template_sections[i].format_map(lmql_outputs))
            print('end')

            new_scenic = template_sections[i].format_map(lmql_outputs).strip()
            uncompiled_scenic = final_scenic + '\n' + new_scenic
            working_scenic = final_scenic
            print('4. Working scenic')
            print(working_scenic)
            print()
            print('5. Uncompiled scenic')
            print(uncompiled_scenic)

            compiles, error_message = check_compile(uncompiled_scenic)
            # reassign values in model_input
            model_input.set_fasp(uncompiled_scenic)
            model_input.set_err(error_message)
            
            if not compiles:
                print('6. Error message on uncompiled scenic (if any)')
                print(error_message)
            else:
                print('6. No error on uncompiled scenic')

            # check if compiles
            if not compiles:
                #regenerate this section and next
                print(f"{i} {num_retries} ERROR {error_message}")

                print('7. Inputs to regenerate scenic')

                print("A. The following natural language description:")
                print(f"{model_input.nat_lang_scene_des}")
                print("B. The following scenic_program with compiler errors that models the description:")
                print(f"{model_input.first_attempt_scenic_program}")
                print("C. The first compiler error raised with the scenic program:")
                print(f"{model_input.compiler_error}")

                print("D. Please output a modified version of scenic_program modified so the compiler error does not appear.")
                f"{working_scenic}"

                # lmql_outputs_tmp = regenerate_scenic(model_input, working_scenic, lmql_outputs)
                lmql_outputs_tmp = regenerate_scenic_code(model_input, example_prompt, working_scenic, new_scenic, lmql_outputs)
                lmql_outputs_tmp = {k: v for k, v in lmql_outputs_tmp.items() if v is not None} ### this is bad
                for key in lmql_outputs_tmp:
                    lmql_outputs[key] = lmql_outputs_tmp[key]
                num_retries -= 1
            else:
                final_scenic = uncompiled_scenic
                working_key = section_keys.pop(0)
                lmql_outputs.pop(working_key, None)
                i += 1
                num_retries = max_retries
        if num_retries == 0:
            print("RAN OUT OF RETRIES RIP")
            new_scenic = '\n'.join([template_section.format_map(lmql_outputs).strip() for template_section in template_sections[i:]])
            print('8. Displaying final scenic before appending')
            print(final_scenic)
            print()

            final_scenic += new_scenic

            print('9. Displaying final scenic after appending')
            print(final_scenic)
            print()

        print('CHECKING FINAL SCENIC')
        compiles, message = check_compile(final_scenic)
        print(f'compiles: {compiles}, message: {message}')

    return final_scenic

def construct_scenic_program(model_input, example_prompt, nat_lang_scene_des, segmented_retry=True, max_retries=5):
    """
    constructs a scenic program using the template in lmql_template_limited.scenic 
    """

    #Load known variable sets from blueprints
    towns = list(np.load('src/scenicNL/constraints/blueprints/towns.npy'))
    vehicles = list(np.load('src/scenicNL/constraints/blueprints/vehicles.npy')) 
    weather = list(np.load('src/scenicNL/constraints/blueprints/weather.npy')) 

    # #Load output template
    scenic_template_path = f"src/scenicNL/constraints/lmql_template_limited.scenic"
    scenic_template = open(scenic_template_path, 'r').read()

    #query lmql to get fill in blanks
    lmql_outputs = generate_scenic_code(example_prompt, towns, vehicles, weather)

    lmql_outputs["OTHER_CONSTANTS_TODO"] = strip_other_constants(lmql_outputs["OTHER_CONSTANTS_TODO"])
    lmql_outputs["TEXT_DESCRIPTION_TODO"] = nat_lang_scene_des

    section_keys =  ["OTHER_CONSTANTS_TODO", "VEHICLE_BEHAVIORS_TODO" , "SPATIAL_RELATIONS_TODO"]
    
    #complete the template using the lmql_outputs
    if not segmented_retry:
        final_scenic = scenic_template.format_map(lmql_outputs)
    else:
        print('Segmenting retries')
        template_sections = scenic_template.split("##")
        template_sections = ["##" + section for section in template_sections]
        print('TEMNPLAE_SECTIONS', template_sections)
        final_scenic = template_sections[0].format_map(lmql_outputs) + '\n' #+ template_sections[1].format_map(lmql_outputs) #this should compile everytime
        
        i = 1
        num_retries = max_retries
        while i < len(template_sections) and num_retries > 0:
            print(f'\n\n\n\n\n\n\n{i} {num_retries} {i} {num_retries} {i} {num_retries}')
            uncompiled_scenic = final_scenic + '\n' + template_sections[i].format_map(lmql_outputs)
            working_scenic = final_scenic

            compiles, error_message = check_compile(uncompiled_scenic)
            # reassign values in model_input
            model_input.set_fasp(uncompiled_scenic)
            model_input.set_err(error_message)
            print('****\n\n\n\n')
            print(uncompiled_scenic)
            print('%%%%\n\n\n\n')
            print(working_scenic)
            print('$$$$\n\n\n\n')
            print(error_message)

            # check if compiles
            if not compiles:
                print(f'{i} {num_retries} DID NOT COMPILE: \n\n{uncompiled_scenic}')
                #regenerate this section and next
                print(f"{i} {num_retries} ERROR {error_message}")
                lmql_outputs = regenerate_scenic(model_input, working_scenic, lmql_outputs)
                lmql_outputs = {k: v for k, v in lmql_outputs.items() if v is not None}
                num_retries -= 1
            else:
                final_scenic = uncompiled_scenic
                working_key = section_keys.pop(0)
                lmql_outputs.pop(working_key, None)
                i += 1
                num_retries = 3
        if num_retries == 0:
            print("RAN OUT OF RETRIES RIP")
        print('CHECKING FINAL SCENIC')
        compiles, message = check_compile(final_scenic)
        print(f'compiles: {compiles}, message: {message}')

    return final_scenic

def check_compile(scenic_program):
    retries_dir = os.path.join(os.curdir, 'temp_dir')
    os.makedirs(retries_dir, exist_ok=True)
    works, error_message = True, ""
    with tempfile.NamedTemporaryFile(dir=retries_dir, delete=False, suffix='.scenic') as temp_file:
        fname = temp_file.name
        with open(fname, 'w') as f:
            f.write(scenic_program)
        try:
            scenario = scenic.scenarioFromFile(fname, mode2D=True)
            print('No execution error! (1/2)')
        except Exception as e:
            try:
                error_message = f"Error details below..\nerror message: {str(e)}\nerror text: {e.text}\nerror lineno: {e.lineno}\nend_lineno: {e.end_lineno}\nerror offset: {e.offset}\nerror end_offset: {e.end_offset}"
                print(error_message)
            except:
                error_message = f'Error details below..\nerror message: {str(e)}'
                print(error_message)
            works = False
        try:
            if works: 
                ast = scenic.syntax.parser.parse_file(fname)
                print('No execution or compilation error! (2/2)')
        except Exception as e:
            try:
                error_message = f"Error details below..\nerror message: {str(e)}\nerror text: {e.text}\nerror lineno: {e.lineno}\nend_lineno: {e.end_lineno}\nerror offset: {e.offset}\nerror end_offset: {e.end_offset}"
                print(error_message)
            except:
                error_message = f'Error details below..\nerror message: {str(e)}'
                print(error_message)
            works = False
    return works, error_message