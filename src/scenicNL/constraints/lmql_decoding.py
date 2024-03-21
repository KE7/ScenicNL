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
def generate_reasoning(description, example, towns, vehicles, objects, weather, ANSWERS={}): # ANSWERS not used
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

    "Original description:\n"
    "{description}\n"

    
    "QUESTION ONE:\n"

    "Based on the description, what are the all of the vehicles that need to be included in the scene?\n"
    "First provide step-by-step justification for all vehicles you chose then provide your final answer as:\n"

    "JUSTIFICATION:\n"
    "[Q1_JUSTIFICATION]\n" where STOPS_BEFORE(Q1_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q1_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q1_FINAL_ANSWER]\n" where STOPS_BEFORE(Q1_FINAL_ANSWER, "QUESTION TWO:") and len(TOKENS(Q1_FINAL_ANSWER)) < 100

    
    "QUESTION TWO:\n"

    "Given the relevant vehicles and objects that you identified above, find the closest matching Scenic vehicle from the list. You cannot choose vehicles that are not in the lists so if you cannot find the vehicle in the list, you must choose the closest matching vehicle.\n"

    "Previously Answered Question One:\n"
    "{Q1_FINAL_ANSWER}\n"

    "VEHICLES:\n"
    "{vehicles}\n"

    "OBJECTS:\n"
    "{objects}\n"

    "Based on the description, what are the all of the objects that need to be included in the scene? Let ego denote the self-driving car.\n"
    "Each expert must first provide step-by-step justification for all objects they chose, then provide the final answer. For example:\n"

    "JUSTIFICATION:\n"
    "<justification_for_the_objects>\n"

    "FINAL_ANSWER:\n"
    "ego = 'vehicle.audi.a2'\n"
    "bicycle = 'vehicle.diamondback.century'\n"
    "pedestrian = 'walker.pedestrian.0003'\n"

    "Now please provide your justification and final answer.\n"

    "JUSTIFICATION:\n"
    "[Q2_JUSTIFICATION]\n" where STOPS_BEFORE(Q2_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q2_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q2_FINAL_ANSWER]\n" where STOPS_BEFORE(Q2_FINAL_ANSWER, "QUESTION THREE:") and len(TOKENS(Q2_FINAL_ANSWER)) < 100
    

    "QUESTION THREE:\n"

    "Previously Provided Original Description:\n"
    "{description}\n"
    
    "What details about the world and environment are missing from the description? (e.g. what is the weather, time of day, etc.)\n"

    "JUSTIFICATION:\n"
    "[Q3_JUSTIFICATION]\n" where STOPS_BEFORE(Q3_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q3_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q3_FINAL_ANSWER]\n" where STOPS_BEFORE(Q3_FINAL_ANSWER, "QUESTION FOUR:") and len(TOKENS(Q3_FINAL_ANSWER)) < 100


    "QUESTION FOUR:\n"

    return {
        "Q1_FINAL_ANSWER_TODO": Q1_FINAL_ANSWER,
        "Q2_FINAL_ANSWER_TODO": Q2_FINAL_ANSWER,
        "Q3_FINAL_ANSWER_TODO": Q3_FINAL_ANSWER,
        "Q1_JUSTIFICATION_TODO": Q1_JUSTIFICATION,
        "Q2_JUSTIFICATION_TODO": Q2_JUSTIFICATION,
        "Q3_JUSTIFICATION_TODO": Q3_JUSTIFICATION,
    }

    '''

@retry(
    wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
)
@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning2(description, example, towns, vehicles, objects, weather, ANSWERS):
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

    "Original description:\n"
    "{description}\n"

    
    "QUESTION FOUR:\n"
    
    "Previously Provided Original Description:\n"
    "{description}\n"

    "Previously Answered Question One:\n"
    "{ANSWERS.get('Q1_FINAL_ANSWER')}\n"
    
    "Given the relevant objects, what details are missing from the description that you would need to ask the author about in order to create a more accurate scene? (e.g. what color is the car, how many pedestrians are there, how fast is the car moving, how far away is the car from the pedestrian, etc.)\n"

    "JUSTIFICATION:\n"
    "[Q4_JUSTIFICATION]\n" where STOPS_BEFORE(Q4_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q4_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q4_FINAL_ANSWER]\n" where STOPS_BEFORE(Q4_FINAL_ANSWER, "QUESTION FIVE:") and len(TOKENS(Q4_FINAL_ANSWER)) < 100
    

    "QUESTION FIVE:\n"

    "Previously Provided Original Description:\n"
    "{description}\n"

    "Previously Answered Question Four:\n"
    "{Q4_FINAL_ANSWER}\n"

    "Based on the missing information above, provide a reasonable probability distribution over the missing values. For example, if the time of day is missing but you know that the scene is in the morning, you could use a normal distribution with mean 8am and standard deviation 1 hour. If the color of the car is missing, you could use a uniform distribution over common car colors. If the car speed is missing, you could use a normal distribution with mean around a reasonable speed limit for area of the scene and standard deviation of 5 mph, etc.\n"

    "JUSTIFICATION:\n"
    "[Q5_JUSTIFICATION]\n" where STOPS_BEFORE(Q5_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q5_JUSTIFICATION)) < 500

    "FINAL ANSWER:\n"
    "[Q5_FINAL_ANSWER]\n" where STOPS_BEFORE(Q5_FINAL_ANSWER, "QUESTION SIX:") and len(TOKENS(Q5_FINAL_ANSWER)) < 100


    "QUESTION SIX:\n"


    return {
        "Q4_FINAL_ANSWER_TODO": Q4_FINAL_ANSWER,
        "Q5_FINAL_ANSWER_TODO": Q5_FINAL_ANSWER,
        "Q4_JUSTIFICATION_TODO": Q4_JUSTIFICATION,
        "Q5_JUSTIFICATION_TODO": Q5_JUSTIFICATION,
    }

    '''

# @retry(
#     wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
# )
# @lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
# def generate_reasoning3(description, example, towns, vehicles, objects, weather, ANSWERS):
#     '''lmql
#     "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"
    
#     "We are going to continue playing a game. For the following questions, imagine that you are 3 different autonomous driving experts. For every question, each expert must provide a step-by-step explanation for how they came up with their answer. After all the experts have answered the question, you will need to provide a final answer using the best parts of each expert's explanation. Use the following format:\n"
#     "EXPERT_1:\n"
#     "<expert_1_answer>\n"
#     "EXPERT_2:\n"
#     "<expert_2_answer>\n"
#     "EXPERT_3:\n"
#     "<expert_3_answer>\n"
#     "FINAL_ANSWER:\n"
#     "<final_answer>\n"

#     "Here is one example of a Scenic program:\n"
#     "{example}\n"

#     "Original description:\n"
#     "{description}\n"

#     "QUESTION FOUR:\n"
    
#     "Previously Provided Original Description:\n"
#     "{description}\n"

#     "Previously Answered Question One:\n"
#     "{ANSWERS.get('Q1_FINAL_ANSWER')}\n"
    
    # "Previously Answered Missing Info:\n"
    # "{ANSWERS[Q4_FINAL_ANSWER]}\n"

    # "Distributions for Missing Information:\n"
    # "{Q5_FINAL_ANSWER}\n"

    # "Based on the missing information and distributions above, pick from the following list of distributions that are supported. You may not use any of the other distributions. If you cannot find a distribution that matches the missing information, you must choose the closest matching distribution:\n"
    # "Range(low, high) - Uniform distribution over the range [[low, high]]\n"
    # "DiscreteRange(low, high) - Uniform distribution over the discreet integer range [[low, high]]\n"
    # "Normal(mean, std) - Normal distribution with mean and standard deviation\n"
    # "TruncatedNormal(mean, stdDev, low, high) - Normal distribution with mean and standard deviation truncated to the range [[low, high]]\n"
    # "Uniform(value, …) - Uniform distribution over the values provided\n"
    # "Discrete([[value: weight, … ]]) - Discrete distribution over the values provided with the given weights\n"

    # "JUSTIFICATION:\n"
    # "[Q6_JUSTIFICATION]\n" where STOPS_BEFORE(Q6_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q6_JUSTIFICATION)) < 500

    # "FINAL ANSWER:\n"
    # "[Q6_FINAL_ANSWER]\n" where STOPS_BEFORE(Q6_FINAL_ANSWER, "QUESTION SEVEN:") and len(TOKENS(Q6_FINAL_ANSWER)) < 100


    # "QUESTION SEVEN:\n"
        # "Q6_FINAL_ANSWER_TODO": Q6_FINAL_ANSWER,
        # "Q6_JUSTIFICATION_TODO": Q6_JUSTIFICATION,
# "Q7_FINAL_ANSWER_TODO": Q7_FINAL_ANSWER,
# "Q8_FINAL_ANSWER_TODO": Q8_FINAL_ANSWER,
# "Q9_FINAL_ANSWER_TODO": Q9_FINAL_ANSWER,
# "Q10_FINAL_ANSWER_TODO": Q10_FINAL_ANSWER,
# "Q7_JUSTIFICATION_TODO": Q7_JUSTIFICATION,
# "Q8_JUSTIFICATION_TODO": Q8_JUSTIFICATION,
# "Q9_JUSTIFICATION_TODO": Q9_JUSTIFICATION,
# "Q10_JUSTIFICATION_TODO": Q10_JUSTIFICATION,

# example - example scenic program * static
# description - original description
# vehicles - valid list of Scenic vehicles * static
# objects - valid list of Scenic objects * static

# "Each expert and the final answer should be provided in the following format:\n"
# "MISSING_ENV_INFO:\n"
# "<missing_env_info>\n"


# "Here is one example of a fully compiling Scenic program:\n"
# "{example_1}\n"


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
    # reasoning_funcs = [generate_reasoning, generate_reasoning2]
    # lmql_tot_full = {}
    # for reasoning_count, reasoning_func in enumerate(reasoning_funcs):
    #     lmql_tot_temp = reasoning_func(description, example, towns, vehicles, objects, weather, lmql_tot_full)
    #     update(temp = lmql_tot_temp, full = lmql_tot_full)
    #     print(f'Completed generate reasoning pt {reasoning_count+1}/{len(reasoning_funcs)}!')

    # print('$%$%$%')
    # # print(lmql_tot_full)
    # for key in sorted(lmql_tot_full.keys()):
    #     if 'FINAL_ANSWER' in key:
    #         print(key, '-', lmql_tot_full[key])
    # print('$%$%$%')
    # #End reasoning

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

            uncompiled_scenic = final_scenic + '\n' + template_sections[i].format_map(lmql_outputs).strip()
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
                lmql_outputs_tmp = regenerate_scenic(model_input, working_scenic, lmql_outputs)
                lmql_outputs = {k: v for k, v in lmql_outputs_tmp.items() if v is not None} ### this is bad
                num_retries -= 1
            else:
                final_scenic = uncompiled_scenic
                working_key = section_keys.pop(0)
                lmql_outputs.pop(working_key, None)
                i += 1
                num_retries = max_retries
        if num_retries == 0:
            print("RAN OUT OF RETRIES RIP")
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