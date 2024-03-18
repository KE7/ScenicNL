import lmql
import numpy as np
import string
import time
import tempfile
import scenic
import os

@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_scenic_code(example_prompt, towns, vehicles, weather, questions):
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
    "[OTHER_CONSTANTS]\n"  where STOPS_BEFORE(OTHER_CONSTANTS, "## DEFINING BEHAVIORS")
    
    "## DEFINING BEHAVIORS\n"
    "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "## DEFINING SPATIAL RELATIONS") and len(TOKENS(SPATIAL_RELATIONS)) < 200

    "## DEFINING SPATIAL RELATIONS\n"
    "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 500
    
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

@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def generate_reasoning(example_prompt, towns, vehicles, weather):
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

    "Based on the description, what are the all of the objects that need to be included in the scene?\n"
    "First provide step-by-step justification for all objects you chosen then provide your final answer as:\n"

    "JUSTIFICATION:\n"
    "[Q1_JUSTIFICATION]\n" where STOPS_BEFORE(Q1_JUSTIFICATION, "FINAL ANSWER:") and len(TOKENS(Q1_JUSTIFICATION)) < 600

    "FINAL ANSWER:\n"
    "[Q1_FINAL_ANSWER]\n" where STOPS_BEFORE(Q1_FINAL_ANSWER, "F) and len(TOKENS(Q1_FINAL_ANSWER)) < 200

    
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

# example - example scenic program * static
# description - original description
# vehicles - valid list of Scenic vehicles * static
# objects - valid list of Scenic objects * static


# "Here is one example of a fully compiling Scenic program:\n"
# "{example_1}\n"
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
        "[OTHER_CONSTANTS]\n"  where STOPS_BEFORE(OTHER_CONSTANTS, "## DEFINING BEHAVIORS")
    else:
        OTHER_CONSTANTS = None
    
    if "VEHICLE_BEHAVIORS_TODO" in lmql_outputs:
        "## DEFINING BEHAVIORS\n"
        "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "## DEFINING SPATIAL RELATIONS") and len(TOKENS(SPATIAL_RELATIONS)) < 200
    else:
        BEHAVIORS = None
    

    if "SPATIAL_RELATIONS_TODO" in lmql_outputs:
        "## DEFINING SPATIAL RELATIONS\n"
        "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 500
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
    constructs a scenic program using the template in lmql_template.scenic 
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
        print(template_sections)
        final_scenic = template_sections[0].format_map(lmql_outputs) + '\n' + template_sections[1].format_map(lmql_outputs) #this should compile everytime
        
        i = 2
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
        print(template_sections)
        final_scenic = template_sections[0].format_map(lmql_outputs) + '\n' + template_sections[1].format_map(lmql_outputs) #this should compile everytime
        
        i = 2
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
            if works: ast = scenic.syntax.parser.parse_file(fname)
            print('No compilation error! (2/2)')
        except Exception as e:
            try:
                error_message = f"Error details below..\nerror message: {str(e)}\nerror text: {e.text}\nerror lineno: {e.lineno}\nend_lineno: {e.end_lineno}\nerror offset: {e.offset}\nerror end_offset: {e.end_offset}"
                print(error_message)
            except:
                error_message = f'Error details below..\nerror message: {str(e)}'
                print(error_message)
            works = False
    return works, error_message