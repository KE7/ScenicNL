import lmql
import numpy as np
import string
import time
import tempfile
import scenic
import os

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
def regenerate_scenic(uncompiled_scenic, error_message, lmql_outputs):
    '''lmql
    
    "Scenic is a probabilistic programming language for modeling the environments of autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world. We use CARLA to render the scenes and simulate the agents.\n"

    "Here is one example of a fully compiling Scenic program:\n"
    "{example_1}\n"

    "Create a fully compiling Scenic program that models the description based on:\n"

    "1. The following natural language description:\n"
    "{natural_language_description}\n"

    "2. The following scenic program with compiler errors that models the description:\n"
    "{first_attempt_scenic_program}\n"

    "3. The first compiler error raised with the scenic program:\n"
    "{compiler_error}\n"

    "Please output a modified version of scenic_program modified so the compiler error does not appear.\n"

    "OUTPUT NO OTHER LEADING OR TRAILING TEXT OR WHITESPACE BESIDES THE CORRECTED SCENIC PROGRAM. NO ONE CARES.\n"

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
    


def construct_scenic_program(example_prompt, nat_lang_scene_des, segmented_rety=False):
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
    segmented_retry = True
    if not segmented_retry:
        final_scenic = scenic_template.format_map(lmql_outputs)
    else:
        print('Segmenting retries')
        template_sections = scenic_template.split("##")
        template_sections = ["##" + section for section in template_sections]
        print(template_sections)
        final_scenic = template_sections[0].format_map(lmql_outputs) + '\n' + template_sections[1].format_map(lmql_outputs) #this should compile everytime
        
        i = 2
        max_retries = 3
        while i < len(template_sections) and max_retries > 0:
            uncompiled_scenic = final_scenic + '\n' + template_sections[i].format_map(lmql_outputs)
            #check if compiles
            compiles, error_message = check_compile(uncompiled_scenic)
            if not compiles:
                print(f'DID NOT COMPILE: {uncompiled_scenic}')
                #regenerate this section and next
                print(f"ERROR {error_message}")
                lmql_outputs = regenerate_scenic(uncompiled_scenic, error_message, lmql_outputs)
                lmql_outputs = {k: v for k, v in lmql_outputs.items() if v is not None}
                max_retries -= 1
            else:
                final_scenic = uncompiled_scenic
                working_key = section_keys.pop(0)
                lmql_outputs.pop(working_key, None)
                i += 1
                max_retries = 3
        if max_retries == 0:
            print("RAN OUT OF RETRIES RIP")

    return final_scenic

def check_compile(scenic_program):
    retries_dir = os.path.join(os.curdir, 'temp_dir')
    os.makedirs(retries_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=retries_dir, delete=False, suffix='.scenic') as temp_file:
        fname = temp_file.name
        try:
            # ast = scenic.syntax.parser.parse_file(fname)
            scenario = scenic.scenarioFromFile(fname, mode2D=True)
            print('No error!')
            retries = 0 # If this statement is reached program worked -> terminates loop
            return True, ""
        except Exception as e:
            print(f'Retrying... {retries}')
            try:
                error_message = f"Error details below..\nmessage: {str(e)}\ntext: {e.text}\nlineno: {e.lineno}\nend_lineno: {e.end_lineno}\noffset: {e.offset}\nend_offset: {e.end_offset}"
                print(error_message)
            except:
                error_message = f'Error details below..\nmessage: {str(e)}'
                print(error_message)
            return False, error_message