import lmql
import numpy as np
import string
import time

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
    "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "## DEFINING SPATIAL RELATIONS")

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


def strip_other_constants(other_constants):
    """
    removes weird number characters at the beginning of other constants generation
    """
    nonalpha = string.digits + string.punctuation + string.whitespace
    return other_constants.lstrip(nonalpha)
    


def construct_scenic_program(example_prompt, nat_lang_scene_des):
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
    
    #complete the template using the lmql_outputs
    final_scenic = scenic_template.format_map(lmql_outputs)

    return final_scenic