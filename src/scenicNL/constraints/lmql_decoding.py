import lmql
import numpy as np
import string
import time
import scenic


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
def fix_starting_other_constants(fix_prompt, working_scenic):
    '''lmql
    
    {fix_prompt}

    {working_scenic}   
    "[OTHER_CONSTANTS]\n"  where STOPS_BEFORE(OTHER_CONSTANTS, "## DEFINING BEHAVIORS")
    
    "## DEFINING BEHAVIORS\n"
    "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "## DEFINING SPATIAL RELATIONS") and len(TOKENS(SPATIAL_RELATIONS)) < 200

    "## DEFINING SPATIAL RELATIONS\n"
    "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 500
    
    return {
        "OTHER_CONSTANTS_TODO" : OTHER_CONSTANTS,
        "VEHICLE_BEHAVIORS_TODO" : BEHAVIORS,
        "SPATIAL_RELATIONS_TODO" : SPATIAL_RELATIONS,
    }

    '''

@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def fix_starting_behaviors(fix_prompt, working_scenic):
    '''lmql
    
    {fix_prompt}

    {working_scenic}   
   
    "## DEFINING BEHAVIORS\n"
    "[BEHAVIORS]"  where  STOPS_BEFORE(BEHAVIORS, "## DEFINING SPATIAL RELATIONS") and len(TOKENS(SPATIAL_RELATIONS)) < 200

    "## DEFINING SPATIAL RELATIONS\n"
    "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 500
    
    return {
        "VEHICLE_BEHAVIORS_TODO" : BEHAVIORS,
        "SPATIAL_RELATIONS_TODO" : SPATIAL_RELATIONS,
    }

    '''

@lmql.query(model ='openai/gpt-3.5-turbo-instruct', max_len=10000)
def fix_starting_spatial_relations(fix_prompt, working_scenic):
    '''lmql
    
    {fix_prompt}

    {working_scenic}   
   
    "## DEFINING SPATIAL RELATIONS\n"
    "[SPATIAL_RELATIONS]\n" where len(TOKENS(SPATIAL_RELATIONS)) < 500
    
    return {
        "SPATIAL_RELATIONS_TODO" : SPATIAL_RELATIONS,
    }

    '''



def strip_other_constants(other_constants):
    """
    removes weird number characters at the beginning of other constants generation
    """
    nonalpha = string.digits + string.punctuation + string.whitespace
    return other_constants.lstrip(nonalpha)
    


def construct_scenic_program(example_prompt, nat_lang_scene_des, segmented_retry=True):
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
    if not segmented_retry:
        final_scenic = scenic_template.format_map(lmql_outputs)
    else:
        template_sections = scenic_template.split("##")
        final_scenic = template_sections[0].format_map(lmql_outputs) #this should compile everytime

        for i in range(1, len(template_sections)):
            uncompiled_scenic = final_scenic + '\n' + template_sections[i].format_map(lmql_outputs)
            #check if compiles
            compiles, error_message = check_compile(uncompiled_scenic)
            if not compiles:
                #regenerate this section and next
                lmql_outputs = regenerate_lmql(uncompiled_scenic, error_message)
            else:
                final_scenic = uncompiled_sceni

    return final_scenic

def check_compile(scenic_program):
    with tempfile.NamedTemporaryFile(dir=retries_dir, delete=False, suffix='.scenic') as temp_file:
        fname = temp_file.name
        print(f'$$$: {fname}')
        try:
            # ast = scenic.syntax.parser.parse_file(fname)
            scenario = scenic.scenarioFromFile(fname, mode2D=True)
            if verbose_retry: print('No error!')
            retries = 0 # If this statement is reached program worked -> terminates loop
            return True, ""
        except Exception as e:
            if verbose_retry: print(f'Retrying... {retries}')
            try:
                error_message = f"Error details below..\nmessage: {str(e)}\ntext: {e.text}\nlineno: {e.lineno}\nend_lineno: {e.end_lineno}\noffset: {e.offset}\nend_offset: {e.end_offset}"
                if verbose_retry: print(error_message)
            except:
                error_message = f'Error details below..\nmessage: {str(e)}'
                if verbose_retry: print(error_message)
            return False, error_message
    

