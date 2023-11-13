import lmql

@lmql.query()
def generate_text_description(text_description, scenic_program):
    '''lmql
    "This is an example of a Scenic program: {scenic_program}\n"
    
    "We will write a new Scenic program based on the inputted natural language description."
    "The text description is: {text_description}\n"
    
    "This is the template we will fill out to write our Scenic program: {scenic_template}."
    "We will fill out each section marked with TODO.\n"
    
    "The first part of the new Scenic Program is a text description of the scenario."
    "This will complete the TEXT_DESCRIPTION_TODO section in the given template."
    "Please repeat the all of the text description in this part as a multi-line comment [TEXT_DESCRIPTION].\n" where type(TEXT_DESCRIPTION) == str and STOPS_BEFORE(TEXT_DESCRIPTION, "##")
    
    "The next section of the Scenic Program is setting the map and model."
    "This gives us all definitions of objects and their properties such as vehicle types, pedestrians, and road library."
    "Please select a CARLA map [CARLA_MAP_NAME] that will complete the CARLA_MAP_NAME_TODO in the template.\n" where type(CARLA_MAP_NAME) == str and CARLA_MAP_NAME in towns
    
    "The next section of our template is CONSTANTS."
    "First pick the vehicle blueprint ID that most closely matches the description but is supported in CARLA: [EGO_VEHICLE_BLUEPRINT_ID]" where type(EGO_VEHICLE_BLUEPRINT_ID) == str and EGO_VEHICLE_BLUEPRINT_ID in vehicles
    "The ego vehicle should also have a speed [EGO_VEHICLE_SPEED] based on the text description.\n" where INT(EGO_VEHICLE_SPEED) 
    
    "The next section of our template is OTHER CONSTANTS."
    "Include any other variables that you may need for the rest of the scenic program especially those "
    "that may describe behaviors: [OTHER_VARIABLES].\n" where STOPS_BEFORE(OTHER_VARIABLES, "\n\n")
    
    "The next section of the template is DEFINING BEHAVIORS."
    "Define the behavior of the ego vehicle and any other vehicles according to the description."
    "This where the dynamics of a scene occur [VEHICLE_BEHAVIORS]\n" where STOPS_BEFORE(VEHICLE_BEHAVIORS, "DEFINING SPATIAL RELATIONS")

    "The next section of the template is DEFINING SPATIAL RELATIONS."
    "Define the spatial relation of the ego vehicle and any other vehicles or obstacles"
    "according to the description: [SPATIAL_RELATIONS]\n" where STOPS_BEFORE(SPATIAL_RELATIONS, "require")

    "The last section is POSTCONDITIONS."
    "Summarize any postconditions or properties that the text description may have implied."
    "[POSTCONDITIONS]\n"
    
    
    return {
        "TEXT_DESCRIPTION": TEXT_DESCRIPTION,
        "CARLA_MAP_NAME" : CARLA_MAP_NAME,
        "EGO_VEHICLE_BLUEPRINT_ID" : EGO_VEHICLE_BLUEPRINT_ID,
        "EGO_VEHICLE_SPEED" : EGO_VEHICLE_SPEED,
        "OTHER_VARIABLES" : OTHER_VARIABLES,
        "VEHICLE_BEHAVIORS" : VEHICLE_BEHAVIORS,
        "SPATIAL_RELATIONS" : SPATIAL_RELATIONS,
        "POSTCONDITIONS" : POSTCONDITIONS
    }
    
    
    '''