import lmql

@lmql.query
def generate_scenic_carla(scenic_program_example):
    scenic_program_example
    "We are going to write a Scenic Program from a natural language description."
    "Because this natural language description is of a car driving scenario, we will "
    "use the car driving simulator CARLA to visualize the scene."

    "The first part of a Scenic Program is a text description of the scenario. "
    "Please repeat the input in this part as a multi-line comment [TEXT_DESCRIPTION]\n" where type(TEXT_DESCRIPTION) == str

    "The next part of the Scenic Program is setting the map and model."
    "This gives us all definitions of objects and their properties such as vehicle types, pedestrians, and road library."

    "param map = localPath([PATH_TO_CARLA_MAP])  # or other CARLA map that definitely works" where type(PATH_TO_CARLA_MAP) == str
    "param carla_map = [CARLA_MAP_NAME]" where type(CARLA_MAP_NAME) == str
    "Finally we have to declare a model so use the default CARLA one by repeating this: model scenic.simulators.carla.model "

    "Now we must pick a vehicle blueprint ID from the CARLA map for our EGO vehicle. "
    "Pick the one that most closely matches the description but is supported in CARLA. [EGO_VEHICLE_BLUEPRINT_ID: r'vehicle.(\w+-?\w+).(\w+)']\n"
    "The ego vehicle should also have a speed based on the description. [EGO_VEHICLE_SPEED: r'(\d+)']\n"
    "Include any other variables that you may need for the rest of the scenic program especially those "
    "that may describe behaviors: [OTHER_VARIABLES]\n"

    "Define the behavior of the ego vehicle and any other vehicles according to the description. "
    "This where the dynamics of a scene occur [VEHICLE_BEHAVIORS]\n"

    "Finally summarize any postconditions or properties that the text description may have implied. "
    "require [POSTCONDITIONS]\n"

    return {
        "TEXT_DESCRIPTION" : TEXT_DESCRIPTION,
        "PATH_TO_CARLA_MAP" : PATH_TO_CARLA_MAP,
        "CARLA_MAP_NAME" : CARLA_MAP_NAME,
        "EGO_VEHICLE_BLUEPRINT_ID" : EGO_VEHICLE_BLUEPRINT_ID,
        "EGO_VEHICLE_SPEED" : EGO_VEHICLE_SPEED,
        "OTHER_VARIABLES" : OTHER_VARIABLES,
        "VEHICLE_BEHAVIORS" : VEHICLE_BEHAVIORS,
        "POSTCONDITIONS" : POSTCONDITIONS,
    }
