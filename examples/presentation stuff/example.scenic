# SCENARIO DESCRIPTION
"""
On August 16th, 2022 at 3:52 AM PST a Waymo Autonomous Vehicle (“Waymo AV”) operating in San Francisco, California was in a collision involving a SUV on Martin Luther King Jr. Drive at 25th Avenue.  The Waymo AV was traveling West on Martin Luther King Jr. Drive and stopped at the stop sign at the intersection with 25th Avenue. As the Waymo AV slowed to a stop, a SUV traveling in the opposite lane, East on Martin Luther King Jr. Drive, crossed the intersection with 25th Avenue and came to a stop in the opposite lane, next to the Waymo AV. After stopping, the SUV began moving again and the front  left corner of the SUV made contact with the left side of the Waymo AV. The SUV then left the scene. The Waymo AV sustained minor damage.
"""
# SCENARIO CODE

## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath(f'../../../assets/maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
param weather = 'ClearNoon'
model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = 'vehicle.lincoln.mkz_2017'
EGO_SPEED = 10
EGO_BRAKING_THRESHOLD = 12
SUV_SPEED = 15
SUV_BRAKING_THRESHOLD = 10
BRAKE_ACTION = 1.0


## DEFINING BEHAVIORS
# EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the SUV
behavior EgoBehavior(speed=10):
    try:
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyCars(self, EGO_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

# SUV BEHAVIOR: Follow lane, and brake after passing a threshold distance to the Waymo AV
behavior SUVBehavior(speed=15):
    try:
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyCars(self, SUV_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)



## DEFINING SPATIAL RELATIONS
# Please refer to scenic/domains/driving/roads.py how to access detailed road infrastructure
# 'network' is the 'class Network' object in roads.py

# make sure to put '*' to uniformly randomly select from all elements of the list, 'lanes'
lane = Uniform(*network.lanes)

waymoAV = new Car following roadDirection from lane.centerline for Range(0, 5),
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(EGO_SPEED)

suv = new Car following roadDirection from waymoAV for Range(0, 5),
        with behavior SUVBehavior(SUV_SPEED)

require (distance from waymoAV to suv) < 5
require (distance from waymoAV to intersection) > 10
terminate when (distance from waymoAV to suv) < 1 and (distance from waymoAV to suv) > 0.5
