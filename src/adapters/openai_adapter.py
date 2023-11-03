import json
from typing import Dict
from model_adapter import ModelAdapter
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import os

import openai

from common import LLMPromptType, ModelInput


class OpenAIModel(Enum):
    GPT_35_TURBO = "gpt-3.5-turbo-0613"
    GPT_4 = "gpt-4-0613"


class OpenAIAdapter(ModelAdapter):
    """
    This class servers as a wrapper for the OpenAI API.
    """
    def __init__(self, model: OpenAIModel):
        super().__init__()
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model
        
    def _zero_shot_prompt(
            self,
            model_input: ModelInput
        ) -> list[Dict[str, str]]:
        """
        Format the message for the OpenAI API for zero shot prediction.
        @TODO: Ana please figure out this format.
        """
        return [
            {"role": "system", "content": "Please generate a scenic program for a CARLA " +
             "simulation from this natural language description."},
            {"role": "user", "content": model_input.nat_lang_scene_des},
        ]
    
    def _few_shot_prompt(
            self,
            model_input: ModelInput
        ) -> list[Dict[str, str]]:
        """
        Format the message for the OpenAI API for few shot prediction.
        @TODO: Devan please figure out this format.
        """
        return [
            {"role": "system", "content": "Please generate a scenic program for a CARLA " +
             "simulation from this natural language description." + 
             "Here are some examples of how to do that: " + model_input.examples[0] +
             model_input.examples[1] + model_input.examples[2] + model_input.examples[3]},
            {"role": "user", "content": model_input.nat_lang_scene_des},
        ]

    def _scenic_tutorial_prompt(
        self,
        model_input: ModelInput
    ) -> list[Dict[str, str]]:
        """
        Format the message for the OpenAI API for scenic tutorial (technically zero-shot?) prediction.
        @TODO: Karim feel free to move this code or rename this to a more relevant name.
        @TODO: Experiments - moving most of sys prompt to first user prompt, one vs multiple messages, etc.
        Note: did this all with a single user prompt and no system prompt locally, results may differ.
        Note: API for GPT 3.5 not great with long system prompts, may help to break into 1+ user prompts.
        """
        return [
            {"role": "system", "content": "Please generate a scenic program for a CARLA " +
             "simulation from a natural language description.\n" + self._format_scenic_tutorial_prompt(model_input)},
            {"role": "user", "content": "Natural language description: " + model_input.nat_lang_scene_des},
        ]

    def _python_api_prompt(
        self,
        model_input: ModelInput
    ) -> list[Dict[str, str]]:
        """
        Format the message for the OpenAI API for scenic3_api usage (?) prediction.
        @TODO: Karim how can we call expr() or eval() on the output of this prompt?

        @TODO: Refer to the TODOs below where the API is for an updated list of TODOs.
        @TODO: TODOs presented for _scenic_tutorial_prompt still hold for this prompt.
        Note that some API issues can be resolved by changing prompting and vice versa.
        """
        return [
            {"role": "system", "content": "Please write me python3 code to generate a scenic program for a CARLA " +
             "simulation from a natural language description using the scenic3 API as described below.\n" + 
             self._format_python_api_prompt(model_input)},
            {"role": "user", "content": "Natural language description: " + model_input.nat_lang_scene_des},
        ] 
    
    def _format_message(
        self,
        *,
        model_input: ModelInput,
        prompt_type: LLMPromptType,
    ) -> list[Dict[str, str]]:
        """
        Format the message for the OpenAI API.
        """
        if prompt_type == LLMPromptType.PREDICT_ZERO_SHOT:
            return self._zero_shot_prompt(model_input=model_input)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT:
            return self._few_shot_prompt(model_input=model_input)
        elif prompt_type == LLMPromptType.PREDICT_SCENIC_TUTORIAL:
            return self._scenic_tutorial_prompt(model_input=model_input)
        elif prompt_type == LLMPromptType.PREDICT_PYTHON_API:
            return self._python_api_prompt(model_input=model_input)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

    def get_cache_key(
        self,
        *,
        model_input: ModelInput,
        temperature: float,
        max_tokens: int,
        prompt_type: LLMPromptType,
    ) -> str:
        return json.dumps(
            {
                "model": self.model.value,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message": self._format_message(model_input=model_input, prompt_type=prompt_type),
            },
            sort_keys=True,
        )

    @retry(
        wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
    )
    def _predict(
        self, 
        *, 
        model_input: ModelInput, 
        temperature: float, 
        max_length_tokens: int,
        prompt_type: LLMPromptType,
    ) -> str:
        messages = self._format_message(model_input=model_input, prompt_type=prompt_type)

        response = openai.Completion.create(
            temperature=temperature,
            model=self.model.value,
            max_tokens=max_length_tokens,
            messages=messages
        )
        return response.choices[0].message.content

    def _format_scenic_tutorial_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        """
        Formats the message providing introduction to Scenic language and syntax.
        """
        preamble = "param map = localPath('../../../assets/maps/CARLA/Town05.xodr')\nparam carla_map = 'Town05'\nmodel scenic.simulators.carla.model"
        param_def = "speed = Range(15, 25)"
        obstacle_def = "lane = Uniform(*network.lanes)\nspawnPt = new OrientedPoint on lane.centerline\nobstacle = new Trash at spawnPt offset by Range(1, -1) @ 0"
        ego_def = "behavior EgoBehavior(speed=10):\n\ttry:\n\t\tdo FollowLaneBehavior(speed)\n\tinterrupt when withinDistanceToAnyObjs(self, EGO_BRAKING_THRESHOLD):\n\t\ttake SetBrakeAction(BRAKE_ACTION)"
        ego_assign = "ego = new Car following roadDirection from spawnPt for Range(-50, -30),\n\twith blueprint EGO_MODEL,\n\twith behavior EgoBehavior(EGO_SPEED)"
        ex_require = "require (distance to intersection) > 75"
        ex_terminate = "terminate when ego.speed < 0.1 and (distance to obstacle) < 15"
        st_prompt = "Here is a quick tutorial about the Scenic language."
        st_prompt += "\nScenic scripts are typically divided into three sections: parameter definitions, scene setup, and behaviors."
        st_prompt += "\n\n1. Parameter Definitions:\n In the parameter definitions section, you handle imports and define any parameters your scenario will use."
        st_prompt += "\nA Scenic script begins with importing necessary libraries."
        st_prompt += f"\nThe first lines could be: \"{preamble}\" to import the simulator library."
        st_prompt += f"\nThen define any scene parameters, for example: \"{param_def}\" defines a parameter speed with values ranging from 15 to 25."
        st_prompt += "\n\n2. Scene Setup:\nIn the scene setup section, you describe the static aspects of the scenario."
        st_prompt += f"\nFor example, \"{obstacle_def}\" creates a Trash obstacle offset from the centerline of a random lane."
        st_prompt += "\n\n3. Behaviors:\nIn the behavior section, you describe the dynamic aspects of the scenario."
        st_prompt += f"\nFor example, \"{ego_def}\" defines a behavior for a vehicle to follow a lane and brake once any vehicle comes within a certain distance."
        st_prompt += f"\nAfter this, \"{ego_assign}\" defines a dynamic agent with this behavior and other properties. All scenes must have an ego vehicle."
        st_prompt += "\nScenic provides a rich set of built-in behaviors but also allows for custom behavior definitions."
        st_prompt += "\nAfter all behaviors and agents are defined, the last optional require and terminate statements can be used to enforce conditions that determine how long the simulationa runs."
        st_prompt += f"\nFor example (require statement): \"{ex_require}\" or (terminate statement): \"{ex_terminate}\" might be added to the end of a program but are optional."
        st_prompt += "\nThe output should be a single block of Python code for a Scenic script that sets up a scene that models the given natural language description."
        return st_prompt

    def _format_python_api_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        pa_prompt = "Here is a comprehensive tutorial about the scenic3_api."
        pa_prompt += " The Scenic-3 script is structured in a way that initially sets up the scenario by specifying the map, model, and any constants, followed by defining behaviors and object placements for scenario dynamics."
        
        # Setup part
        pa_prompt += "\n\n--- Setup ---"
        pa_prompt += "\nIn scenic3_api, to specify a map, you would use the `set_map` method, for example: `scenic3.set_map('../../../assets/maps/CARLA/Town01.xodr')`."
        pa_prompt += "\nTo specify the model, you would use the `set_model` method, for example: `scenic3.set_model('scenic.simulators.carla.model')`."
        pa_prompt += "\nConstants can be defined using the `define_constant` method, for example: `scenic3.define_constant('EGO_SPEED', 10)`."
        # pa_prompt += "\nIn scenic3_api, to specify a map, you would use the `set_map` method, for example: `scenic3.set_map('../../../assets/maps/CARLA/Town01.xodr')`."
        # pa_prompt += "\nTo specify the model, you would use the `set_model` method, for example: `scenic3.set_model('scenic.simulators.carla.model')`."
        # pa_prompt += "\nConstants can be defined using the `define_constant` method, for example: `scenic3.define_constant('EGO_SPEED', 10)`."
        
        # Behavior definitions part
        pa_prompt += "\n\n--- Behavior Definitions ---"
        # pa_prompt += "\nBehaviors are defined using the `define_behavior` method, for example: `scenic3.define_behavior('EgoBehavior', speed=10)`."
        # pa_prompt += "\nInside the behavior, actions are specified using the `do` method followed by the behavior name and parameters, for example: `scenic3.do('FollowLaneBehavior', speed)`."
        # pa_prompt += "\nLooping structures can be created with `do_while`, `do_until`, and `try_except` methods, for example: `scenic3.do_while('FollowLaneBehavior', speed, 'some_condition')`."
        # pa_prompt += "\nInterrupts can be used to specify conditions under which the behavior should be interrupted, for example: `scenic3.interrupt('withinDistanceToAnyCars(self, DISTANCE_THRESHOLD)')`."
        # pa_prompt += "\nAfter an interrupt, the `take` method can be used to specify an action to take, for example: `scenic3.take('SetBrakeAction', BRAKE_ACTION)`."
        pa_prompt += "\nIn scenic3_api, behaviors are defined using the `define_behavior` method."
        pa_prompt += "\nInside the behavior, code is indented in python style and specified using the `do` method followed by the behavior name and parameters."
        pa_prompt += "\nFor example (multiline with indents): `scenic3.define_behavior('EgoBehavior', speed=10)`."
        pa_prompt += "\n`scenic3.do('FollowLaneBehavior', speed, indent=1)`." # speed
        pa_prompt += "\nLooping structures can be created with `do_while`, `do_until`, and `try_except` methods."
        pa_prompt += "\nFor example (multiline with indents): `scenic3.define_behavior('EgoBehavior', safety_distance=10)`."
        pa_prompt += "\n`scenic3.do_while('FollowLaneBehavior', speed, 'withinDistanceToAnyObjs(self, DISTANCE_THRESHOLD)', indent=1)`." # speed
        pa_prompt += "\nInterrupts can be used to specify conditions under which the behavior should be interrupted."
        pa_prompt += "\nFor example: `scenic3.interrupt('withinDistanceToAnyCars(self, DISTANCE_THRESHOLD)')`."
        pa_prompt += "\nAfter an interrupt, the `take` method can be used to specify an action to take."
        pa_prompt += "\nFor example: `scenic3.take('SetBrakeAction', BRAKE_ACTION)`."
        # pa_prompt += "\nAssignments for objects to have blueprints and behaviors can be done using `assign_blueprint` and `assign_behavior` methods."
        # pa_prompt += "\nFor example: `scenic3.assign_blueprint(ego, 'vehicle.lincoln.mkz_2017')` and `scenic3.assign_behavior(ego, 'EgoBehavior', EGO_SPEED)`."
        
        # Assignments and object placements part
        pa_prompt += "\n\n--- Assignments and Object Placements ---"
        # pa_prompt += "\nAssignments for objects to have blueprints and behaviors can be done using `assign_blueprint` and `assign_behavior` methods, for example: `scenic3.assign_blueprint(ego, 'vehicle.lincoln.mkz_2017')` and `scenic3.assign_behavior(ego, 'EgoBehavior', EGO_SPEED)`."
        pa_prompt += "\nNew objects can be created and placed using the `new` method, for example: `scenic3.new(var_name='ego', obj_type='Car', at='spawnPt', blueprint='EGO_MODEL', behavior='EgoBehavior(EGO_SPEED)')`."
        pa_prompt += "\nSpatial relations between objects can be defined using the `spatial_relation` method, for example: `scenic3.spatial_relation('ego', 'following', 'leadCar', distance='Range(-15, -10)')`."

        # pa_prompt += "\n\nYour output must be only executable Python code to create a complete Scenic-3 script as per the given requirements. No explanation needed."
        pa_prompt += "\n\n--- Output Rules ---"
        pa_prompt += "\nYour output must be only executable Python code that sets up the scenario. Every line should invoke a method or nested method of the form scenic3.<method>(args) - every line should start wiht scenic3.<method>(args) and no placeholder values <> should be present. No explanation or imports needed."
        return pa_prompt



"""
Scenic3 Mini API Helper Class
@TODO: Karim convert args to string when possible to prevent errors from failing.
@TODO: Devan add try / except blocks inside methods to avoid errors thrown by string arguments.
(For context, calling ```exec()``` on all lines of LLM output throws some errors
 that can be caught by calling the str() command on function calling inputs or adding quotes to args. )
@TODO: Devan create a more extensive API that can fully express Scenic programs.
(All TODOs for Devan but anyone with extra time welcome to look over.)
@TODO: Devan make the API usage 1. more closely resemble UCLID5 paper 2. find better indentation solution

Usage:
llm_output_text = (.. output of LLM call - see below for an ex..)
scenic3 = Scenic3()
exec(llm_output_text) 
# equivalent of calling [eval(line) for line in llm_output_text.split('\n')]
# line-by-line eval approach might be more reliable


# Example LLM Output - yes it's slightly improper scenic #
scenic3.set_map('../../../assets/maps/CARLA/Town01.xodr')
scenic3.set_model('scenic.simulators.carla.model')

scenic3.define_constant('EGO_SPEED', 10)

scenic3.define_behavior('EgoBehavior', speed=EGO_SPEED)
scenic3.do('FollowLaneBehavior', speed=EGO_SPEED, indent=1)
scenic3.do_while('FollowIntersectionBehavior', indent=2, condition='not hasClearedIntersection()')
scenic3.interrupt('withinDistanceToAnyCars(self, DISTANCE_THRESHOLD)')
scenic3.take('SetBrakeAction', 1)

scenic3.new(var_name='ego', obj_type='Car', at='spawnPt', blueprint='EGO_MODEL', behavior='EgoBehavior(EGO_SPEED)')
scenic3.new(var_name='leadCar', obj_type='Car', at='leadSpawnPt')
scenic3.spatial_relation('ego', 'following', 'leadCar', distance='Range(-10, -5)')
# End Ex #

## Example Expr Error ## 
NameError                                 Traceback (most recent call last)
/... ... line 3
      1 print(llm_output_text)
      2 scenic3 = Scenic3()
----> 3 exec(llm_output_text)

File <string>:6

NameError: name 'EGO_SPEED' is not defined
>> from this: scenic3.define_behavior('EgoBehavior', speed=EGO_SPEED)
"""

_indent_ = '    '
class Scenic3:

    def __init__(self):
        self.code = []

    def set_map(self, map_name, indent=0):
        indent_str = _indent_ * indent
        self.code.append(f"{indent_str}param map = localPath('{map_name}')")

    def set_model(self, model_name, indent=0):
        indent_str = _indent_ * indent
        self.code.append(f"{indent_str}model {model_name}")

    def define_constant(self, name, value, indent=0):
        indent_str = _indent_ * indent
        self.code.append(f"{indent_str}{name} = {value}")

    def define_behavior(self, name, indent=0, **kwargs):
        indent_str = _indent_ * indent
        kwargs_str = ', '.join(f'{str(k)}={str(v)}' for k, v in kwargs.items())
        self.code.append(f"{indent_str}behavior {name}({kwargs_str}):")

    def do(self, behavior_name, indent=0, **kwargs):
        indent_str = _indent_ * indent
        behavior_name = str(behavior_name)
        kwargs_str = ', '.join(f'{str(k)}={str(v)}' for k, v in kwargs.items())
        self.code.append(f"{indent_str}do {behavior_name}({kwargs_str})")

    def do_while(self, behavior_name, var_name, condition, indent=0):
        indent_str = _indent_ * indent
        behavior_name, var_name, condition = str(behavior_name), str(var_name), str(condition)
        self.code.append(f"{indent_str}do {behavior_name}({var_name}) while {condition}")

    def do_until(self, behavior_name, var_name, condition, indent=0):
        indent_str = _indent_ * indent
        behavior_name, var_name = str(behavior_name), str(var_name)
        self.code.append(f"{indent_str}do {behavior_name}({var_name}) until {condition}")

    def try_except(self, try_behavior, except_behavior, indent=0):
        indent_str = _indent_ * indent
        try_behavior, except_behavior = str(try_behavior), str(except_behavior)
        self.code.append(f"{indent_str}try:")
        self.code.append(f"{indent_str}{_indent_}do {try_behavior}")
        self.code.append(f"{indent_str}except:")
        self.code.append(f"{indent_str}{_indent_}do {except_behavior}")

    def interrupt(self, condition, indent=0, *args):
        indent_str = _indent_ * indent
        args_str = ', '.join(str(arg) for arg in args)
        self.code.append(f"{indent_str}interrupt when {condition}({args_str}):")

    def take(self, action_name, indent=0, **params):
        indent_str = _indent_ * indent
        params_str = ', '.join(f'{str(k)}={str(v)}' for k, v in params.items())
        self.code.append(f"{indent_str}take {action_name}({params_str})")

    def new(self, var_name, obj_type, at=None, indent=0, **kwargs):
        indent_str = _indent_ * indent
        new_line = f"{indent_str}{var_name} = new {obj_type.capitalize()}"
        
        if at:
            new_line += f" at {at}"
        
        for k, v in kwargs.items():
            new_line += f", {str(k)}={str(v)}"

        self.code.append(new_line)

    def spatial_relation(self, obj1, keyword, obj2, distance=None, indent=0):
        indent_str = _indent_ * indent

        if distance:
            self.code.append(f"{indent_str}{obj1} {keyword} {obj2} for {distance}")
        else:
            self.code.append(f"{indent_str}{obj1} {keyword} {obj2}")

    def Uniform(self, seq, indent=0):
        indent_str = _indent_ * indent
        return f"{indent_str}Uniform({seq})"

    def Range(self, start, end, indent=0):
        indent_str = _indent_ * indent
        return f"{indent_str}Range({start}, {end})"

    def require(self, condition, indent=0):
        indent_str = _indent_ * indent
        self.code.append(f"{indent_str}require {condition}")

    def terminate(self, condition, indent=0):
        indent_str = _indent_ * indent
        self.code.append(f"{indent_str}terminate when {condition}")

    def get_code(self):
        return "\n".join(self.code).strip()
