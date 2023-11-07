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
        Design considerations: short system prompt for GPT 3.5.
        Note: originally implemented with no system prompt, one system prompt locally.
        Note: future work - sys vs user prompt usage, one vs multiple messages, etc.
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
        @TODO: Karim let me know if there is a better way for me to expr() / eval() the output of this prompt?

        Note: TODOs presented for _scenic_tutorial_prompt still hold for this prompt.
        Note: Some API issues can be resolved by changing prompting and vice versa.
        """
        return [
            {"role": "system", "content": "Please write me python3 code to generate a scenic program for a CARLA " +
             "simulation from a natural language description using the scenic3 API as described below.\n" + 
             self._format_python_api_prompt(model_input)},
            {"role": "user", "content": "Natural language description: " + model_input.nat_lang_scene_des},
        ]

    def _python_api_prompt_oneline(
        model_input: ModelInput
    ) -> list[Dict[str, str]]:
        """
        Format the message for the OpenAI API for scenic3_api usage (?) prediction.
        @TODO: Karim let me know if there is a better way for me to expr() / eval() the output of this prompt?

        Note: TODOs presented for _scenic_tutorial_prompt still hold for this prompt.
        Note: Some API issues can be resolved by changing prompting and vice versa.
        """
        intro_prompt = f"Consider the following Scenic-3 programs.\n"
        intro_prompt += f"\n\n\n{model_input.examples[0]}\n\n\n{model_input.examples[1]}\n\n\n{model_input.examples[2]}"
        main_prompt = "Write me one line of valid Scenic-3 code based on the Python API input provided."
        main_prompt += f"\nThe output should be a single block of valid Scenic code from the API call: {model_input}"
        main_prompt += "\nOutput just one short block of Scenic-3 code as your output, with four spaces per indent if any. Provide no other output text."

        return [
            {"role": "user", "content": intro_prompt},
            {"role": "user", "content": main_prompt},
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
        elif prompt_type == LLMPromptType.PREDICT_PYTHON_API_ONELINE: # for one-line corrections of function calling
            return OpenAIAdapter._python_api_prompt_oneline(model_input=model_input)
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

        # Behavior definitions part
        pa_prompt += "\n\n--- Behavior Definitions ---"
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

        # Assignments and object placements part
        pa_prompt += "\n\n--- Assignments and Object Placements ---"
        pa_prompt += "\nNew objects can be created and placed using the `new` method, for example: `scenic3.new(var_name='ego', obj_type='Car', at='spawnPt', blueprint='EGO_MODEL', behavior='EgoBehavior(EGO_SPEED)')`."
        pa_prompt += "\nSpatial relations between objects can be defined using the `spatial_relation` method, for example: `scenic3.spatial_relation('ego', 'following', 'leadCar', distance='Range(-15, -10)')`."

        # pa_prompt += "\n\nYour output must be only executable Python code to create a complete Scenic-3 script as per the given requirements. No explanation needed."
        pa_prompt += "\n\n--- Output Rules ---"
        pa_prompt += "\nYour output must be only executable Python code that sets up the scenario. Every line should invoke a method or nested method of the form scenic3.<method>(args) - every line should start wiht scenic3.<method>(args) and no placeholder values <> should be present. No explanation or imports needed."
        pa_prompt += "\nPlease enter all function inputs with strings surrounding, ie scenic3.do('AvoidObstacleBehavior', speed='EGO_SPEED', indent=1)"
        return pa_prompt
