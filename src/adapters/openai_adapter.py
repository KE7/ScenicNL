from api_adapter import Scenic3
from model_adapter import ModelAdapter

import json
from typing import Dict
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
        self.PROMPT_PATH = os.path.join(os.curdir, 'src', 'adapters', 'prompts')

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
        Add back + model_input.examples[3]
        return [
            {"role": "system", "content": "Please generate a scenic program for a CARLA " +
             "simulation from this natural language description." + 
             "Here are some examples of how to do that: " + model_input.examples[0] +
             model_input.examples[1] + model_input.examples[2]},
            {"role": "user", "content": model_input.nat_lang_scene_des},
        ]
        """
        return [
            {"role": "system", "content": "Please generate a scenic program for a CARLA " +
             "simulation based on the input natural language description below." + 
             "\n-- Here is a scenic tutorial. --\n" + self._format_scenic_tutorial_prompt(model_input) +
             "\n-- Here are some example scenic programs. --\n" + model_input.examples[0] 
             + model_input.examples[1] + model_input.examples[2]},
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
        self,
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
        main_prompt += f"\nThe output should be a single block of valid Scenic code from the API call: {model_input.nat_lang_scene_des}"
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
        if prompt_type.value == LLMPromptType.PREDICT_ZERO_SHOT.value:
            return self._zero_shot_prompt(model_input=model_input)
        elif prompt_type.value == LLMPromptType.PREDICT_FEW_SHOT.value:
            return self._few_shot_prompt(model_input=model_input)
        elif prompt_type.value == LLMPromptType.PREDICT_SCENIC_TUTORIAL.value:
            return self._scenic_tutorial_prompt(model_input=model_input)
        elif prompt_type.value == LLMPromptType.PREDICT_PYTHON_API.value:
            return self._python_api_prompt(model_input=model_input)
        elif prompt_type.value == LLMPromptType.PREDICT_PYTHON_API_ONELINE.value: # for one-line corrections of function calling
            return self._python_api_prompt_oneline(model_input=model_input)
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
        response = openai.ChatCompletion.create(
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
        st_prompt = ''
        with open(os.path.join(self.PROMPT_PATH, 'scenic_tutorial_prompt.txt')) as f:
            st_prompt = f.read()
        return st_prompt

    def _format_python_api_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        pa_prompt = ''
        with open(os.path.join(self.PROMPT_PATH, 'python_api_prompt.txt')) as f:
            pa_prompt = f.read()
        return pa_prompt

    def _api_fallback(
        self,
        model_input: ModelInput
    ) -> str:
        """Generate code from API calls, with _line_helper generating backups in case of API failures."""
        scenic3 = Scenic3() # function call aggregator
        description = model_input.nat_lang_scene_des
        for line in description.split('\n'):
            if not line: continue
            try:
                eval(line)
            except:
                new_input = ModelInput(examples=model_input.examples, nat_lang_scene_des=line)
                result = self._line_helper(new_input)
                scenic3.add_code(result.split('\n'))
        return scenic3.get_code()

    def _line_helper(
        self,
        line_input: ModelInput
    ) -> str:
        """
        Intent: use mini LLM calls to correct any malformed Scenic3_API calls (ones that fail `eval`).
        Idea: even imperfect API calls contain all info needed to formulate Scenic expression.
        @TODO: Karim how can I turn off caching for these one-line calls?
        Wasn't sure how to handle post-API-call eval from openai_adapter block.
        """
        prediction = self._predict(
            model_input=line_input,
            temperature=0,
            max_length_tokens=40,
            prompt_type=LLMPromptType.PREDICT_PYTHON_API_ONELINE, # @TODO: timeout 
        )
        return prediction