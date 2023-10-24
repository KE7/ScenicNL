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