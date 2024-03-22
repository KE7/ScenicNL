from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.adapters.api_adapter import Scenic3

import os
import openai
import json
from enum import Enum
from typing import Dict
from scenicNL.common import LLMPromptType, ModelInput, format_scenic_tutorial_prompt
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


from scenicNL.constraints.lmql_decoding import construct_scenic_program, construct_scenic_program_tot


class LMQLModel(Enum):
    LMQL = "lmql" #TODO: we're just using the default lmql model rn, no real need to specify

class LMQLAdapter(ModelAdapter):
    """
    This class servers as a wrapper for the LMQL OpenAI API.
    """
    def __init__(self, model: LMQLModel):
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_ORGANIZATION") and len(os.getenv("OPENAI_ORGANIZATION")) > 0:
            openai.organization = os.getenv("OPENAI_ORGANIZATION")
        self.PROMPT_PATH = os.path.join(os.curdir, 'src', 'scenicNL', 'adapters', 'prompts')
        self._model = model
        
    def _lmql_prompt(
        self,
        model_input: ModelInput
    ) -> str:
         return (
            f"Please generate a scenic program for a CARLA "
            f"simulation to replicate the input natural language description below."
            f"Here are some examples scenic programs.\n{model_input.examples[1]}\n"
            f"Given the following report, write a scenic program that models it: \n"
            f"\n\n<user_input>{model_input.nat_lang_scene_des}\n\n</user_input> "
            f"Ouput the scenic program. Do not include any other text."
            f"\n\n"
        )
    
    def _lmql_correction_prompt(
        self,
        model_input: ModelInput
    ) -> str:
         
         return (
            f"Please generate a scenic program for a CARLA "
            f"simulation to replicate the input natural language description below."
            f"Here are some examples scenic programs.\n{model_input.examples[1]}\n"
            f"Given the following report, write a scenic program that models it: \n"
            f"\n\n<user_input>{model_input.nat_lang_scene_des}\n\n</user_input> "
            f"Ouput the scenic program. Do not include any other text."
            f"\n\n"
        )
    
    def _format_message(
        self,
        *,
        model_input: ModelInput,
        prompt_type: LLMPromptType,
        verbose: bool,
    ) -> str:
        """
        Formats the message to be sent to the API.
        """
        
        msg = self._lmql_prompt(model_input=model_input)
        
        if verbose:
            print(f"Message formatted using {LLMPromptType(prompt_type).name}")
            print("Message = ", msg)

        if msg is None:
            raise NotImplementedError(f"Prompt type {prompt_type} not implemented for LMQL.")
        
        return msg
    
    
    def get_cache_key(
        self, 
        *, 
        model_input: ModelInput, 
        temperature: float, 
        max_tokens: int, 
        prompt_type: LLMPromptType
    ) -> str:
        return json.dumps(
            {
                "model": self._model.value,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message": self._format_message(model_input=model_input, prompt_type=prompt_type, verbose=False),
            },
            sort_keys=True,
        )

    @retry(
        wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(1)
    )
    def _predict(
        self, 
        *, 
        model_input: ModelInput, 
        temperature: float, 
        max_length_tokens: int,
        prompt_type: LLMPromptType,
        verbose: bool,
        max_retries: int,
    ) -> str:
        
        example_prompt = self._format_message(model_input=model_input, prompt_type=prompt_type, verbose=verbose)
        
        if prompt_type == LLMPromptType.PREDICT_LMQL_TOT_RETRY:
            response = construct_scenic_program_tot(model_input, example_prompt, model_input.nat_lang_scene_des)
        else:
            response = construct_scenic_program(model_input, example_prompt, model_input.nat_lang_scene_des)
        return response