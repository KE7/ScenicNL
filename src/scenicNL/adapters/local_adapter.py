from enum import Enum
import json

import requests
from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.common import LOCAL_MODEL_DEFAULT_PARAMS, LOCAL_MODEL_ENDPOINT, LLMPromptType, ModelInput, VectorDB
from scenicNL.constraints.gbnf_decoding import CompositionalScenic


class LocalModel(Enum):
    local = "local"
    

class LocalAdapter(ModelAdapter):

    def __init__(self, model : LocalModel):
        super().__init__()
        self._model = model
        self.index = VectorDB(index_name='scenic-programs')

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
        msg = None
        # TODO: Add more prompt types

        # if msg is None:
        #     raise NotImplementedError(f"Prompt type {prompt_type} was not formatted for Local model {self._model.value}")
        
        return msg
    

    def _predict(
        self, 
        *, 
        model_input: ModelInput, 
        temperature: float, 
        max_length_tokens: int, 
        prompt_type: LLMPromptType,
        verbose: bool
    ) -> str:
        if prompt_type == LLMPromptType.COMPOSITIONAL_GBNF:
            program_generator = CompositionalScenic()
            return program_generator.compositionally_construct_scenic_program(
                model_input=model_input,
                temperature=temperature,
                max_tokens=max_length_tokens,
                verbose=verbose
            )

        prompt = self._format_message(
            model_input=model_input,
            prompt_type=prompt_type,
            verbose=verbose,
        )

        data = {"prompt": prompt, "temperature": temperature} | LOCAL_MODEL_DEFAULT_PARAMS
        if max_length_tokens > 0:
            data["max_tokens"] = max_length_tokens

        # TODO:
        # - Add logic for grammars then add the grammar to the request as: "grammar": grammar
        #   This would be one grammar per strict subset of expert questions
        # - Add logic to check the correctness of each partial scenic program
        # - Add logic to synthesize the full scenic program from all of the partial scenic programs

        response = requests.post(LOCAL_MODEL_ENDPOINT, json=data)
        if response.status_code != 200:
            raise ValueError(f"Local model {self._model} returned status code {response.status_code}") 
        response_body = response.json()
        content = response_body["content"]
        if verbose:
            print(f"Local Model {self._model.value}\n"
                  f"_format_message: {prompt}\n"
                  f"_predict: {content}")
        return content
