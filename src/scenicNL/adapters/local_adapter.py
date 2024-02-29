from enum import Enum
import json

import requests
from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.common import LLMPromptType, ModelInput, VectorDB


class LocalModel(Enum):
    MIXTRAL = "mixtral-8x7b"
    

class LocalAdapter(ModelAdapter):
    ENDPOINT = "http://127.0.0.1:8080/completion"
    DEFAULT_PARAMS = {
        "cache_prompt": False,
        "image_data": [],
        "mirostat": 0,
        "mirostat_eta": 0.1,
        "mirostat_tau": 5,
        "n_predict": -1,
        "n_probs": 0,
        "presence_penalty": 0,
        "repeat_last_n": 241,
        "repeat_penalty": 1.18,
        "slot_id": 0,
        "stop": ["Question:", "Answer:"],
        #"stream": False,
        "tfs_z": 1,
        "top_k": 40,
        "top_p": 0.5,
        "typical_p": 1,
    }

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

        if msg is None:
            raise NotImplementedError(f"Prompt type {prompt_type} was not formatted for Anthropic model {self._model.value}")
        
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
        prompt = self._format_message(
            model_input=model_input,
            prompt_type=prompt_type,
            verbose=verbose,
        )

        data = {"prompt": prompt, "temperature": temperature} | self.DEFAULT_PARAMS
        if max_length_tokens > 0:
            data["max_tokens"] = max_length_tokens

        # TODO:
        # - Add logic for grammars then add the grammar to the request as: "grammar": grammar
        #   This would be one grammar per strict subset of expert questions
        # - Add logic to check the correctness of each partial scenic program
        # - Add logic to synthesize the full scenic program from all of the partial scenic programs

        response = requests.post(self.ENDPOINT, json=data)
        if response.status_code != 200:
            raise ValueError(f"Local model {self._model} returned status code {response.status_code}") 
        response_body = response.json()
        content = response_body["content"]
        if verbose:
            print(f"Local Model {self._model.value}\n"
                  f"_format_message: {prompt}\n"
                  f"_predict: {content}")
        return content
