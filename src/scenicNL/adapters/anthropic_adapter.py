from enum import Enum
import json

from anthropic import Anthropic, AI_PROMPT, HUMAN_PROMPT
import httpx
import os
from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.common import LLMPromptType, ModelInput


class AnthropicModel(Enum):
    CLAUDE_INSTANT = "claude-instant-1.2"
    CLAUDE_2 = "claude-2.0"
    

class AnthropicAdapter(ModelAdapter):
    """
    This class servers as a wrapper for the Anthropic API.
    """
    def __init__(self, model : AnthropicModel):
        super().__init__()
        self._model = model #.value
        self.PROMPT_PATH = os.path.join(os.curdir, 'src', 'scenicNL', 'adapters', 'prompts')

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
                "message": self._format_message(model_input=model_input, prompt_type=prompt_type),
            },
            sort_keys=True,
        )
    
    def _few_shot_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        return (
            f"{HUMAN_PROMPT} Please generate a scenic program for a CARLA simulation based on the input natural language description below.\n"
            f"-- Here is a scenic tutorial. --\n{self._format_scenic_tutorial_prompt(model_input)}\n\n"
            f"-- Here are some example scenic programs. --\n{model_input.examples[0]}\n{model_input.examples[1]}\n{model_input.examples[2]}\n"
            f"\n\n<user_input>{model_input.nat_lang_scene_des}</user_input>"
            f"\n\n{AI_PROMPT}"
        )

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

    def _format_message(
        self,
        *,
        model_input: ModelInput,
        prompt_type: LLMPromptType
    ) -> str:
        """
        Formats the message to be sent to the API.
        """
        if prompt_type == LLMPromptType.PREDICT_FEW_SHOT:
            return self._few_shot_prompt(model_input=model_input)
        else:
            raise NotImplementedError(f"Prompt type {prompt_type} not implemented for Anthropic.")

    def _predict(
        self, 
        *, 
        model_input: ModelInput, 
        temperature: float, 
        max_length_tokens: int, 
        prompt_type: LLMPromptType
    ) -> str:
        # to prevent misuse of file handlers
        limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)
        with Anthropic(connection_pool_limits=limits, max_retries=10) as claude:
            claude_response = claude.completions.create(
                prompt=self._format_message(model_input=model_input, prompt_type=prompt_type),
                temperature=temperature,
                max_tokens_to_sample=max_length_tokens,
                model=self._model
            )
        
        return claude_response.completion
