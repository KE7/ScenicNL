from enum import Enum
import json
import os

from anthropic import Anthropic, AI_PROMPT, HUMAN_PROMPT
import httpx
from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.common import LLMPromptType, ModelInput, VectorDB, format_scenic_tutorial_prompt


class AnthropicModel(Enum):
    CLAUDE_INSTANT = "claude-instant-1.2"
    CLAUDE_2 = "claude-2.0"
    

class AnthropicAdapter(ModelAdapter):
    """
    This class servers as a wrapper for the Anthropic API.
    """
    def __init__(self, model : AnthropicModel):
        super().__init__()
        self._model = model
        self.PROMPT_FILE = 'scenic_tutorial_prompt.txt' # 12%
        self.PROMPT_PATH = os.path.join(os.curdir, 'src', 'scenicNL', 'adapters', 'prompts', self.PROMPT_FILE)
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
    
    def _few_shot_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        return (
            f"{HUMAN_PROMPT} Please generate a scenic program for a CARLA "
            f"simulation to replicate the input natural language description below."
            f"\n-- Here is a scenic tutorial. --\n {format_scenic_tutorial_prompt(self.PROMPT_PATH)}\n\n"
            f"Here are some examples scenic programs. \n{model_input.examples[0]}\n{model_input.examples[1]}\n{model_input.examples[2]}\n"
            f"Given the following report, write a scenic program that models it: \n"
            f"\n\n<user_input>{model_input.nat_lang_scene_des}\n\n</user_input>"
            f"Output the original scene description as a comment at the top of the file first, "
            f"then the scenic program. Do not include any other text."
            f"\n\n{AI_PROMPT}"
        )

    def _few_shot_prompt_with_rag(
        self,
        model_input: ModelInput,
        top_k: int = 3,
    ) -> str:
        # this query might not make sense since the index is not built on descriptions
        # but rather on scenic programs so we should actually call this function
        # after the LLM does a first attempt at generating a scenic program
        # and then we can use the scenic program to query the index
        if model_input.retrieval_query is None:
            return self._few_shot_prompt(model_input=model_input)
        
        examples = self.index.query(model_input.retrieval_query, top_k=top_k)
        if examples is None:
            return self._few_shot_prompt(model_input=model_input)
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            retrieval_query=model_input.retrieval_query,
        )
        return self._few_shot_prompt(model_input=relevant_model_input)

    def _few_shot_prompt_with_hyde(
        self,
        model_input: ModelInput,
        top_k: int = 3,
    ) -> str:
        # this query might not make sense since the index is not built on descriptions
        # but rather on scenic programs so we should actually call this function
        # after the LLM does a first attempt at generating a scenic program
        # and then we can use the scenic program to query the index
        if model_input.retrieval_query is None:
            return self._few_shot_prompt(model_input=model_input)
        
        examples = self.index.query(model_input.retrieval_query, top_k=top_k)
        if examples is None:
            return self._few_shot_prompt(model_input=model_input)
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            retrieval_query=model_input.retrieval_query,
        )
        return self._few_shot_prompt(model_input=relevant_model_input)

    def _zero_shot_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        return (
            f"{HUMAN_PROMPT} Please generate a scenic program for a CARLA "
            f"simulation to replicate the input natural language description below."
            f"\n-- Here is a scenic tutorial. --\n {format_scenic_tutorial_prompt(self.PROMPT_PATH)}\n\n"
            f"Given the following report, write a scenic program that models it: \n"
            f"\n\n<user_input>{model_input.nat_lang_scene_des}\n\n</user_input>"
            f"Output the original scene description as a comment at the top of the file first, "
            f"then the scenic program. Do not include any other text."
            f"\n\n{AI_PROMPT}"
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
        if prompt_type == LLMPromptType.PREDICT_FEW_SHOT:
            msg = self._few_shot_prompt(model_input=model_input)
        elif prompt_type == LLMPromptType.PREDICT_ZERO_SHOT:
            msg = self._zero_shot_prompt(model_input=model_input)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_RAG:
            msg = self._few_shot_prompt_with_rag(model_input=model_input)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE:
            msg = self._few_shot_prompt_with_hyde(model_input=model_input)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        
        if verbose:
            print(f"Message formatted using {LLMPromptType(prompt_type).name}")
            print("Message = ", msg)

        if msg is None:
            raise NotImplementedError(f"Prompt type {prompt_type} not implemented for Anthropic.")
        
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
        # to prevent misuse of file handlers
        limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)
        with Anthropic(connection_pool_limits=limits, max_retries=10) as claude:
            if prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_RAG:
                new_model_input = ModelInput(
                    examples=model_input.examples, # this will get overwritten by the search query
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    retrieval_query=model_input.nat_lang_scene_des, # this is used for the query search
                )
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
            elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE:
                claude_response = claude.completions.create(
                    prompt=self._format_message(
                        model_input=model_input, 
                        prompt_type=LLMPromptType.PREDICT_FEW_SHOT, 
                        verbose=verbose
                    ),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                # We need to call Claude again
                new_model_input = ModelInput(
                    examples=model_input.examples, # this will get overwritten by the search query
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    retrieval_query=claude_response.completion, # this is used for the query search
                )
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
            else:
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=prompt_type, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
        
        return claude_response.completion
        