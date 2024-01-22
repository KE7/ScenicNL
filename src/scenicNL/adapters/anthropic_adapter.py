from enum import Enum
import json
from typing import cast

from anthropic import Anthropic, AI_PROMPT, HUMAN_PROMPT
import httpx
from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.common import DISCUSSION_TEMPERATURE, LLMPromptType, ModelInput, VectorDB, few_shot_prompt_with_rag
from scenicNL.common import format_discussion_prompt, format_discussion_to_program_prompt, format_scenic_tutorial_prompt


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
        model_input: ModelInput,
        verbose: bool,
    ) -> str:
        prompt = (
            f"{HUMAN_PROMPT} Here are some examples scenic programs. \n{model_input.examples[0]}\n{model_input.examples[1]}\n{model_input.examples[2]}\n"
            f"{format_scenic_tutorial_prompt()}\n\n"
            f"Here is the natural language description from the user: \n"
            f"\n\n<user_input>{model_input.nat_lang_scene_des}\n\n</user_input>"
            f"\n\n{AI_PROMPT}"
        )

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_few_shot_prompt: {prompt}")
            
        return prompt
    
    
    def _few_shot_prompt_with_rag(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> str:
        prompt = few_shot_prompt_with_rag(
            vector_index=self.index,
            model_input=model_input,
            few_shot_prompt_generator=self._few_shot_prompt,
            top_k=top_k,
        )

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_few_shot_prompt_with_rag: {prompt}")
            
        prompt = cast(str, prompt)
        return prompt
        

    def _few_shot_reasoning_hyde(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> str:
        if model_input.first_attempt_scenic_program is None:
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_reasoning_hyde: no first attempt scenic program, using original examples")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)

        examples = self.index.query(model_input.first_attempt_scenic_program, top_k=top_k)
        if examples is None: # if the query fails, we will use the original examples
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_reasoning_hyde: query into index for HyDE failed, using original examples")
            examples = model_input.examples
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            first_attempt_scenic_program=model_input.first_attempt_scenic_program,
            expert_discussion=model_input.expert_discussion,
        )

        prompt = format_discussion_to_program_prompt(model_input=relevant_model_input)

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_few_shot_reasoning_hyde: {prompt}")

        return (
            f"{HUMAN_PROMPT}\n"
            f"{prompt}\n"
            f"{AI_PROMPT}"
        )

    
    def _few_shot_prompt_with_hyde(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> str:
        # this query might not make sense since the index is not built on descriptions
        # but rather on scenic programs so we should actually call this function
        # after the LLM does a first attempt at generating a scenic program
        # and then we can use the scenic program to query the index
        if model_input.first_attempt_scenic_program is None:
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_prompt_with_hyde: no first attempt scenic program, using original examples")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)
        
        examples = self.index.query(model_input.first_attempt_scenic_program, top_k=top_k)
        if examples is None: # if the query fails, we just return the few shot prompt
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_prompt_with_hyde: query into index for HyDE failed, using original examples")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            first_attempt_scenic_program=model_input.first_attempt_scenic_program,
        )
        prompt = self._few_shot_prompt(model_input=relevant_model_input, verbose=verbose)

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_few_shot_prompt_with_hyde: {prompt}")
            
        return prompt

    def _zero_shot_prompt(
        self,
        model_input: ModelInput,
        verbose: bool,
    ) -> str:
        prompt = (
            f"{HUMAN_PROMPT} Please generate a scenic program for a CARLA "
            f"simulation to replicate the input natural language description below."
            f"\n-- Here is a scenic tutorial. --\n {format_scenic_tutorial_prompt()}\n\n"
            f"Given the following report, write a scenic program that models it: \n"
            f"\n\n<user_input>{model_input.nat_lang_scene_des}\n\n</user_input>"
            f"Output the original scene description as a comment at the top of the file first, "
            f"then the scenic program. Do not include any other text."
            f"{AI_PROMPT}"
        )

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_zero_shot_prompt: {prompt}")
            
        return prompt

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
            msg = self._few_shot_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.EXPERT_DISCUSSION:
            msg = f"{HUMAN_PROMPT}\n{format_discussion_prompt(model_input=model_input, verbose=verbose)}\n{AI_PROMPT}"
        elif prompt_type == LLMPromptType.PREDICT_ZERO_SHOT:
            msg = self._zero_shot_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_RAG:
            msg = self._few_shot_prompt_with_rag(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE:
            msg = self._few_shot_prompt_with_hyde(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_TOT_THEN_HYDE:
            msg = self._few_shot_reasoning_hyde(model_input=model_input, verbose=verbose)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        if msg is None:
            raise NotImplementedError(f"Prompt type {prompt_type} was not formated for Anthropic model {self._model.value}")
        
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
            if prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE or prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE_TOT:
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.PREDICT_FEW_SHOT, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                # We need to call Claude again
                new_model_input = ModelInput(
                    examples=model_input.examples, # this will get overwritten by the search query
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    first_attempt_scenic_program=claude_response.completion, # this is used for the query search
                )
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
            elif prompt_type == LLMPromptType.PREDICT_TOT_THEN_HYDE:
                # 1. Use tree of thought to answer all questions in the prompt
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_DISCUSSION, verbose=verbose),
                    temperature=DISCUSSION_TEMPERATURE,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                discussion = claude_response.completion
                if verbose:
                    print(f"Anthropic Model {self._model.value}\n"
                          f"Expert Discussion: {discussion}")

                # 2. Do a few shot predict on the natural language description
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.PREDICT_FEW_SHOT, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )

                # 3. Use the resulting program to query the index to do HyDE thus obtaining the top k programs
                # We need to call Claude again
                new_model_input = ModelInput(
                    examples=[], # this will get overwritten by the search query
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    first_attempt_scenic_program=claude_response.completion, # this is ONLY used for the query search in HyDE/RAG
                    expert_discussion=discussion,
                )

                # 4. Use the top k programs as examples for the few shot prediction along with the answer from the tree of thought
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
        