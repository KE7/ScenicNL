from enum import Enum
import json
from typing import cast

from anthropic import Anthropic, AI_PROMPT, HUMAN_PROMPT
import httpx
import os
from pathlib import Path
from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.common import DISCUSSION_TEMPERATURE, NUM_EXPERTS, LLMPromptType, ModelInput, VectorDB, few_shot_prompt_with_rag, get_expert_synthesis_prompt
from scenicNL.common import get_discussion_prompt, get_discussion_to_program_prompt, format_scenic_tutorial_prompt, get_few_shot_ast_prompt, get_tot_nl_prompt, get_discussion_to_split_prompt
import re
import scenic
import tempfile

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
            f"\n\nWrite a scenic program that models the natural language description. Provide NO additional commentary before or after. Only output the code."
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
                      f"_few_shot_reasoning_hyde: no first attempt scenic program, using original examples\n")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)

        examples = self.index.query(model_input.first_attempt_scenic_program, top_k=top_k)
        if examples is None: # if the query fails, we will use the original examples
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_reasoning_hyde: query into index for HyDE failed, using original examples\n")
        else:
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_reasoning_hyde: query into index for HyDE successful, using examples {examples}\n")
                
        examples = model_input.examples # TODO: ignore HyDE for now
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            first_attempt_scenic_program=model_input.first_attempt_scenic_program,
            expert_discussion=model_input.expert_discussion,
        )

        prompt = get_discussion_to_program_prompt()

        task_and_others = prompt.split("{natural_language_description}")
        task = task_and_others[0]
        others = task_and_others[1].split("{example_1}")
        example_1 = others[0]
        others = others[1].split("{example_2}")
        example_2 = others[0]
        others = others[1].split("{example_3}")
        example_3 = others[0]

        prompt = (
            f"{HUMAN_PROMPT}\n"
            f"{task}{model_input.nat_lang_scene_des}\n"
            f"{example_1}{relevant_model_input.examples[0]}\n"
            f"{example_2}{relevant_model_input.examples[1]}\n"
            f"{example_3}{relevant_model_input.examples[2]}\n"
            f"{AI_PROMPT}"
        )

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_few_shot_reasoning_hyde: {prompt}")

        return prompt

    def _few_shot_reasoning_nl(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> str:
        if model_input.first_attempt_scenic_program is None:
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_reasoning_split: no first attempt scenic program, using original examples\n")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)

        examples = self.index.query(model_input.first_attempt_scenic_program, top_k=top_k)
        if examples is None: # if the query fails, we will use the original examples
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_reasoning_hyde: query into index for HyDE failed, using original examples\n")
        else:
            if verbose:
                print(f"Anthropic Model {self._model.value}\n"
                      f"_few_shot_reasoning_hyde: query into index for HyDE successful, using examples {examples}\n")
                
        examples = model_input.examples # TODO: ignore HyDE for now
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            first_attempt_scenic_program=model_input.first_attempt_scenic_program,
            expert_discussion=model_input.expert_discussion,
        )

        prompt = get_discussion_to_program_prompt()

        task_and_others = prompt.split("{natural_language_description}")
        task = task_and_others[0]
        others = task_and_others[1].split("{example_1}")
        example_1 = others[0]
        others = others[1].split("{example_2}")
        example_2 = others[0]
        others = others[1].split("{example_3}")
        example_3 = others[0]

        prompt = (
            f"{HUMAN_PROMPT}\n"
            f"{task}{model_input.nat_lang_scene_des}\n"
            f"{example_1}{relevant_model_input.examples[0]}\n"
            f"{example_2}{relevant_model_input.examples[1]}\n"
            f"{example_3}{relevant_model_input.examples[2]}\n"
            f"\nPlease start your answer with the character #"
            f"{AI_PROMPT}"
        )

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_few_shot_reasoning_split: {prompt}")

        return prompt

    def _few_shot_reasoning_split(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> str:
        example_1 = model_input.examples[0]
        example_2 = model_input.examples[1]
        example_3 = model_input.examples[2]
        prompt = (
            f"{HUMAN_PROMPT}\n"
            f"For the following Scenic program, write just the parameter definition section for the input below.{model_input.nat_lang_scene_des}\n"
            f"{example_1}{model_input.examples[0]}\n"
            f"{example_2}{model_input.examples[1]}\n"
            f"{example_3}{model_input.examples[2]}\n"
            f"{AI_PROMPT}"
        )

        return prompt

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

    def _ast_feedback_prompt(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> str:
        prompt = get_few_shot_ast_prompt(model_input=model_input)

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_few_shot_ast_prompt: {prompt}")
            
        prompt = cast(str, prompt)
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



    def _format_discussion_prompt(
        self,
        model_input: ModelInput,
        verbose: bool,
    ) -> str:
        prompt = get_discussion_prompt()

        task_and_others = prompt.split("{example_1}")
        task = task_and_others[0]
        others = task_and_others[1].split("{natural_language_description}")
        nl_prompt = others[0]
        questions = others[1]

        prompt = (
            f"{HUMAN_PROMPT}\n"
            f"{task}{model_input.examples[0]}\n"
            f"{nl_prompt}{model_input.nat_lang_scene_des}\n"
            f"{questions}"
            f"{AI_PROMPT}"
        )

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_format_discussion_prompt: {prompt}")
            
        return prompt
    
    def _format_expert_synthesis_prompt(
        self,
        model_input: ModelInput,
        verbose: bool,
    ) -> str:
        prompt = get_expert_synthesis_prompt()

        task_and_others = prompt.split("{natural_language_description}")
        task = task_and_others[0]
        others = task_and_others[1].split("{expert_1}")
        expert_1 = others[0]
        others = others[1].split("{expert_2}")
        expert_2 = others[0]
        others = others[1].split("{expert_3}")
        expert_3 = others[0]

        prompt = (
            f"{HUMAN_PROMPT}\n"
            f"{task}{model_input.nat_lang_scene_des}\n"
            f"{expert_1}{model_input.panel_discussion[0]}\n"
            f"{expert_2}{model_input.panel_discussion[1]}\n"
            f"{expert_3}{model_input.panel_discussion[2]}\n"
            f"{AI_PROMPT}"
        )

        if verbose:
            print(f"Anthropic Model {self._model.value}\n"
                  f"_format_expert_synthesis_prompt: {prompt}")
            
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
            msg = self._format_discussion_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_ZERO_SHOT:
            msg = self._zero_shot_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_RAG:
            msg = self._few_shot_prompt_with_rag(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE:
            msg = self._few_shot_prompt_with_hyde(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_TOT_THEN_HYDE:
            msg = self._few_shot_reasoning_hyde(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_TOT_THEN_SPLIT:
            msg = self._few_shot_reasoning_split(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_TOT_INTO_NL:
            msg = self._few_shot_reasoning_nl(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.EXPERT_SYNTHESIS:
            msg = self._format_expert_synthesis_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.AST_FEEDBACK:
            msg = self._ast_feedback_prompt(model_input=model_input, verbose=verbose)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

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
        verbose: bool,
        max_retries: int,
        verbose_retries: bool,
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
                panel_answers = []
                for _ in range(NUM_EXPERTS):
                    claude_response = claude.completions.create(
                        prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_DISCUSSION, verbose=verbose),
                        temperature=DISCUSSION_TEMPERATURE,
                        max_tokens_to_sample=max_length_tokens,
                        model=self._model.value,
                    )
                    panel_answers.append(claude_response.completion)

                if len(panel_answers) != NUM_EXPERTS:
                    raise ValueError(f"Expected {NUM_EXPERTS} panel answers, but got {len(panel_answers)}")

                model_input = ModelInput(
                    examples=model_input.examples,
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    first_attempt_scenic_program=model_input.first_attempt_scenic_program,
                    panel_discussion=panel_answers,
                    expert_discussion=None # 
                )
                if verbose:
                    print(f"Anthropic Model {self._model.value}\n"
                          f"Tree of Thought: {panel_answers}\n")
                
                # 2. Ask an expert to synthesize the answers into a single answer
                expert_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_SYNTHESIS, verbose=verbose),
                    temperature=DISCUSSION_TEMPERATURE,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                expert_synthesis = expert_response.completion
                
                if verbose:
                    print(f"Anthropic Model {self._model.value}\n"
                          f"Expert Synthesis: {expert_synthesis}\n")

                # 3. Do a few shot predict on the natural language description
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.PREDICT_FEW_SHOT, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )

                # 4. Use the resulting program to query the index to do HyDE thus obtaining the top k programs
                # We need to call Claude again
                new_model_input = ModelInput(
                    examples=model_input.examples, # this will get overwritten by the search query
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    first_attempt_scenic_program=claude_response.completion, # this is ONLY used for the query search in HyDE/RAG
                    expert_discussion=expert_synthesis,
                    panel_discussion=panel_answers
                )

                # 5. Use the top k programs as examples for the few shot prediction along with the answer from the tree of thought
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
            elif prompt_type == LLMPromptType.PREDICT_TOT_THEN_SPLIT:
                # 1. Use tree of thought to answer all questions in the prompt
                panel_answers = []
                for _ in range(NUM_EXPERTS):
                    claude_response = claude.completions.create(
                        prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_DISCUSSION, verbose=verbose),
                        temperature=DISCUSSION_TEMPERATURE,
                        max_tokens_to_sample=max_length_tokens,
                        model=self._model.value,
                    )
                    panel_answers.append(claude_response.completion)

                if len(panel_answers) != NUM_EXPERTS:
                    raise ValueError(f"Expected {NUM_EXPERTS} panel answers, but got {len(panel_answers)}")

                model_input = ModelInput(
                    examples=model_input.examples,
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    first_attempt_scenic_program=model_input.first_attempt_scenic_program,
                    panel_discussion=panel_answers,
                    expert_discussion=None # 
                )
                if verbose:
                    print(f"Anthropic Model {self._model.value}\n"
                          f"Tree of Thought: {panel_answers}\n")
                
                # 2. Ask an expert to synthesize the answers into a single answer
                expert_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_SYNTHESIS, verbose=verbose),
                    temperature=DISCUSSION_TEMPERATURE,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                expert_synthesis = expert_response.completion
                
                if verbose:
                    print(f"Anthropic Model {self._model.value}\n"
                          f"Expert Synthesis: {expert_synthesis}\n")
                print(f"Anthropic Model {self._model.value}\n"
                      f"Expert Synthesis: {expert_synthesis}\n")
                reasoning = expert_synthesis
                print(reasoning)
                reasoning_list = reasoning.split(r'\n\d\.')
                for item in reasoning_list:
                    print('>>', item[:20])

                # self._few_shot_reasoning_split(model_input=model_input, verbose=False, top_k=3, reasoning=expert_synthesis)
                incremental_scenic_program = '# Start of scenic program'
                
                split_prompt = get_discussion_to_split_prompt()
                split_prompt = split_prompt.split('\n**\n')

                # 3. Do a few shot predict on the natural language description
                for rest_prompt in split_prompt[1:]:
                    head_prompt = split_prompt[0].format(
                        example_1=model_input.examples[0],
                        natural_language_description=model_input.nat_lang_scene_des,
                        partial_scenic_program=incremental_scenic_program)
                    combo_prompt = head_prompt + rest_prompt

                    combo_prompt = (
                        f"{HUMAN_PROMPT}\n"
                        f"{combo_prompt}"
                        f"{AI_PROMPT}"
                    )

                    claude_response = claude.completions.create(
                        prompt=combo_prompt, 
                        temperature=temperature,
                        max_tokens_to_sample=max_length_tokens,
                        model=self._model.value,
                    )
                    incremental_scenic_program += claude_response.completion

                stitch_prompt = (
                    f"{HUMAN_PROMPT}\n"
                    f"Please connect together the following program snippets by deleting any redundant lines of text. You CANNOT make any other changes. The final output will be executed directly so please do not output any leading or trailing text. Thanks honey.\n"
                    f"{incremental_scenic_program}"
                    f"{AI_PROMPT}"
                )

                claude_response = claude.completions.create(
                    prompt=stitch_prompt, 
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                print('$$$$')
                print(claude_response.completion)
                # 4. Use the resulting program to query the index to do HyDE thus obtaining the top k programs
                # We need to call Claude again
                # new_model_input = ModelInput(
                #     examples=model_input.examples, # this will get overwritten by the search query
                #     nat_lang_scene_des=model_input.nat_lang_scene_des,
                #     first_attempt_scenic_program=claude_response.completion, # this is ONLY used for the query search in HyDE/RAG
                #     expert_discussion=expert_synthesis,
                #     panel_discussion=panel_answers
                # )

                # 5. Use the top k programs as examples for the few shot prediction along with the answer from the tree of thought
                # claude_response = claude.completions.create(
                #     prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                #     temperature=temperature,
                #     max_tokens_to_sample=max_length_tokens,
                #     model=self._model.value,
                # )
            elif prompt_type == LLMPromptType.PREDICT_TOT_INTO_NL:
                # 1. Use tree of thought to answer all questions in the prompt
                panel_answers = []
                for _ in range(NUM_EXPERTS):
                    claude_response = claude.completions.create(
                        prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_DISCUSSION, verbose=verbose),
                        temperature=DISCUSSION_TEMPERATURE,
                        max_tokens_to_sample=max_length_tokens,
                        model=self._model.value,
                    )
                    panel_answers.append(claude_response.completion)

                if len(panel_answers) != NUM_EXPERTS:
                    raise ValueError(f"Expected {NUM_EXPERTS} panel answers, but got {len(panel_answers)}")

                model_input = ModelInput(
                    examples=model_input.examples,
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    first_attempt_scenic_program=model_input.first_attempt_scenic_program,
                    panel_discussion=panel_answers,
                    expert_discussion=None # 
                )
                if verbose:
                    print(f"Anthropic Model {self._model.value}\n"
                          f"Tree of Thought: {panel_answers}\n")
                
                # 2. Ask an expert to synthesize the answers into a single answer
                expert_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_SYNTHESIS, verbose=verbose),
                    temperature=DISCUSSION_TEMPERATURE,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                expert_synthesis = expert_response.completion
                
                if verbose:
                    print(f"Anthropic Model {self._model.value}\n"
                          f"Expert Synthesis: {expert_synthesis}\n")

                # 3. Create a new model input
                new_model_input = ModelInput(
                    examples=model_input.examples, # this will get overwritten by the search query
                    nat_lang_scene_des=model_input.nat_lang_scene_des,
                    first_attempt_scenic_program=claude_response.completion, # this is ONLY used for the query search in HyDE/RAG
                    expert_discussion=expert_synthesis,
                    panel_discussion=panel_answers
                )

                # 3. Add the expert_synthesis directly into the natural language description 
                # USING PREDICT_TOT_INTO_NL PROMPT - needs fixes for chattiness
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=new_model_input, prompt_type=LLMPromptType.PREDICT_TOT_INTO_NL, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )

                # 4. Final new model input
                final_model_input = ModelInput(
                    examples=model_input.examples, # this will get overwritten by the search query
                    nat_lang_scene_des=claude_response.completion,
                    # first_attempt_scenic_program=claude_response.completion, # this is ONLY used for the query search in HyDE/RAG
                    # expert_discussion=expert_synthesis,
                    # panel_discussion=panel_answers
                )
                
                # 5. Do a few shot predict on the natural language description
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=final_model_input, prompt_type=LLMPromptType.PREDICT_FEW_SHOT, verbose=verbose),
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

            model_result = str(claude_response.completion)

            with tempfile.TemporaryDirectory(dir=os.curdir) as temp_dir:
                retries = max_retries
                retries_dir = os.path.join(temp_dir, 'temp_dir')
                os.makedirs(retries_dir)

                with tempfile.NamedTemporaryFile(dir=retries_dir, delete=False, suffix='.scenic') as temp_file:
                    fname = temp_file.name

                    while retries:
                        with open(fname, 'w') as f:
                            pattern = r'\s+param map'
                            replacement = r'\nparam map'
                            model_result = re.sub(pattern, replacement, model_result)
                            f.write(model_result)
                        try:
                            # ast = scenic.syntax.parser.parse_file(fname)
                            scenario = scenic.scenarioFromFile(fname, mode2D=True)
                            if verbose_retries: print('No error!')
                            retries = 0 # If this statement is reached program worked -> terminates loop
                        except Exception as e:
                            if verbose_retries: print(f'Retrying... {retries}')
                            try:
                                error_message = f"Error details below..\nmessage: {str(e)}\ntext: {e.text}\nlineno: {e.lineno}\nend_lineno: {e.end_lineno}\noffset: {e.offset}\nend_offset: {e.end_offset}"
                                if verbose_retries: print(error_message)
                            except:
                                error_message = f'Error details below..\nmessage: {str(e)}'
                                if verbose_retries: print(error_message)

                            # Constructing correcting claude call
                            new_model_input = ModelInput(
                                examples=model_input.examples, # this will get overwritten by the search query
                                nat_lang_scene_des=model_input.nat_lang_scene_des,
                                first_attempt_scenic_program=str(model_result),
                                compiler_error=error_message
                            )

                            claude_response = claude.completions.create(
                                prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                                temperature=temperature,
                                max_tokens_to_sample=max_length_tokens,
                                model=self._model.value,
                            )
                            model_result = str(claude_response.completion)
                            retries -= 1
        if verbose_retries: print(model_result)
        return model_result
        