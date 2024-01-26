from enum import Enum
import json
from typing import cast

from anthropic import Anthropic, AI_PROMPT, HUMAN_PROMPT
import httpx
import os
from scenicNL.adapters.model_adapter import ModelAdapter
from scenicNL.common import DISCUSSION_TEMPERATURE, NUM_EXPERTS, LLMPromptType, ModelInput, VectorDB, few_shot_prompt_with_rag, get_expert_synthesis_prompt
from scenicNL.common import get_discussion_prompt, get_discussion_to_program_prompt, format_scenic_tutorial_prompt, get_few_shot_ast_prompt
import scenic

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

    def _few_shot_ast_prompt_(
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
        elif prompt_type == LLMPromptType.EXPERT_SYNTHESIS:
            msg = self._format_expert_synthesis_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_AST:
            msg = self._few_shot_ast_prompt_(model_input=model_input, verbose=verbose)
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
            elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_AST:
                # Start with a standard few shot
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=LLMPromptType.PREDICT_FEW_SHOT, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )
                # Up to {retries} retries - depending on compiler feedback
                if False:
                    retries = 0
                    while retries:
                        # print('\n\n\n^^^^^^^^\n\n\n')
                        # print(claude_response.completion)
                        # print('\n\n\n^^^^^^^^\n\n\n')
                        with open('_temp.txt', 'w') as f:
                            f.write(claude_response.completion)
                        try:
                            scenic.syntax.parser.parse_file('_temp.txt')
                            print('No error!')
                            retries = 0 # If this statement is reached program worked -> terminates loop
                        except Exception as e:
                            print(f'Retrying... {retries}')
                            error_message = f"Error details below..\nmessage: {str(e)}\ntext: {e.text}\nlineno: {e.lineno}\nend_lineno: {e.end_lineno}\noffset: {e.offset}\nend_offset: {e.end_offset}"
                            print(error_message)

                            # Constructing correcting claude call
                            new_model_input = ModelInput(
                                examples=model_input.examples, # this will get overwritten by the search query
                                nat_lang_scene_des=model_input.nat_lang_scene_des,
                                first_attempt_scenic_program=str(claude_response.completion),
                                compiler_error=error_message
                            )
                            # Call claude with few_shot_ast function call type
                            # print(f'\n\n\n%%%%%%%%')
                            # print(self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose))
                            # print(f'%%%%%%%%\n\n\n')
                            claude_response = claude.completions.create(
                                prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                                temperature=temperature,
                                max_tokens_to_sample=max_length_tokens,
                                model=self._model.value,
                            )
                            retries -= 1
                    print(claude_response.completion)
            else:
                claude_response = claude.completions.create(
                    prompt=self._format_message(model_input=model_input, prompt_type=prompt_type, verbose=verbose),
                    temperature=temperature,
                    max_tokens_to_sample=max_length_tokens,
                    model=self._model.value,
                )

            retries = 6
            while retries:
                # print('\n\n\n^^^^^^^^\n\n\n')
                # print(claude_response.completion)
                # print('\n\n\n^^^^^^^^\n\n\n')
                with open('_temp.txt', 'w') as f:
                    f.write(claude_response.completion)
                try:
                    scenic.syntax.parser.parse_file('_temp.txt')
                    print('No error!')
                    retries = 0 # If this statement is reached program worked -> terminates loop
                except Exception as e:
                    print(f'Retrying... {retries}')
                    error_message = f"Error details below..\nmessage: {str(e)}\ntext: {e.text}\nlineno: {e.lineno}\nend_lineno: {e.end_lineno}\noffset: {e.offset}\nend_offset: {e.end_offset}"
                    print(error_message)

                    # Constructing correcting claude call
                    new_model_input = ModelInput(
                        examples=model_input.examples, # this will get overwritten by the search query
                        nat_lang_scene_des=model_input.nat_lang_scene_des,
                        first_attempt_scenic_program=str(claude_response.completion),
                        compiler_error=error_message
                    )
                    # Call claude with few_shot_ast function call type
                    # print(f'\n\n\n%%%%%%%%')
                    # print(self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose))
                    # print(f'%%%%%%%%\n\n\n')
                    claude_response = claude.completions.create(
                        prompt=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
                        temperature=temperature,
                        max_tokens_to_sample=max_length_tokens,
                        model=self._model.value,
                    )
                    retries -= 1
            # os.remove('_temp.txt')
        print(claude_response.completion)
        return claude_response.completion
        