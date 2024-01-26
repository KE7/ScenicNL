from scenicNL.adapters.api_adapter import Scenic3
from scenicNL.adapters.model_adapter import ModelAdapter

import json
from typing import Dict, List, cast
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import os

import openai

from scenicNL.common import DISCUSSION_TEMPERATURE, LLMPromptType, ModelInput, PromptFiles, VectorDB, few_shot_prompt_with_rag, get_discussion_prompt, get_discussion_to_program_prompt, get_expert_synthesis_prompt, remove_llm_prose


class OpenAIModel(Enum):
    GPT_35_TURBO = "gpt-3.5-turbo-0613"
    GPT_4 = "gpt-4-0613"
    GPT_4_TURBO = "gpt-4-1106-preview"
    GPT_4_32K = "gpt-4-32k-0613"


class OpenAIAdapter(ModelAdapter):
    """
    This class servers as a wrapper for the OpenAI API.
    """
    def __init__(self, model: OpenAIModel):
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_ORGANIZATION") and len(os.getenv("OPENAI_ORGANIZATION")) > 0:
            openai.organization = os.getenv("OPENAI_ORGANIZATION")
        self.model = model
        self.index = VectorDB(index_name='scenic-programs')

    def _zero_shot_prompt(
        self,
        model_input: ModelInput,
        verbose: bool
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
        model_input: ModelInput,
        verbose: bool
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
            {
                "role": "system", "content": self._format_reasoning_prompt(model_input)
            }
        ]

    def _few_shot_tot_prompt(
        self,
        model_input: ModelInput,
        verbose: bool
    ) -> list[Dict[str, str]]:
        """
        Format the message for the OpenAI API for few shot prediction using Tree of Thought.
        """
        return [
            {"role": "system", "content": "Please generate a scenic program for a CARLA " +
             "simulation based on the input natural language description below." + 
             "\n-- Here is a scenic tutorial. --\n" + self._format_scenic_tot_tutorial_prompt(model_input) +
             "\n-- Here are some example scenic programs. --\n" + model_input.examples[0] 
             + model_input.examples[1] + model_input.examples[2]},
            {"role": "user", "content": model_input.nat_lang_scene_des},
        ]

    def _scenic_tutorial_prompt(
        self,
        model_input: ModelInput,
        verbose: bool
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
        model_input: ModelInput,
        verbose: bool
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
        model_input: ModelInput,
        verbose: bool
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
    
    def _few_shot_prompt_with_rag(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        prompt = few_shot_prompt_with_rag(
            vector_index=self.index,
            model_input=model_input,
            few_shot_prompt_generator=self._few_shot_prompt,
            top_k=top_k,
        )
        prompt = cast(List[Dict[str, str]], prompt)
        if verbose:
            print(f"Few shot prompt with RAG: {prompt}")
        return prompt
    

    def _few_shot_prompt_with_hyde(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> List[Dict[str, str]]:
        # this query might not make sense since the index is not built on descriptions
        # but rather on scenic programs so we should actually call this function
        # after the LLM does a first attempt at generating a scenic program
        # and then we can use the scenic program to query the index
        if model_input.first_attempt_scenic_program is None:
            if verbose:
                print("No first attempt scenic program found. Using few shot prompt.")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)
        
        examples = self.index.query(model_input.first_attempt_scenic_program, top_k=top_k)
        if examples is None:
            if verbose:
                print("Index query failed. Using original few shot prompt.")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            first_attempt_scenic_program=model_input.first_attempt_scenic_program,
        )
        prompt = self._few_shot_prompt(model_input=relevant_model_input, verbose=verbose)
        if verbose:
            print(f"Few shot prompt with HyDE: {prompt}")
        return prompt
    

    def _few_shot_reasoning_hyde(
        self,
        model_input: ModelInput,
        verbose: bool,
        top_k: int = 3,
    ) -> List[Dict[str, str]]:
        if model_input.first_attempt_scenic_program is None:
            if verbose:
                print("No first attempt scenic program found. Using few shot prompt.")
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)
        
        examples = self.index.query(model_input.first_attempt_scenic_program, top_k=top_k)
        if examples is None: # if the query fails, we will use the original examples
            if verbose:
                print("Index query failed. Using original few shot prompt.")
        else:
            if verbose:
                print(f"Index query successful. Using the following examples: {examples}")
        
        examples = model_input.examples # TODO: skip HyDE for now
        
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

        prompt = [
            {"role": "system", "content": task},
            {"role": "user", "content": model_input.nat_lang_scene_des},
            {"role": "system", "content": example_1},
            {"role": "user", "content": relevant_model_input.examples[0]},
            {"role": "system", "content": example_2},
            {"role": "user", "content": relevant_model_input.examples[1]},
            {"role": "system", "content": example_3},
            {"role": "user", "content": relevant_model_input.examples[2]},
        ]

        if verbose:
            print(f"Few shot prompt with ToT and HyDE: {prompt}")
        
        return prompt
    

    def _format_discussion_prompt(
        self,
        model_input: ModelInput,
        verbose: bool,
    ) -> List[Dict[str, str]]:
        prompt = get_discussion_prompt()

        task_and_others = prompt.split("{example_1}")
        task = task_and_others[0]
        others = task_and_others[1].split("{natural_language_description}")
        nl_prompt = others[0]
        questions = others[1]

        prompt = [
            {"role": "system", "content": task},
            {"role": "user", "content": model_input.examples[0]},
            {"role": "system", "content": nl_prompt},
            {"role": "user", "content": model_input.nat_lang_scene_des},
            {"role": "system", "content": questions},
        ]  

        if verbose:
            print(f"Expert discussion prompt: {prompt}")
        return prompt
    

    def _format_expert_synthesis_prompt(
        self,
        model_input: ModelInput,
        verbose: bool,
    ) -> List[Dict[str, str]]:
        prompt = get_expert_synthesis_prompt()

        task_and_others = prompt.split("{natural_language_description}")
        task = task_and_others[0]
        others = task_and_others[1].split("{expert_1}")
        expert_1 = others[0]
        others = others[1].split("{expert_2}")
        expert_2 = others[0]
        others = others[1].split("{expert_3}")
        expert_3 = others[0]

        prompt = [
            {"role": "system", "content": task},
            {"role": "user", "content": model_input.nat_lang_scene_des},
            {"role": "system", "content": expert_1},
            {"role": "user", "content": model_input.panel_discussion[0]},
            {"role": "system", "content": expert_2},
            {"role": "user", "content": model_input.panel_discussion[1]},
            {"role": "system", "content": expert_3},
            {"role": "user", "content": model_input.panel_discussion[2]},
        ]

        if verbose:
            print(f"Expert synthesis prompt: {prompt}")
        
        return prompt


    def _format_message(
        self,
        *,
        model_input: ModelInput,
        prompt_type: LLMPromptType,
        verbose: bool
    ) -> List[Dict[str, str]]:
        """
        Format the message for the OpenAI API.
        """
        if prompt_type == LLMPromptType.PREDICT_ZERO_SHOT:
            return self._zero_shot_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT:
            return self._few_shot_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_SCENIC_TUTORIAL:
            return self._scenic_tutorial_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_PYTHON_API:
            return self._python_api_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_PYTHON_API_ONELINE: # for one-line corrections of function calling
            return self._python_api_prompt_oneline(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_RAG:
            return self._few_shot_prompt_with_rag(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE:
            return self._few_shot_prompt_with_hyde(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE_TOT:
            return self._few_shot_tot_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.PREDICT_TOT_THEN_HYDE:
            return self._few_shot_reasoning_hyde(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.EXPERT_DISCUSSION:
            return self._format_discussion_prompt(model_input=model_input, verbose=verbose)
        elif prompt_type == LLMPromptType.EXPERT_SYNTHESIS:
            return self._format_expert_synthesis_prompt(model_input=model_input, verbose=verbose)
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
                "message": self._format_message(model_input=model_input, prompt_type=prompt_type, verbose=False),
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
        verbose: bool,
    ) -> str:
        if prompt_type == LLMPromptType.PREDICT_TOT_THEN_HYDE:
            # 1. Use tree of thought to answer all questions in the prompt
            response = openai.ChatCompletion.create(
                temperature=DISCUSSION_TEMPERATURE,
                model=self.model.value,
                max_tokens=max_length_tokens,
                messages=self._format_message(model_input=model_input, prompt_type=LLMPromptType.EXPERT_DISCUSSION, verbose=verbose),
            )
            panel_answer = response.choices[0].message.content
            
            model_input = ModelInput(
                examples=model_input.examples, 
                nat_lang_scene_des=model_input.nat_lang_scene_des,
                first_attempt_scenic_program=model_input.first_attempt_scenic_program,
                panel_discussion=panel_answer,
                expert_discussion=None
            )
            
            if verbose:
                print(f"Tree of thought:\n{panel_answer}\n")

            # 2. Do a few shot predict on the natural language description
            response = openai.ChatCompletion.create(
                temperature=temperature,
                model=self.model.value,
                max_tokens=max_length_tokens,
                messages=self._format_message(model_input=model_input, prompt_type=LLMPromptType.PREDICT_FEW_SHOT, verbose=verbose),
            )

            # 3. Use the resulting program to query the index to do HyDE thus obtaining the top k programs
            new_model_input = ModelInput(
                examples=model_input.examples, # this will get overwritten by the search query
                nat_lang_scene_des=model_input.nat_lang_scene_des,
                first_attempt_scenic_program=response.choices[0].message.content,
                expert_discussion=panel_answer,
                panel_discussion=panel_answer,
            )

            # 4. Use the top k programs as examples for the few shot prediction along with the answer from the tree of thought
            response = openai.ChatCompletion.create(
                temperature=temperature,
                model=self.model.value,
                max_tokens=max_length_tokens,
                messages=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose),
            )

        elif prompt_type != LLMPromptType.PREDICT_FEW_SHOT_WITH_HYDE:
            messages = self._format_message(model_input=model_input, prompt_type=prompt_type, verbose=verbose)
            response = openai.ChatCompletion.create(
                temperature=temperature,
                model=self.model.value,
                max_tokens=max_length_tokens,
                messages=messages
            )

        else: # HyDE
            response = openai.ChatCompletion.create(
                temperature=temperature,
                model=self.model.value,
                max_tokens=max_length_tokens,
                messages=self._format_message(model_input=model_input, prompt_type=LLMPromptType.PREDICT_FEW_SHOT, verbose=verbose)
            )
            # We need to call GPT again
            new_model_input = ModelInput(
                examples=model_input.examples, # this will get overwritten by the search query
                nat_lang_scene_des=model_input.nat_lang_scene_des,
                first_attempt_scenic_program=response.choices[0].message.content,
            )
            response = openai.ChatCompletion.create(
                temperature=temperature,
                model=self.model.value,
                max_tokens=max_length_tokens,
                messages=self._format_message(model_input=new_model_input, prompt_type=prompt_type, verbose=verbose)
            )

        # Before we return the completion, let's clean out any "helpful LLM comments"
        scenic_program = remove_llm_prose(response.choices[0].message.content, verbose)
        # TODO: Check for compiler errors

        return scenic_program

    def _format_scenic_tutorial_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        """
        Formats the message providing introduction to Scenic language and syntax.
        """
        st_prompt = ''
        with open(PromptFiles.SCENIC_TUTORIAL.value) as f:
            st_prompt = f.read()
        return st_prompt
    
    def _format_reasoning_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        """
        Formats the message providing introduction to Scenic language and syntax.
        """
        st_prompt = ''
        with open(PromptFiles.QUESTION_REASONING.value) as f:
            st_prompt = f.read()
            return st_prompt.format(
                example_1=model_input.examples[0],
                example_2=model_input.examples[1],
                example_3=model_input.examples[2],
                natural_language_description=model_input.nat_lang_scene_des
            )

    def _format_python_api_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        pa_prompt = ''
        with open(PromptFiles.PYTHON_API.value) as f:
            pa_prompt = f.read()
        return pa_prompt

    def _format_scenic_tot_tutorial_prompt(
        self,
        model_input: ModelInput
    ) -> str:
        """
        Formats the message providing introduction to Scenic language and syntax.
        """
        st_prompt = ''
        with open(PromptFiles.DYNAMIC_SCENARIOS.value) as f:
            st_prompt = f.read()
        return st_prompt

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
                new_input = ModelInput(
                    examples=model_input.examples, 
                    nat_lang_scene_des=line)
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
            max_length_tokens=80,
            prompt_type=LLMPromptType.PREDICT_PYTHON_API_ONELINE, # @TODO: timeout 
            verbose=False,
        )
        return prediction