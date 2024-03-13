

from typing import Dict, List
import httpx
from openai import OpenAI
import os
import requests

import yaml
import scenic
from scenicNL.adapters.anthropic_adapter import AnthropicAdapter, AnthropicModel
from scenicNL.adapters.openai_adapter import OpenAIAdapter, OpenAIModel
from scenicNL.common import LOCAL_MODEL_DEFAULT_PARAMS, LOCAL_MODEL_ENDPOINT, ModelInput, PromptFiles
import tempfile

class CompositionalScenic():

    DEFAULT_PARAMS = {
    "cache_prompt": True,
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
    "temperature": 0.7,
    "tfs_z": 1,
    "top_k": 40,
    "top_p": 0.9,
    "typical_p": 1,
}

    def __init__(self):
        super().__init__()
        self.mixtral = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="sk-no-key-required"
        )
        self.coder = OpenAI(
            base_url="http://localhost:8079/v1",
            api_key="sk-no-key-required"
        )
        self.anthropic = AnthropicAdapter(model=AnthropicModel.CLAUDE_2, use_index=False)
        self.gpt = OpenAIAdapter(model=OpenAIModel.GPT_35_TURBO_16k, use_index=False)


    def _query_and_parse_for_final_answer(
        self,
        messages: List[Dict],
        temperature: float,
        question: str,
        grammar: str,
        coding: bool,
        verbose: bool,
    ) -> str:
        response = None
        if coding:
            # response = self.coder.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages,
            #     temperature=temperature,
            #     grammar=grammar,
            # )
            response = httpx.post(
                "http://localhost:8079/v1/chat/completions",
                json={
                    "model": "wizardCoder33B",
                    "messages": messages,
                    "temperature": temperature,
                    # "grammar": grammar,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0,
                },
                timeout=1000
            ).json()
        else:
            # response = self.mixtral.chat.completions.create(
            #     model="LLama_CPP",
            #     messages=messages,
            #     temperature=temperature,
            #     grammar=grammar,
            # )
            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                json={
                    "model": "mixtral-8x7b",
                    "messages": messages,
                    "temperature": temperature,
                    "grammar": grammar,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0,
                },
                timeout=1000
            ).json()
        answer = response['choices'][0]['message']['content']
        if verbose:
            print(f"--- BEGIN: Question {question} Response ---")
            print(answer)
            print(f"--- END: Question {question} Response ---")

        # the answer is provided as 3 experts and 1 final where each of them
        # says JUSTIFICATION and FINAL_ANSWER
        # we just want the FINAL_ANSWER of the final
        answer = answer.split("FINAL_ANSWER:")[-1]

        return answer
    

    def validate_with_feedback(self, program: str) -> None:
        for _ in range(3):
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(program)
                temp_file_path = f.name
            try:
                scenic.scenarioFromFile(temp_file_path, mode2D=True)
                os.remove(temp_file_path)
                return program
            except Exception as e:
                feedback = str(e)
                prompt = f"The following Scenic program is invalid: {program}\n\n"
                prompt += f"Given the following compiler feedback: {feedback}\n\n"
                prompt += "Please correct the program to make it valid. Do not include any other text."
                prompt += "Your response will be taken as your corrected program."
                new_program = openai.ChatCompletion.create(
                    model="LLama_CPP",
                    messages=prompt,
                )
                new_program = new_program.choices[0].message.content
                program = new_program

                         


    def _step_one(
        self,
        model_input: ModelInput,
        temperature: float,
        system_message: dict,
        questions: dict,
        verbose: bool,
    ) -> str:
        question_num = "one"
        messages = self._get_messages(questions, question_num)
        messages[-1]["content"] = messages[-1]["content"].format(
            description=model_input.nat_lang_scene_des
        )
        # TODO: re-enable gpt
        answer = self.gpt.predict(messages=messages)
        # after a few tests, it seems that GPT_35_TURBO does much better than mixtral here
        # mixtral_answer = httpx.post(
        #     "http://localhost:8080/v1/chat/completions",
        #     json={
        #         "model": "mixtral-8x7b",
        #         "messages": messages,
        #         "temperature": temperature,
        #         "max_tokens": 1000,
        #         "top_p": 1,
        #         "frequency_penalty": 0.1,
        #         "presence_penalty": 0,
        #         "grammar": questions["single_answer_grammar"],
        #     },
        #     timeout=1000
        # ).json()
        # mixtral_answer = mixtral_answer['choices'][0]['message']['content']
        if verbose:
            print(f"--- BEGIN: Question {question_num} Response ---")
            print("GPT answer:")
            print(answer)
            # print("\nMixTral answer:")
            # print(mixtral_answer)
            print(f"--- END: Question {question_num} Response ---")

        answer = answer.split("FINAL_ANSWER:")[-1].strip()
        # mixtral_answer = mixtral_answer.split("FINAL_ANSWER:")[-1].strip()

        return answer
    

    def _get_messages(
        self,
        questions: dict,
        get_objs_q_num: str,
        num_examples: int = 3,
    ) -> List[Dict]:
        user_message = {
            "role": "user",
            "content": questions[get_objs_q_num]["user_question"],
        }
        system_msg = {
            "role": "system",
            "content": questions[get_objs_q_num]["system_question"],
        }
        example_question_1 = {
            "role": "user",
            "content": questions[get_objs_q_num]["user_question_1"]
        }
        assistant_answer_1 = {
            "role": "assistant",
            "content": questions[get_objs_q_num]["assistant_answer_1"]
        }
        if num_examples == 1:
            messages = [
                system_msg,
                example_question_1,
                assistant_answer_1,
                user_message
            ]
            return messages
        example_question_2 = {
            "role": "user",
            "content": questions[get_objs_q_num]["user_question_2"]
        }
        assistant_answer_2 = {
            "role": "assistant",
            "content": questions[get_objs_q_num]["assistant_answer_2"]
        }
        if num_examples == 2:
            messages = [
                system_msg,
                example_question_1,
                assistant_answer_1,
                example_question_2,
                assistant_answer_2,
                user_message
            ]
            return messages
        example_question_3 = {
            "role": "user",
            "content": questions[get_objs_q_num]["user_question_3"]
        }
        assistant_answer_3 = {
            "role": "assistant",
            "content": questions[get_objs_q_num]["assistant_answer_3"]
        }
        messages = [
            system_msg,
            example_question_1,
            assistant_answer_1,
            example_question_2,
            assistant_answer_2,
            example_question_3,
            assistant_answer_3,
            user_message
        ]

        return messages
    

    def _build_constants(
        self,
        model_input: ModelInput,
        temperature: float,
        system_message: dict,
        objects: str,
        questions: dict,
        verbose: bool,
    ) -> str:
        # get_objs_q_num = "nine"
        # single_question_grammar = questions["single_answer_grammar"]
        # messages = self._get_messages(questions, get_objs_q_num)
        # messages[-1]["content"] = messages[-1]["content"].format(
        #     description=model_input.nat_lang_scene_des,
        #     objects_from_one=objects
        # )
        # response = self.gpt.predict(messages=messages)
        # # local_response = self._query_and_parse_for_final_answer(
        # #     messages=messages,
        # #     temperature=temperature,
        # #     question=get_objs_q_num,
        # #     grammar=single_question_grammar,
        # #     coding=True,
        # #     verbose=verbose,
        # # )
        # objects = response.split("FINAL_ANSWER:")[-1].strip()
        # other_objects = local_response.split("FINAL_ANSWER:")[-1].strip()
        
        # now in case there are any lists of objects, we need to convert them to probability distributions
        get_objs_q_num = "nine_b"
        messages = self._get_messages(questions, get_objs_q_num)
        messages[-1]["content"] = messages[-1]["content"].format(
            description=model_input.nat_lang_scene_des,
            program=objects
        )
        
        response = self.gpt.predict(messages=messages)

        current_program = f"# Scenic Program for the description:\n\"\"\"{model_input.nat_lang_scene_des}\"\"\"\n\n\n"
        current_program += f"{objects}\n\n\n"

        """
        objects_list = objects.split("\n")
        objects_list = map(lambda x: x.strip(), objects_list)
        objects_list = [obj for obj in objects_list if obj != ""]
        program_objects = ""
        for object in objects_list:
            object = object.strip()
            user_message = {
                "role": "user",
                "content": questions[get_objs_q_num]["question"].format(objects_from_one=object),
            }
            single_question_grammar = questions["single_answer_grammar"]
            messages = [user_message]
            program_object = self._query_and_parse_for_final_answer(
                messages=messages,
                temperature=temperature,
                verbose=verbose,
                question=get_objs_q_num,
                grammar=single_question_grammar,
            )
            program_object = program_object.strip()
            program_objects = program_objects + program_object + "\n"

        self.validate_with_feedback(current_program)
        """

        # Question 4 b
        get_objs_q_num = "four_b"
        messages = self._get_messages(questions, get_objs_q_num, num_examples=2)
        messages[-1]["content"] = messages[-1]["content"].format(
            description=model_input.nat_lang_scene_des,
            objects_from_nine=objects
        )
        missing_objects = self.gpt.predict(messages=messages)

        # Question 5
        generic_dists_q_num = "five"
        messages = self._get_messages(questions, generic_dists_q_num, num_examples=2)
        messages[-1]["content"] = messages[-1]["content"].format(
            description=model_input.nat_lang_scene_des,
            objects_from_nine=objects,
            missing_object_info=missing_objects
        )
        generic_dists = self.gpt.predict(messages=messages)
        # generic_dists = generic_dists.split("FINAL_ANSWER:")[-1].strip()

        scenic_dists_q_num = "six"
        messages = self._get_messages(questions, scenic_dists_q_num, num_examples=2)
        messages[-1]["content"] = messages[-1]["content"].format(
            description=model_input.nat_lang_scene_des,
            missing_info=missing_objects,
            distributions=generic_dists
        )
        # scenic_dists = self._query_and_parse_for_final_answer(
        #     messages=messages,
        #     temperature=temperature,
        #     verbose=verbose,
        #     question=scenic_dists_q_num,
        # )
        scenic_dists = self.gpt.predict(messages=messages)
        program_dists = scenic_dists.split("FINAL_ANSWER:")[-1].strip()
        updated_program = self.compile_scenic_program(
            questions["complier_system_prompt"].format(
                dist_info=questions[scenic_dists_q_num]["dist_info"],
            ),
            questions[scenic_dists_q_num]["complier_user_prompt"],
            program_dists,
            temperature,    
            verbose
        )

        current_program += "\n# PARAMETERS:\n" + updated_program

        return current_program

        
    def get_dynamic_objects(
        self,
        model_input: ModelInput,
        questions: dict,
        temperature: float,
        verbose: bool,
    ) -> str:
        temperature = 0.23
        messages=[
            {
                "role": "system",
                "content": questions["objects"]["dynamic"]["system_prompt"].format(
                    object_info=questions["objects"]["dynamic"]["object_info"]
                )
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["one"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["dynamic"]["assistant_answer_1"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["two"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["dynamic"]["assistant_answer_2"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["three"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["dynamic"]["assistant_answer_3"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["four"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["dynamic"]["assistant_answer_4"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["five"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["dynamic"]["assistant_answer_5"]
            },
            {
                "role": "user",
                "content": model_input.nat_lang_scene_des
            }
        ]

        response = self.gpt.predict(
            messages=messages,
            temperature=temperature,
        )

        if verbose:
            print(f"Response: {response}")

        response = response.split("My answer is:")[-1].strip()
        return response
    

    def get_static_objects(
        self,
        model_input: ModelInput,
        questions: dict,
        temperature: float,
        verbose: bool,
    ) -> str:
        temperature = 0.23
        messages=[
            {
                "role": "system",
                "content": questions["objects"]["static"]["system_prompt"].format(
                    object_info=questions["objects"]["static"]["object_info"]
                )
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["one"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["static"]["assistant_answer_1"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["two"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["static"]["assistant_answer_2"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["three"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["static"]["assistant_answer_3"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["four"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["static"]["assistant_answer_4"]
            },
            {
                "role": "user",
                "content": questions["examples"]["nat_lang_descriptions"]["five"]
            },
            {
                "role": "assistant",
                "content": questions["objects"]["static"]["assistant_answer_5"]
            },
            {
                "role": "user",
                "content": model_input.nat_lang_scene_des
            }
        ]

        response = self.gpt.predict(
            messages=messages,
            temperature=temperature,
        )

        if verbose:
            print(f"Response: {response}")

        response = response.split("My answer is:")[-1].strip()

        # TODO: complie check the response
        

        return response


    def compile_scenic_program(
        self,
        system_prompt: str,
        user_prompt: str,
        scenic_program: str,
        temperature: float,
        verbose: bool,
    ) -> str:
        for _ in range(5):
            try:
                with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".scenic") as f:
                    f.write(scenic_program)
                    f.flush()
                    temp_file_path = f.name
                    scenic.scenarioFromFile(temp_file_path, mode2D=True)

                return scenic_program
            except Exception as e:
                if verbose:
                    print("Error compiling Scenic program.")
                    print(e)

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt.format(
                            error=str(e),
                            program=scenic_program
                        )
                    },
                ]

                response = self.gpt.predict(
                    messages=messages,
                    temperature=temperature,
                )

                if verbose:
                    print(f"Updated scenic program:\n{response}")

                scenic_program = response
                
                print("Error compiling Scenic program.")
                print(e)
                print(scenic_program)
                print("Retrying...")

        # We could not compile the program so return the original
        # TODO: do a GPT-Judge thing here and ask which was the best then return that one?
        return scenic_program
    


    def compositionally_construct_scenic_program(
        self,
        model_input: ModelInput,
        temperature: float,
        max_tokens: int,
        verbose: bool,
    ) -> str:
        """
        Constructs a scenic program by parts
        """
        questions = None
        with open(PromptFiles.COMPOSITIONAL_GBNF.value, "r") as f:
            contents = f.read()
            questions = yaml.safe_load(contents)

        system_message = {
            "role": "system",
            "content": questions["context"],
        }

        # Step 1: getting the objects
        # objects = self._step_one(
        #     model_input=model_input,
        #     temperature=temperature,
        #     system_message=system_message,
        #     questions=questions,
        #     verbose=verbose,
        # )

        dynamic_objects = self.get_dynamic_objects(
            model_input=model_input,
            temperature=temperature,
            questions=questions,
            verbose=verbose,
        )
        
        dynamic_objects = self.compile_scenic_program(
            system_prompt=questions["complier_system_prompt"],
            user_prompt=questions["complier_user_prompt"],
            scenic_program=dynamic_objects,
            temperature=temperature,
            verbose=verbose
        )

        static_objects = self.get_static_objects(
            model_input=model_input,
            temperature=temperature,
            questions=questions,
            verbose=verbose,
        )

        static_objects = self.compile_scenic_program(
            system_prompt=questions["objects"]["complier_system_prompt"],
            user_prompt=questions["objects"]["complier_user_prompt"],
            scenic_program=static_objects,
            temperature=temperature,
            verbose=verbose
        )

        objects = "# Dynamic Objects:\n" + dynamic_objects + "\n\n# Static Objects:\n" + static_objects

        # Step 2: write the constants and variables section of the scenic program
        constants = self._build_constants(
            model_input=model_input,
            temperature=temperature,
            system_message=system_message,
            objects=objects,
            questions=questions,
            verbose=verbose,
        )

        # Step 3: 
        
    
