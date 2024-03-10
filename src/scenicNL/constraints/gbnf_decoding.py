

from typing import Dict, List
import openai
import os
import requests

import yaml
import scenic
from scenicNL.adapters.anthropic_adapter import AnthropicAdapter, AnthropicModel
from scenicNL.common import LOCAL_MODEL_DEFAULT_PARAMS, LOCAL_MODEL_ENDPOINT, ModelInput, PromptFiles
import tempfile

class CompositionalScenic():

    def __init__(self):
        super().__init__()
        openai.api_base = "http://localhost:8080/v1"
        openai.api_key = "sk-no-key-required"
        self.anthropic = AnthropicAdapter(model=AnthropicModel.CLAUDE_3_MEDIUM)


    @staticmethod
    def _query_and_parse_for_final_answer(
        messages: List[Dict],
        temperature: float,
        question: str,
        grammar: str,
        verbose: bool,
    ) -> str:
        response = openai.ChatCompletion.create(
            model="LLama_CPP",
            messages=messages,
            temperature=temperature,
            grammar=grammar,
        )
        answer = response.choices[0].message.content
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
        user_message = {
            "role": "user",
            "content": questions[question_num]["question"].format(description=model_input.nat_lang_scene_des),
        }
        messages = [system_message, user_message]
        answer = self._query_and_parse_for_final_answer(
            messages=messages,
            temperature=temperature,
            verbose=verbose,
            question=question_num,
            grammar=questions["single_answer_grammar"],
        )

        return answer
    

    def _build_constants(
        self,
        model_input: ModelInput,
        temperature: float,
        system_message: dict,
        objects: str,
        questions: dict,
        verbose: bool,
    ) -> str:
        get_objs_q_num = "nine"
        prompt = questions[get_objs_q_num]["question"].format(objects_from_one=objects)
        user_message = {
            "role": "user",
            "content": prompt,
        }
        messages = [user_message]
        response = self.anthropic.predict(messages=messages)
        current_program = response.split("FINAL_ANSWER:")[-1].strip()

        # objects_list = objects.split("\n")
        # objects_list = map(lambda x: x.strip(), objects_list)
        # objects_list = [obj for obj in objects_list if obj != ""]
        # program_objects = ""
        # for object in objects_list:
        #     object = object.strip()
        #     user_message = {
        #         "role": "user",
        #         "content": questions[get_objs_q_num]["question"].format(objects_from_one=object),
        #     }
        #     single_question_grammar = questions["single_answer_grammar"]
        #     messages = [user_message]
        #     program_object = self._query_and_parse_for_final_answer(
        #         messages=messages,
        #         temperature=temperature,
        #         verbose=verbose,
        #         question=get_objs_q_num,
        #         grammar=single_question_grammar,
        #     )
        #     program_object = program_object.strip()
        #     program_objects = program_objects + program_object + "\n"

        # self.validate_with_feedback(current_program)

        user_message = {
            "role": "user",
            "content": questions["four"]["b"]["question"].format(
                description=model_input.nat_lang_scene_des, 
                objects_from_nine=current_program,
            ),
        }
        messages = [user_message]
        missing_objects = self._query_and_parse_for_final_answer(
            messages=messages,
            temperature=temperature,
            verbose=verbose,
            question="four b",
            grammar="",
        )

        single_question_grammar = questions["single_answer_grammar"]
        generic_dists_q_num = "five"
        user_message = {
            "role": "user",
            "content": questions[generic_dists_q_num]["question"].format(
                description=model_input.nat_lang_scene_des, 
                missing_object_info=missing_objects,
            ),
        }
        messages = [user_message]
        generic_dists = self._query_and_parse_for_final_answer(
            messages=messages,
            temperature=temperature,
            verbose=verbose,
            question=generic_dists_q_num,
            grammar=single_question_grammar,
        )
        generic_dists = generic_dists.split("FINAL_ANSWER:")[-1].strip()

        scenic_dists_q_num = "six"
        user_message = {
            "role": "user",
            "content": questions[scenic_dists_q_num]["question"].format(
                description=model_input.nat_lang_scene_des, 
                missing_info=missing_objects,
                distributions=generic_dists,
            ),
        }
        # messages = [system_message, user_message]
        # scenic_dists = self._query_and_parse_for_final_answer(
        #     messages=messages,
        #     temperature=temperature,
        #     verbose=verbose,
        #     question=scenic_dists_q_num,
        # )
        messages = [user_message]
        scenic_dists = self.anthropic.predict(messages=messages)
        program_dists = scenic_dists.split("FINAL_ANSWER:")[-1].strip()

        current_program = "# CONSTANTS\n" + current_program + "\n" + program_dists

        # Finally, we are ready to construct the first part of a Scenic program
        # we will ask the LLM to take the objects and scenic distributions and
        # declare them as constants in a Scenic program
        # we can re-try up to compiler_error many times
        # TODO: get compiler_error from config
        prompt = questions["constants"]["prompt"].format(
                    example_1=model_input.examples[0], # TODO: these examples are wrong
                    example_2=model_input.examples[1], # TODO: they are full scenic programs
                    example_3=model_input.examples[2], # TODO: we need just the constants section
                    objects=objects,
                    distributions=scenic_dists,
                )
        for _ in range(5):
            user_message = {
                "role": "user",
                "content": prompt,
            }
            data = {"prompt": prompt, "temperature": temperature} | LOCAL_MODEL_DEFAULT_PARAMS
            data["grammar"] = questions["constants"]["grammar"]

            response = requests.post(LOCAL_MODEL_ENDPOINT, json=data)
            if response.status_code != 200:
                raise ValueError(f"Local model returned status code {response.status_code}") 
            response_body = response.json()
            content = response_body["content"]
            if verbose:
                print("---- Predicted constants section of Scenic program ----")
                print(content)
                print("---- End of predicted constants section of Scenic program ----")
            scenic_program = content.split("FINAL_ANSWER: ")[-1]
            # Let's write the scenic program to a temp file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(scenic_program)
                temp_file_path = f.name
            try:
                scenic.scenarioFromFile(temp_file_path, mode2D=True)
                os.remove(temp_file_path)
                return scenic_program
            except Exception as e:
                # TODO modify prompt to include the compiler_error
                print("Error compiling Scenic program.")
                print(e)
                print(scenic_program)
                print("Retrying...")
                os.remove(temp_file_path)
                continue

        # we could not compile the program so we just return our best effort
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
        objects = self._step_one(
            model_input=model_input,
            temperature=temperature,
            system_message=system_message,
            questions=questions,
            verbose=verbose,
        )

        # Step 2: write the constants and variables section of the scenic program
        constants = self._build_constants(
            model_input=model_input,
            temperature=temperature,
            system_message=system_message,
            objects=objects,
            questions=questions,
            verbose=verbose,
        )
        
    
