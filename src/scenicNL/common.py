from enum import Enum
import json
import os
import subprocess
import tempfile
import pinecone
import random
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch

MAX_TOKEN_LENGTH = 3600
DISCUSSION_TEMPERATURE = 0.8
NUM_EXPERTS = 3

class LLMPromptType(Enum):
    PREDICT_ZERO_SHOT = "predict_zero_shot"
    PREDICT_FEW_SHOT = "predict_few_shot"
    PREDICT_SCENIC_TUTORIAL = "predict_scenic_tutorial"
    PREDICT_PYTHON_API = "predict_python_api"
    PREDICT_PYTHON_API_ONELINE = "predict_python_api_oneline"
    PREDICT_LMQL = "predict_lmql"
    PREDICT_FEW_SHOT_WITH_RAG = "predict_few_shot_with_rag"
    PREDICT_FEW_SHOT_WITH_HYDE = "predict_few_shot_hyde"
    PREDICT_FEW_SHOT_WITH_HYDE_TOT = "predict_few_shot_hyde_tot"
    PREDICT_TOT_THEN_HYDE = "predict_tot_then_hyde"
    EXPERT_DISCUSSION = "expert_discussion"
    EXPERT_SYNTHESIS = "expert_synthesis"
    FIX_COMPILER_ERROR = "fix_compiler_error"


class PromptFiles(Enum):
    PROMPT_PATH = os.path.join(os.curdir, 'src', 'scenicNL', 'adapters', 'prompts')
    DISCUSSION_TO_PROGRAM = os.path.join(PROMPT_PATH, 'discussion_to_program.txt')
    DYNAMIC_SCENARIOS = os.path.join(PROMPT_PATH, 'dynamic_scenarios_prompt.txt')
    PYTHON_API = os.path.join(PROMPT_PATH, 'python_api_prompt.txt')
    QUESTION_REASONING = os.path.join(PROMPT_PATH, 'question_reasoning.txt')
    SCENIC_TUTORIAL = os.path.join(PROMPT_PATH, 'scenic_tutorial_prompt.txt')
    TOT_EXPERT_DISCUSSION = os.path.join(PROMPT_PATH, 'tot_questions.txt')
    EXPERT_SYNTHESIS = os.path.join(PROMPT_PATH, 'expert_synthesis.txt')
    FIX_COMPILER_ERROR = os.path.join(PROMPT_PATH, 'fix_compiler_error.txt')


@dataclass(frozen=True)
class ModelInput:
    """
    The inputs that we give to an LLM. Since each LLM has it's own way of receiving inputs 
    (ex. GPT using roles and Llama using INST and SYS tokens), we abstract the data here
    and let each implementing model worry about how to input the data.
    """
    examples: list[str]
    nat_lang_scene_des: str
    first_attempt_scenic_program: Optional[str] = None
    expert_discussion: Optional[str] = None
    panel_discussion: Optional[List[str]] = None
    compiler_error: Optional[str] = None


def load_jsonl(
        dataset_path: Path,
        *,
        max_examples: Optional[int]
    ) -> Iterable[Dict[Any, Any]]:
    n_yielded = 0
    with dataset_path.open() as lines:
        for line in lines:
            if max_examples is not None and n_yielded >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n_yielded += 1


def write_jsonl(
        output_path: Path,
        data: Iterable[Dict[Any, Any]],
        *,
        shuffle_seed: Optional[int]
    ) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shuffle_seed is not None:
        data = list(data)
        random.Random(shuffle_seed).shuffle(data)

    with output_path.open("w") as output_file:
        # we will separate each example with a newline but we don't want a newline at the end
        for idx, example in enumerate(data):
            if idx > 0:
                output_file.write("\n")
            json.dump(example, output_file)


def format_scenic_tutorial_prompt() -> str:
        """
        Formats the message providing introduction to Scenic language and syntax.
        """
        st_prompt = ''
        with open(PromptFiles.SCENIC_TUTORIAL.value) as f:
            st_prompt = f.read()
        return st_prompt


def format_reasoning_prompt(model_input: ModelInput) -> str:
        """
        Formats the message providing introduction to Scenic language and syntax.
        """
        st_prompt = ''
        with open(PromptFiles.QUESTION_REASONING.value) as f:
            st_prompt = f.read()
        st_prompt = st_prompt.format(
            example_1=model_input.examples[0],
            example_2=model_input.examples[1],
            example_3=model_input.examples[2],
            natural_language_description=model_input.nat_lang_scene_des
        )
        return st_prompt


def get_discussion_prompt() -> str:
        prompt = ""
        with open(PromptFiles.TOT_EXPERT_DISCUSSION.value) as f:
            prompt = f.read()

        # prompt = prompt.format(
        #     example_1=model_input.examples[0],
        #     natural_language_description=model_input.nat_lang_scene_des,
        # )

        return prompt


def get_expert_synthesis_prompt() -> str:
        prompt = ""
        with open(PromptFiles.EXPERT_SYNTHESIS.value) as f:
            prompt = f.read()

        # prompt = prompt.format(
        #     natural_language_description=model_input.nat_lang_scene_des,
        #     expert_1=model_input.panel_discussion[0],
        #     expert_2=model_input.panel_discussion[1],
        #     expert_3=model_input.panel_discussion[2],
        # )

        return prompt


def get_discussion_to_program_prompt() -> str:
        prompt = ""
        with open(PromptFiles.DISCUSSION_TO_PROGRAM.value) as f:
            prompt = f.read()

        # prompt = prompt.format(
        #     natural_language_description=model_input.nat_lang_scene_des,
        #     example_1=model_input.examples[0],
        #     example_2=model_input.examples[1],
        #     example_3=model_input.examples[2],
        #     expert_discussion=model_input.expert_discussion,
        # )

        return prompt


def get_compiler_feedback_prompt() -> str:
        prompt = ""
        with open(PromptFiles.FIX_COMPILER_ERROR.value) as f:
            prompt = f.read()

        return prompt


def run_scenic_program(scenic_program : str,) -> (bool, str): # (success, output)
    # Step 1: Write the Scenic program to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.scenic', delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(scenic_program.encode())

    # Step 2: Execute the Scenic command
    try:
        # Using subprocess.run to execute the command
        # The command will run for up to 10 seconds
        result = subprocess.run(['scenic', temp_file_name, '--2d'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT, 
                                timeout=10)

        # If the command is successful, return a success message
        if result.returncode == 0:
            return True, scenic_program

        # If there's an error, return the compiler output
        else:
            return False, result.stdout.decode()

    except subprocess.TimeoutExpired:
        # If the command times out after 10 seconds
        # We assume that the simulation was actually ran so we return a success message
            return True, scenic_program

    finally:
        # Clean up: Delete the temporary file
        os.remove(temp_file_name)


class VectorDB():
    def __init__(
            self,
            index_name: str = 'scenic-programs',
            model_name: str = 'sentence-transformers/all-mpnet-base-v2',
            dimension: int = 768,
            verbose: bool = False,
        ) -> None:
        """
        It is the caller's responsibility to ensure that the model's 
        hidden size matches the dimension.
        """
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.index = self._pinecone_init(
            index_name=index_name,
            dimension=dimension,
            verbose=verbose,
        )

    def _pinecone_init(
        self,
        index_name : str,
        dimension : int,
        verbose : bool,
    ) -> pinecone.Index:
        api_key = os.getenv('PINECONE_API_KEY')
        pinecone.init(
            api_key=api_key,
            environment='gcp-starter',
        )

        active_indexes = pinecone.list_indexes()

        if index_name not in active_indexes:
            pinecone.create_index(
                name=index_name, 
                metric='dotproduct',
                dimension=dimension,
            )

        index = pinecone.Index(index_name)
        if verbose:
            print(f"Index statistics: \n{index.describe_index_stats()}")
        return index

    # We will use mean pooling to accumulate the attention weights
    def _mean_pooling(
            self,
            model_output,
            attention_mask,
        ) -> torch.Tensor:
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(
        self,
        text : str,
    ) -> List[float]:
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.device))

        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'].to(self.device))
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy().tolist()[0]
    
    def upsert(
        self,
        docs : list,
        index : pinecone.Index,
        start : int = 0
    ) -> None:
        vectors = []
        for i in range(len(docs)):

            doc = docs[i]
            id_batch = str(i + start)
            embedding = self.get_embedding(doc)
            metadata = {'text' : doc}
            vectors.append((id_batch, embedding, metadata))

        if len(vectors) > 0:
            index.upsert(vectors)


    def query(
        self,
        query_or_queries : Union[str, List[str]],
        top_k : int = 3,
    ) -> Optional[List[str]]:
        if isinstance(query_or_queries, str):
            query_or_queries = [query_or_queries]
        
        query_embeddings = [self.get_embedding(query) for query in query_or_queries]
        results_dict = self.index.query(query_embeddings, top_k=top_k, include_metadata=True)
        passages = [results['metadata']['text'] for results in results_dict['matches']]
        if len(passages) < top_k:
            return None
        return passages


def few_shot_prompt_with_rag(
        vector_index: VectorDB,
        model_input: ModelInput,
        few_shot_prompt_generator: Callable[[ModelInput, bool], List[Dict[str, str]]] | Callable[[ModelInput, bool], str],
        top_k: int = 3,
    ) -> str | List[Dict[str, str]]:
        examples = vector_index.query(model_input.nat_lang_scene_des, top_k=top_k)
        if examples is None: # if the query fails, we just return the few shot prompt with default examples
            return few_shot_prompt_generator(model_input, False)
        
        relevant_model_input = ModelInput(
            examples=[example for example in examples],
            nat_lang_scene_des=model_input.nat_lang_scene_des,
            first_attempt_scenic_program=model_input.first_attempt_scenic_program,
        )
        return few_shot_prompt_generator(relevant_model_input, False)