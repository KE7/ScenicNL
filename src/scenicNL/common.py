from enum import Enum
import json
import os
import pinecone
import random
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

MAX_TOKEN_LENGTH = 1000

class LLMPromptType(Enum):
    PREDICT_ZERO_SHOT = "predict_zero_shot"
    PREDICT_FEW_SHOT = "predict_few_shot"
    PREDICT_SCENIC_TUTORIAL = "predict_scenic_tutorial"
    PREDICT_PYTHON_API = "predict_python_api"
    PREDICT_PYTHON_API_ONELINE = "predict_python_api_oneline"
    PREDICT_LMQL = "predict_lmql"
    PREDICT_FEW_SHOT_WITH_RAG = "predict_few_shot_with_rag"


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


def format_scenic_tutorial_prompt(prompt_path: str) -> str:
        """
        Formats the message providing introduction to Scenic language and syntax.
        """
        st_prompt = ''
        with open(prompt_path) as f:
            st_prompt = f.read()
        return st_prompt


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