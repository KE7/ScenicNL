from enum import Enum
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

MAX_TOKEN_LENGTH = 1000

class LLMPromptType(Enum):
    PREDICT_ZERO_SHOT = "detection_zero_shot"
    PREDICT_FEW_SHOT = "detection_few_shot"
    PREDICT_SCENIC_TUTORIAL = "detection_scenic_tutorial"
    PREDICT_PYTHON_API = "detection_python_api"


@dataclass(frozen=True)
class ModelInput:
    """
    The inputs that we give to an LLM. Since each LLM has it's own way of receiving inputs 
    (ex. GPT using roles and Llama using INST and SYS tokens), we abstract the data here
    and let each implementing model worry about how to input the data.
    """
    examples: list[str]
    nat_lang_scene_des: str


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