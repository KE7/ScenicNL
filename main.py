import os
from pathlib import Path
from src.adapters.model_adapter import ModelAdapter
from src.adapters.openai_adapter import OpenAIAdapter, OpenAIModel
from src.common import ModelInput, LLMPromptType
import sys
import click


@click.group()
def main():
    pass

@main.command()  # This decorator turns the function into a command.
@click.argument(
    "query_set",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
    ),
)
@click.option(
    "--cache-path",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
    ),
    default="/users/ke/Documents/Cal/research/scenicNL/cache.db",
    show_default=True,
    help="Path to SQLite3 database for caching.",
)
@click.option(
    "--ignore-cache",
    is_flag=True,
    help="Ignore cache and recompute predictions from scratch. THIS IS VERY EXPENSIVE.",
)


def main(
    query_set: Path,
    cache_path: Path,
    ignore_cache: bool
) -> None:
    """
    Generate simulator scenes from natural language descriptions.
    """
    query_list = []
    with open(query_set) as file:
        for line in file:
            query_list.append(line.strip())
            print(line.strip())

    adapter = OpenAIAdapter(OpenAIModel.GPT_35_TURBO)

    examples, examples_dir = [], 'examples'

    for filename in os.listdir(examples_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(examples_dir, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    examples.append(file.read())

    model_input_list = []
    for query in query_list:
        model_input = ModelInput(examples=examples, nat_lang_scene_des=query)
        model_input_list.append(model_input)

    for output in adapter.predict_batch(
            model_inputs=model_input_list, 
            cache_path=cache_path, num_predictions=1, 
            temperature=0, max_tokens=500, prompt_type=LLMPromptType.PREDICT_PYTHON_API):
        print(output[0])
        print('\n\n\n')


def _launch():
    # to stop Click handling errors
    ctx = main.make_context("main", sys.argv[1:])
    with ctx:
        main.invoke(ctx)

if __name__ == '__main__':
    _launch()