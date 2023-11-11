import click
import os
from pathlib import Path
import scenic
from src.adapters.model_adapter import ModelAdapter
from src.adapters.openai_adapter import OpenAIAdapter, OpenAIModel
from src.common import ModelInput, LLMPromptType
from src.pdf_parse import PDFParser
import sys


@click.group()
def main():
    pass

@main.command()  # This decorator turns the function into a command.
@click.argument(
    "query_path",
    type=click.Path(
        file_okay=True,
        dir_okay=True,
        exists=True,
    ),
)
@click.option(
    "--output-path",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
    ),
    default="outputs",
    show_default=True,
    help="Path to output directory for results.",
)
@click.option(
    "--example-path",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
    ),
    default="examples",
    show_default=True,
    help="Path to examples for few-shot training.",
)
@click.option(
    "--cache-path",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
    ),
    default="cache.db",
    show_default=True,
    help="Path to SQLite3 database for caching.",
)
@click.option(
    "--ignore-cache",
    is_flag=True,
    help="Ignore cache and recompute predictions from scratch. THIS IS VERY EXPENSIVE.",
)


def main(
    query_path: Path,
    output_path: Path,
    example_path: Path,
    cache_path: Path,
    ignore_cache: bool
) -> None:
    """
    Generate simulator scenes from natural language descriptions.
    """
    query_list = []
    if os.path.isdir(query_path):
        for filename in os.listdir(query_path):
            if filename.endswith('.pdf'):
                full_path = os.path.join(query_path, filename)
                parsed_text = PDFParser.pdf_from_path(full_path)
                print(parsed_text)
                query_list.append(parsed_text)
    else:
        with open(query_path) as file:
            for line in file:
                query_list.append(line.strip())
                print(line.strip())

    adapter = OpenAIAdapter(OpenAIModel.GPT_35_TURBO)
    prompt_type = LLMPromptType.PREDICT_SCENIC_TUTORIAL

    example_list = []

    for filename in os.listdir(example_path):
        if filename.endswith('.txt') or filename.endswith('.scenic'):
            file_path = os.path.join(example_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    example_list.append(file.read())

    model_input_list = []
    for query in query_list:
        model_input = ModelInput(examples=example_list, nat_lang_scene_des=query)
        model_input_list.append(model_input)

    scenic_path = os.path.join(output_path, prompt_type.value)
    if not os.path.exists(scenic_path):
        os.makedirs(scenic_path)

    compile_pass, compile_fail = 0, 0
    for index, outputs in enumerate(adapter.predict_batch(
            model_inputs=model_input_list, 
            cache_path=cache_path, num_predictions=1, 
            temperature=0, max_tokens=500, prompt_type=prompt_type)):
        for attempt, output in enumerate(outputs):
            print(f'{output}\n\n')
            fname = os.path.join(scenic_path, f'{index}-{attempt}.scenic')
            with open(fname, 'w') as f:
                f.write(output)
            try:
                scenic.scenarioFromFile(fname, mode2D=True)
                compile_pass += 1
            except Exception as e:
                print(f'Failed with error: {e}')
                compile_fail += 1
            print('----------------\n\n')

    print(f'Compilation success rate: {round((100*compile_pass/(compile_fail+compile_pass)), 2)}%')

def _launch():
    # to stop Click handling errors
    ctx = main.make_context("main", sys.argv[1:])
    with ctx:
        main.invoke(ctx)

if __name__ == '__main__':
    _launch()