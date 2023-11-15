import click
import os
from pathlib import Path
import scenic
from src.adapters.model_adapter import ModelAdapter
from src.adapters.openai_adapter import OpenAIAdapter, OpenAIModel
from src.common import ModelInput, LLMPromptType
from src.utils.pdf_parse import PDFParser
import sys

prompt_types = {
    'OPENAI_PREDICT_ZERO_SHOT': [OpenAIAdapter(OpenAIModel.GPT_35_TURBO), LLMPromptType.PREDICT_ZERO_SHOT],
    'OPENAI_PREDICT_SCENIC_TUTORIAL': [OpenAIAdapter(OpenAIModel.GPT_35_TURBO), LLMPromptType.PREDICT_SCENIC_TUTORIAL],
    'OPENAI_PREDICT_FEW_SHOT': [OpenAIAdapter(OpenAIModel.GPT_35_TURBO), LLMPromptType.PREDICT_FEW_SHOT],
    'OPENAI_PREDICT_PYTHON_API': [OpenAIAdapter(OpenAIModel.GPT_35_TURBO), LLMPromptType.PREDICT_PYTHON_API],
}

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
@click.argument(
    'prompt_type',
    type=click.Choice(
        prompt_types.keys(), 
        case_sensitive=False
    )
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
    "--text-path",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
    ),
    default="report-txts",
    show_default=True,
    help="Path to text directory for report text.",
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
@click.option(
    "--count",
    type=click.INT,
    default=20,
    show_default=True,
    help="Number of files to include for string matching component. Zero runs on all files."
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Boolean condition to display or omit verbose output."
)

def main(
    query_path: Path,
    prompt_type: str,
    output_path: Path,
    text_path: Path,
    example_path: Path,
    cache_path: Path,
    ignore_cache: bool,
    count: int,
    verbose: bool,
) -> None:
    """
    Generate simulator scenes from natural language descriptions.
    """
    adapter, prompt_type = prompt_types[prompt_type]
    query_list = []
    if not os.path.isdir(text_path):
        os.mkdir(text_path)
    if os.path.isdir(query_path):
        file_list = os.listdir(query_path)
        file_list = file_list[:count] if count else file_list
        for filename in file_list:
            full_path = os.path.join(query_path, filename)
            if filename.endswith('.pdf'):
                parsed_text = PDFParser.pdf_from_path(full_path)
                if verbose: print(parsed_text)
                query_list.append(parsed_text)
                dest_path = os.path.join(text_path, filename[:-4] + '.txt')
                with open(dest_path, 'w') as file:
                    file.write(parsed_text)
            elif filename.endswith('.txt'):
                with open(full_path, 'r') as file:
                    parsed_text = file.read()
                    if verbose: print(parsed_text)
                    query_list.append(parsed_text)
    else:
        with open(query_path) as file:
            for line in file:
                query_list.append(line.strip())
    if not verbose: print('Loaded all queries')

    example_list = []

    for filename in os.listdir(example_path):
        if filename.endswith('.txt') or filename.endswith('.scenic'):
            file_path = os.path.join(example_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    example_list.append(file.read())
    if not verbose: print('Loaded all examples')

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
            temperature=0, max_tokens=600, prompt_type=prompt_type)):
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