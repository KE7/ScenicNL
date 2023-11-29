from typing import cast
import click
import os
from pathlib import Path
import scenic
from scenicNL.adapters.anthropic_adapter import AnthropicAdapter, AnthropicModel
from scenicNL.adapters.openai_adapter import OpenAIAdapter, OpenAIModel
from scenicNL.cache import APIError
from scenicNL.common import ModelInput, LLMPromptType, MAX_TOKEN_LENGTH
from scenicNL.utils.pdf_parse import PDFParser
import sys


@click.group()
def main():
    pass

@main.command()  # This decorator turns the function into a command.
@click.option(
    "--query_path",
    type=click.Path(
        file_okay=True,
        dir_okay=True,
        exists=True,
    ),
)

@click.option(
    '-m', '--model',
    type=click.Choice(
        [m.value for m in OpenAIModel] + [m.value for m in AnthropicModel],
        case_sensitive=False
    ),
    default=OpenAIModel.GPT_35_TURBO.value,
    show_default=True,
    help="Model to use for generation.",
)

@click.option(
    '--llm_prompt_type',
    type=click.Choice(
        [p.value for p in LLMPromptType],
        case_sensitive=False
    ),
    default=LLMPromptType.PREDICT_FEW_SHOT.value,
    show_default=True,
    help="Type of prompting strategy to use.",
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
    "--result-path",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
    ),
    default="scenes",
    show_default=True,
    help="Path to all scenes that compile.",
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
    "--offset",
    type=click.INT,
    default=0,
    show_default=True,
    help="Offset to use as start index for evaluation run."
)

@click.option(
    "--verbose",
    is_flag=True,
    help="Boolean condition to display or omit verbose output."
)

@click.option(
    "--should_cache_retry_errors",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Boolean condition to cache or not cache errors."
)

def main(
    query_path: Path,
    output_path: Path,
    text_path: Path,
    example_path: Path,
    result_path: Path,
    cache_path: Path,
    ignore_cache: bool,
    count: int,
    offset: int,
    verbose: bool,
    model: str,
    llm_prompt_type: str,
    should_cache_retry_errors: bool,
) -> None:
    """
    Generate simulator scenes from natural language descriptions.
    """
    # adapter, prompt_type = prompt_types[prompt_type]
    prompt_type = LLMPromptType(llm_prompt_type)
    if model in set(m.value for m in AnthropicModel):
        adapter = AnthropicAdapter(AnthropicModel(model))
    elif model in set(m.value for m in OpenAIModel):
        print('Using OpenAI model: ', model)
        adapter = OpenAIAdapter(OpenAIModel(model))
    else:
        raise ValueError(f'Invalid model {model}')
    query_list = []
    if not os.path.isdir(text_path):
        os.mkdir(text_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    if os.path.isdir(query_path):
        file_list = os.listdir(query_path)
        file_list = file_list[offset:count+offset] if count else file_list[offset:]
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

    compile_pass, compile_fail, api_error = 0, 0, 0
    for index, outputs in enumerate(adapter.predict_batch(
            model_inputs=model_input_list, 
            cache_path=cache_path, num_predictions=1, 
            temperature=0, max_tokens=MAX_TOKEN_LENGTH, prompt_type=prompt_type,
            ignore_cache=ignore_cache, should_cache_retry_errors=should_cache_retry_errors)):
        for attempt, output in enumerate(outputs):
            if verbose:
                print(f'Output for query {index} attempt {attempt}: {output}')
            output = cast((str | APIError), output)
            if isinstance(output, APIError):
                api_error += 1
                continue
            fname = os.path.join(scenic_path, f'{index}-{attempt}.scenic')
            with open(fname, 'w') as f:
                f.write(output)
            try:
                scenic.scenarioFromFile(fname, mode2D=True)
                fname_compile = os.path.join(result_path, f'{index}-{attempt}.scenic')
                with open(fname_compile, 'w') as f:
                    f.write(output)
                print(f'No errors when compiling input {index}-{attempt}')
                compile_pass += 1
            except Exception as e:
                print(f'Error while compiling for input {index}-{attempt}: {e}')
                compile_fail += 1
            print('----------------\n\n')

    print(f'API error rate: {round((100*api_error/(api_error+compile_pass+compile_fail)), 2)}%')
    print(f'Compilation success rate: {round((100*compile_pass/(api_error+compile_pass+compile_fail)), 2)}%')

def _launch():
    # to stop Click handling errors
    ctx = main.make_context("main", sys.argv[1:])
    with ctx:
        main.invoke(ctx)

if __name__ == '__main__':
    _launch()