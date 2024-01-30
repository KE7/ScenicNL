from typing import cast
import click
import os
from pathlib import Path
import scenic
from scenicNL.adapters.anthropic_adapter import AnthropicAdapter, AnthropicModel
from scenicNL.adapters.openai_adapter import OpenAIAdapter, OpenAIModel
from scenicNL.adapters.lmql_adapter import LMQLAdapter, LMQLModel
from scenicNL.cache import APIError
from scenicNL.common import ModelInput, LLMPromptType, MAX_TOKEN_LENGTH
from scenicNL.utils.pdf_parse import PDFParser
import sys
import time


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
        [m.value for m in OpenAIModel] + [m.value for m in AnthropicModel] + [m.value for m in LMQLModel],
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
    default="report_txts",
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

@click.option(
    "--keep-filename",
    is_flag=True,
    help="Boolean condition to display or omit verbose output."
)

@click.option(
    "--temperature",
    type=click.FLOAT,
    default=0.7,
    show_default=True,
    help="Temperature to use for sampling.",
)

@click.option(
    "--num_workers",
    type=click.INT,
    default=1,
    show_default=True,
    help="Number of workers to use for parallel processing.",
)

@click.option(
    "--max_retries",
    type=click.INT,
    default=0,
    show_default=True,
    help="Maximum number of compiler-in-the-loop retries.",
)

@click.option(
    "--verbose_retries",
    is_flag=True,
    help="Boolean condition to display or omit verbose retries."
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
    num_workers: int,
    should_cache_retry_errors: bool,
    keep_filename: bool,
    temperature: float,
    max_retries: int,
    verbose_retries: bool,
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
    elif model in set(m.value for m in LMQLModel):
        print('Using LMQL model')
        adapter = LMQLAdapter(LMQLModel(model))
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
                dest_path = os.path.join(text_path, filename[:-4])
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
        query = query.replace("\n", " ")
        model_input = ModelInput(examples=example_list, nat_lang_scene_des=query)
        model_input_list.append(model_input)

    scenic_path = os.path.join(output_path, prompt_type.value + f'_{model.split(".")[0]}')
    scenic_metadata_path = os.path.join(scenic_path, '_metadata')
    if not os.path.exists(scenic_path):
        os.makedirs(scenic_path)
    if not os.path.exists(scenic_metadata_path):
        os.makedirs(scenic_metadata_path)
    scenic_error_path = os.path.join(scenic_metadata_path, 'errors.txt')
    scenic_metric_path = os.path.join(scenic_metadata_path, 'metrics.txt')
    start_time = time.time()

    compile_pass = compile_fail = execute_pass = execute_fail = api_error = 0
    for index, outputs in enumerate(adapter.predict_batch(
            model_inputs=model_input_list, 
            cache_path=cache_path, 
            num_predictions=1,
            temperature=temperature, 
            max_tokens=MAX_TOKEN_LENGTH, 
            prompt_type=prompt_type,
            should_cache_retry_errors=should_cache_retry_errors,
            verbose=verbose,
            num_workers=num_workers,
            ignore_cache=ignore_cache, 
            max_retries=max_retries,
            verbose_retries=verbose_retries,
            )
        ):
        for attempt, output in enumerate(outputs):
            if verbose:
                print(f'Output for query {index} attempt {attempt}: {output}')
            output = cast((str | APIError), output)
            if isinstance(output, APIError):
                api_error += 1
                continue
            if keep_filename:
                fstub = file_list[index]
            else:
                fstub = f'{index}-{attempt}'
            fstub = fstub[:-4] if fstub.endswith('.txt') else fstub
            debug = f'{index} - {file_list[index]}'
            fname = os.path.join(scenic_path, f'{fstub}.scenic')

            with open(fname, 'w') as f:
                f.write(output)
            try:
                ast = scenic.syntax.parser.parse_file(fname)
                print(f'Compiled input {debug} successfully: {ast}')
                # print(f'No errors when compiling input {debug}')
                compile_pass += 1
            except Exception as e:
                print(f'Error while compiling for input {debug}: {e}')
                compile_fail += 1
                with open(scenic_error_path, 'a') as f:
                    f.write(f'{index} - {fstub} compile error: {e}\n')
            try:
                scenario = scenic.scenarioFromFile(fname, mode2D=True)
                print(f'Executed input {debug} successfully: {scenario}')

                fsave = os.path.join(result_path, f'{fstub}.scenic')
                execute_pass += 1
                with open(fsave, 'w') as f:
                    f.write(output)
                # print(f'No errors when compiling input {debug}')
            except Exception as e:
                print(f'Error while executing for input {debug}: {e}')
                execute_fail += 1
                with open(scenic_error_path, 'a') as f:
                    f.write(f'{index} - {fstub} execute error: {e}\n')
            print('----------------\n\n')

    end_time = time.time()
    total = api_error + compile_fail + compile_pass # assert(total == len(model_input_list))

    api_error_rate = round((100*api_error/total), 2)
    compile_rate = round((100*compile_pass/total), 2)
    execute_rate = round((100*execute_pass/total), 2)
    eval_rate = round((end_time-start_time)/total, 5)

    print(f'Compile success rate: {compile_rate}%')
    print(f'Execute success rate: {execute_rate}%')
    print(f'## API error rate ##: {api_error_rate}%')
    print(scenic_metric_path)
    with open(scenic_metric_path, 'w') as f:
        f.write(f'Compile rate: {compile_rate}\n')
        f.write(f'Execute rate: {execute_rate}\n')
        f.write(f'API error rate: {api_error_rate}\n')
        f.write(f'Secs per program: {eval_rate}\n')

def _launch():
    # to stop Click handling errors
    ctx = main.make_context("main", sys.argv[1:])
    with ctx:
        main.invoke(ctx)

if __name__ == '__main__':
    _launch()