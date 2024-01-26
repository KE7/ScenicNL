import abc
from multiprocessing.pool import ThreadPool
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable, Iterable

from scenicNL.cache import APIError, Cache
from scenicNL.common import LLMPromptType, ModelInput

"""
A lot of the code in this file is inspired from Java and copied from TensorTrust AI.
In general, we use * to force callers to use keyword arguments.
"""


class ModelAdapter(abc.ABC):
    """
    Abstract class for model adapters. This class is meant to be extended by
    model adapters for specific models. These adapters can be API-based
    (ex. OpenAI API) or local (ex. Llama).
    """
    @abc.abstractmethod
    def get_cache_key(
        self,
        *,
        model_input: ModelInput,
        temperature: float,
        max_tokens: int,
        prompt_type: LLMPromptType,
    ) -> str:
        """Generate a cache key for a given input. Must be implemented by
        subclasses.

        A good choice of key is to combine the model name, temperature, max
        token length, and tokenization of the input in the key, along with any
        other settings used for generation (it's fine for they key to be really
        long)."""
        raise NotImplementedError
    

    @abc.abstractmethod
    def _predict(
        self,
        *,
        model_input: ModelInput,
        temperature: float,
        max_length_tokens: int,
        prompt_type: LLMPromptType,
        verbose: bool,
        max_retries: int,
    ) -> str:
        """Perform a single prediction. Must be overriden by subclasses."""
        raise NotImplementedError

    def _api_fallback(
        self,
        description: ModelInput
    ) -> str:
        """Generate code from API calls, with _line_helper generating backups in case of API failures."""
        raise NotImplementedError 

    def _line_helper(
        self,
        line: ModelInput
    ) -> str:
        """
        Intent: use mini LLM calls to correct any malformed Scenic3_API calls (ones that fail `eval`).
        Idea: even imperfect API calls contain all info needed to formulate Scenic expression.
        """
        raise NotImplementedError

    def _batch_processor(
        self, 
        *,
        num_predictions: int,
        temperature: float,
        max_tokens: int,
        should_cache_retry_errors: bool,
        cache: Cache,
        prompt_type: LLMPromptType,
        ignore_cache: bool,
        verbose: bool,
        max_retries: int,
    ) -> Callable[[ModelInput], list[str | APIError]]:
        """
        Return a function that takes a list of model inputs and returns a
        list of predictions. This function will be used by the batch_predict
        method.
        """
        def process_single(
            model_input: ModelInput,
        ) -> list[str | APIError]:
            cache_key = self.get_cache_key(
                model_input=model_input,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_type=prompt_type,
            )
            if not ignore_cache:
                responses: list[str | APIError] = cache.get(cache_key)[:num_predictions]
                if should_cache_retry_errors:
                    responses = [r for r in responses if not isinstance(r, APIError)]
            else:
                responses = []
            # error checking in case we fell short of the number of predictions
            if len(responses) < num_predictions:
                num_missing = num_predictions - len(responses)
                for _ in range(num_missing):
                    # each individual model adapter is responsible for doing their own retries
                    try:
                        prediction = self._predict(
                            model_input=model_input,
                            temperature=temperature,
                            max_length_tokens=max_tokens,
                            prompt_type=prompt_type,
                            verbose=verbose,
                            max_retries=max_retries,
                        )
                        if prompt_type.value == LLMPromptType.PREDICT_PYTHON_API.value:
                            api_input = ModelInput(model_input.examples, prediction)
                            prediction = self._api_fallback(api_input) # specific function calling handlers
                    except Exception as e:
                        stacktrace = traceback.format_exc()
                        prediction = APIError(
                            f"Error while predicting for input {model_input.nat_lang_scene_des}" + 
                            f"Error: {e}\nStacktrace: {stacktrace}"
                        )
                        warnings.warn(f"Error while predicting for input {model_input.nat_lang_scene_des}" + 
                            f"Error: {e}\nStacktrace: {stacktrace}")
                    responses.append(prediction)
                if not ignore_cache:
                    cache.set(cache_key, responses)
            return responses
        
        return process_single

    def predict_batch(
        self,
        *,
        model_inputs: Iterable[ModelInput],
        cache_path: Path,
        num_predictions: int,
        temperature: float,
        max_tokens: int,
        prompt_type: LLMPromptType,
        should_cache_retry_errors: bool = True,
        verbose: bool = False,
        num_workers: int = 10,
        ignore_cache: bool = False,
        max_retries: int = 0,
    ) -> Iterable[list[str | APIError]]:
        """
        Given a stream of model inputs, return a stream of predictions. This
        method will batch the predictions and cache them to avoid repeated
        calls to the API thus saving money $$$$$ Young Mulah Baby
        """
        start_time = time.time()

        if verbose:
            print(f"Starting batch prediction using {self.__class__.__name__} " + 
                  "with {num_workers} workers")
            
        with Cache(cache_path) as cache:
            processor = self._batch_processor(
                num_predictions=num_predictions,
                temperature=temperature,
                max_tokens=max_tokens,
                should_cache_retry_errors=should_cache_retry_errors,
                cache=cache,
                prompt_type=prompt_type,
                ignore_cache=ignore_cache,
                verbose=verbose,
                max_retries=max_retries,
            )

            with ThreadPool(num_workers) as pool:
                for idx, predictions in enumerate(pool.imap(processor, model_inputs)):
                    yield predictions

                    if verbose and (idx + 1) % 10 == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_example = elapsed_time / (idx + 1)
                        print(f"Predicted {idx + 1} examples in {elapsed_time:.2f} seconds " + 
                              f"({avg_time_per_example:.2f} seconds per example)")
                        print(f"Example output: {predictions=}")

            if verbose:
                elapsed_time = time.time() - start_time
                print(f"Finished batch prediction in {elapsed_time:.2f} seconds")