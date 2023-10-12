import openai


class OpenAIAdapter:
    """
    This class servers as a wrapper for the OpenAI API.
    The default strategy is to do zero-shot prompting whichover doesn't work well given
    that GPT does not know about scenic.
    Strategies that we need to implement:
    - Few-shot learning
    - Fine-tuning on the scenic codebase: https://platform.openai.com/docs/guides/fine-tuning
    - Function calling: https://platform.openai.com/docs/guides/gpt/function-calling
    """
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_response(self, prompt, **kwargs):
        return openai.Completion.create(prompt=prompt, **kwargs)