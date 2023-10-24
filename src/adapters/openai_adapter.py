import os

import openai


class OpenAIAdapter:
    """
    This class servers as a wrapper for the OpenAI API.
    """
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def get_response(self, prompt, **kwargs):
        return openai.Completion.create(prompt=prompt, **kwargs)