from model_adapter import ModelAdapter
import os
import openai
import json
from common import LLMPromptType, ModelInput
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


from constraints.lmql_decoding import construct_scenic_program


class LMQLModel(Enum):
    LMQL = "lmql" #TODO: we're just using the default lmql model rn, no real need to specify

class LMQLAdapter(ModelAdapter):
    """
    This class servers as a wrapper for the OpenAI API.
    """
    def __init__(self, model: LMQLModel):
        super().__init__()
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model
        

    def get_cache_key(
        self,
        *,
        model_input: ModelInput,
        temperature: float,
        max_tokens: int,
        prompt_type: LLMPromptType,
    ) -> str:
        return json.dumps(
            {
                "model": self.model.value,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message": self._format_message(model_input=model_input, prompt_type=prompt_type),
                "adapter" : "lmql"
            },
            sort_keys=True,
        )

    @retry(
        wait=wait_exponential_jitter(initial=10, max=60), stop=stop_after_attempt(5)
    )
    def _predict(
        self, 
        *, 
        model_input: ModelInput, 
        temperature: float, 
        max_length_tokens: int,
        prompt_type: LLMPromptType,
    ) -> str:

        response = construct_scenic_program(model_input.examples[0], model_input.nat_lang_scene_des)

        return response