from sammo.components import GenerateText, Output
from sammo.utils import serialize_json

from sammo.base import LLMResult, Costs
from sammo.runners import RestRunner

from typing import Optional, Union

from completion import default_model

class OpenAIRestRunner(RestRunner):
    def __init__(self, model_id=default_model, api_config=None, cache = None, equivalence_class = "major", rate_limit = 2, max_retries = 50, max_context_window = None, retry = True, timeout = 60, max_timeout_retries = 1, use_cached_timeouts = True):
        if api_config == None:
            raise("An api configuration must be passed to the runner")
        super().__init__(model_id, api_config, cache, equivalence_class, rate_limit, max_retries, max_context_window, retry, timeout, max_timeout_retries, use_cached_timeouts)

    async def generate_text(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        randomness: Optional[float] = 0.6,
        seed: int = 0,
        priority: int = 0,
        **kwargs
    ) -> LLMResult:
        request = dict(
            model = self._model_id,
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=self._max_context_window or max_tokens,
            temperature = randomness
        )
        fingerprint = serialize_json({"seed": seed, "generative_model_id": self._model_id, **request})
        return await self._execute_request(request, fingerprint, priority)

    async def _call_backend(self, request: dict) -> dict:
        async with self._get_session() as session:
            async with session.post(
                f"https://api.openai.com/v1/chat/completions",
                json=request,
                headers={"Authorization": f"Bearer {self._api_config['api_key']}"}
            ) as response:
                return await response.json()
            
    def _to_llm_result(self, request: dict, json_data: dict, fingerprint: Union[str, bytes]):
        return LLMResult(
            json_data["choices"][0]["message"]["content"],
            costs=Costs(json_data["usage"]["prompt_tokens"], json_data["usage"]["completion_tokens"])
        )
    
if __name__ == "__main__":

    import dotenv, os

    dotenv.load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    runner = OpenAIRestRunner(
        default_model,
        api_config={"api_key": api_key}
    )

    print(Output(GenerateText("The Nott Memorial is ")).run(runner))