from typing import Protocol

from app.config import get_settings


class LLMUnavailableError(Exception):
    pass


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class UnconfiguredLLMProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMUnavailableError("No reasoning provider is configured.")


class OpenAICompatibleProvider:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 1600):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


def create_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.LLM_PROVIDER.lower() != "openai" or not settings.OPENAI_API_KEY:
        return UnconfiguredLLMProvider()
    return OpenAICompatibleProvider(
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
