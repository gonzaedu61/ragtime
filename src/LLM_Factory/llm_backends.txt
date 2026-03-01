from typing import Protocol, runtime_checkable
from openai import AsyncAzureOpenAI


@runtime_checkable
class LLMBackend(Protocol):
    """
    LLM backend interface.

    Implementations should provide:
    - async acomplete(prompt: str) -> str
    - optional sync complete(prompt: str) -> str (wrapper)
    """

    async def acomplete(self, prompt: str) -> str:
        ...

    def complete(self, prompt: str) -> str:
        ...


class AzureOpenAIBackend:
    """
    Async Azure OpenAI backend with an optional sync wrapper.
    """

    def __init__(self, endpoint: str, api_key: str, deployment: str, model_name: str, api_version: str):
        self.model_name = model_name
        self.deployment = deployment

        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

    async def acomplete(self, prompt: str) -> str:
        """
        Asynchronous completion call.
        """
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR calling LLM: {e}]"

    def complete(self, prompt: str) -> str:
        """
        Synchronous wrapper around the async method.
        Useful for legacy/sync code paths.
        """
        import asyncio

        try:
            return asyncio.run(self.acomplete(prompt))
        except RuntimeError:
            # If there's already a running loop (e.g. in notebooks),
            # fall back to creating a new task and waiting on it.
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.acomplete(prompt))
