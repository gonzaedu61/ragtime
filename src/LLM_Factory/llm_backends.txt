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



import aiohttp
import asyncio
from typing import Optional


class GoogleColabAPIBackend:
    """
    Backend that calls a FastAPI LLM server running in Google Colab.
    Matches the interface of AzureOpenAIBackend.
    """

    def __init__(self, base_url: str):
        """
        base_url: the public ngrok URL, e.g. "https://1234abcd.ngrok-free.app"
        """
        self.base_url = base_url.rstrip("/")

    async def acomplete(self, prompt: str) -> str:
        """
        Asynchronous call to the remote FastAPI endpoint.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/generate",
                    json={"prompt": prompt},
                    timeout=120
                ) as resp:
                    if resp.status != 200:
                        return f"[ERROR {resp.status}: {await resp.text()}]"

                    data = await resp.json()
                    return data.get("response", "").strip()

        except Exception as e:
            return f"[ERROR calling Colab LLM API: {e}]"

    def complete(self, prompt: str) -> str:
        """
        Synchronous wrapper around the async method.
        """
        try:
            return asyncio.run(self.acomplete(prompt))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.acomplete(prompt))

    async def astatus(self) -> bool:
        """
        Asynchronous health check for the Colab FastAPI server.
        Returns True if alive, False otherwise.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/status", timeout=10) as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    return data.get("alive", False)
        except Exception:
            return False

    def status(self) -> bool:
        """
        Synchronous wrapper for the health check.
        """
        import asyncio
        try:
            return asyncio.run(self.astatus())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.astatus())


