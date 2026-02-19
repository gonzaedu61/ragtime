from typing import Literal
from .llm_backends import (
    LLMBackend,
    AzureOpenAIBackend,
)


def create_llm(
    backend: Literal["openai", "azure", "local", "mock"],
    **kwargs
) -> LLMBackend:
    """
    Factory for LLM backends.

    Currently supports:
    - "azure": AzureOpenAIBackend (async-first, with sync wrapper)
    """

    if backend == "azure":
        return AzureOpenAIBackend(
            endpoint=kwargs["endpoint"],
            api_key=kwargs["api_key"],
            deployment=kwargs["deployment"],
            model_name=kwargs["model_name"],
            api_version=kwargs["api_version"],
        )

    raise ValueError(f"Unknown LLM backend: {backend}")
