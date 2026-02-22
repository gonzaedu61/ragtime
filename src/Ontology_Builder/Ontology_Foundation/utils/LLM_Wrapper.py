import json
import traceback

class LLM_Wrapper:
    def __init__(self, llm_backend):
        self.llm = llm_backend

    def call(self, prompt: str) -> str:
        """
        Synchronous LLM call with error handling.
        """
        try:
            response = self.llm.complete(prompt)
            return response.strip()
        except Exception as e:
            traceback.print_exc()
            return f"[LLM_ERROR] {e}"

    async def acall(self, prompt: str) -> str:
        """
        Async LLM call.
        """
        try:
            response = await self.llm.acomplete(prompt)
            return response.strip()
        except Exception as e:
            traceback.print_exc()
            return f"[LLM_ERROR] {e}"
