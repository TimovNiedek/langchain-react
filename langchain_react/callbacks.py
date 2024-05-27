from pprint import pprint
from typing import Dict, Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class LoggingAgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Called when the LLM is started."""
        print(f"***Starting LLM with prompts:***")
        print(prompts[0])
        print("******")

    def on_llm_end(
        self,
        response: LLMResult,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Called when the LLM is finished."""
        print(f"***LLM finished with response:***")
        print(response.generations[0][0].text)
        print("******")
