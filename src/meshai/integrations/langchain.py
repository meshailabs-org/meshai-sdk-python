"""LangChain/LangGraph auto-tracking integration.

Uses LangChain's callback handler system to track token usage
per-call, extracting the actual model from each LLM interaction.

Usage:
    from meshai import MeshAI
    from meshai.integrations.langchain import MeshAICallbackHandler

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="langchain")

    handler = MeshAICallbackHandler(client)

    # Use with any LangChain LLM or chain
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

    # Or with LangGraph
    config = {"callbacks": [handler]}
    result = graph.stream(input, config=config)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


class MeshAICallbackHandler:
    """LangChain callback handler that tracks token usage via MeshAI.

    Compatible with LangChain, LangGraph, and any LangChain-compatible framework.
    Extracts model name and token counts from each LLM call automatically.
    """

    def __init__(self, meshai: MeshAI) -> None:
        self._meshai = meshai

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call completes. Extracts usage and model."""
        try:
            # response is a LLMResult object
            if not hasattr(response, "llm_output") and not hasattr(response, "generations"):
                return

            # Extract model from llm_output
            llm_output = getattr(response, "llm_output", {}) or {}
            model = llm_output.get("model_name", "unknown")

            # Infer provider from model name
            provider = _infer_provider(model)

            # Extract token usage
            token_usage = llm_output.get("token_usage", {})
            if not token_usage:
                # Try usage_metadata from generations
                generations = getattr(response, "generations", [[]])
                if generations and generations[0]:
                    gen = generations[0][0]
                    usage_meta = getattr(gen, "usage_metadata", None)
                    if usage_meta:
                        token_usage = {
                            "prompt_tokens": getattr(usage_meta, "input_tokens", 0),
                            "completion_tokens": getattr(usage_meta, "output_tokens", 0),
                        }

            input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0) or 0
            output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0) or 0

            if input_tokens or output_tokens:
                self._meshai.track_usage(
                    model_provider=provider,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_type="langchain",
                )
        except Exception:
            logger.debug("Failed to track LangChain usage", exc_info=True)

    # Required callback interface methods (no-ops)
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        pass


def _infer_provider(model: str) -> str:
    """Infer LLM provider from model name."""
    model_lower = model.lower()
    if "gpt" in model_lower or "o1" in model_lower or "davinci" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower or "palm" in model_lower:
        return "google"
    if "llama" in model_lower or "mixtral" in model_lower or "mistral" in model_lower:
        return "meta"
    if "command" in model_lower:
        return "cohere"
    if "titan" in model_lower:
        return "bedrock"
    return "unknown"
