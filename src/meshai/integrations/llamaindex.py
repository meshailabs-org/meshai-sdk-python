"""LlamaIndex auto-tracking integration.

Uses LlamaIndex's callback handler system to track token usage per-call.

Usage:
    from meshai import MeshAI
    from meshai.integrations.llamaindex import MeshAILlamaHandler
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="llamaindex")

    handler = MeshAILlamaHandler(client)
    Settings.callback_manager = CallbackManager([handler])
    # All LlamaIndex LLM calls now auto-track usage
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


class MeshAILlamaHandler:
    """LlamaIndex callback handler that tracks token usage via MeshAI.

    Implements the CBEventType.LLM event callbacks to capture
    model name and token counts from each LLM interaction.
    """

    def __init__(self, meshai: MeshAI) -> None:
        self._meshai = meshai
        self._active_models: dict[str, str] = {}

    def on_event_start(
        self,
        event_type: Any,
        payload: dict | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Capture model info at the start of LLM events."""
        try:
            # CBEventType.LLM
            if payload and "model_name" in (payload.get("serialized", {}) or {}):
                self._active_models[event_id] = payload["serialized"]["model_name"]
            elif payload and "model" in (payload.get("serialized", {}) or {}):
                self._active_models[event_id] = payload["serialized"]["model"]
        except Exception:
            pass
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: dict | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Extract token usage at the end of LLM events."""
        try:
            if not payload:
                return

            # Try to get token counts from response
            response = payload.get("response")
            if response is None:
                return

            # Extract from raw response metadata
            raw = getattr(response, "raw", None) or {}
            usage = (
                getattr(raw, "usage", None)
                or (raw.get("usage") if isinstance(raw, dict) else None)
            )

            if usage is None:
                # Try additional_kwargs
                additional = getattr(response, "additional_kwargs", {}) or {}
                usage = additional.get("usage")

            if usage is None:
                return

            if hasattr(usage, "prompt_tokens"):
                input_tokens = usage.prompt_tokens or 0
                output_tokens = usage.completion_tokens or 0
            elif isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            else:
                return

            model = self._active_models.pop(event_id, "unknown")
            provider = _infer_provider(model)

            if input_tokens or output_tokens:
                self._meshai.track_usage(
                    model_provider=provider,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_type="llamaindex",
                )
        except Exception:
            logger.debug("Failed to track LlamaIndex usage", exc_info=True)

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict | None = None,
    ) -> None:
        pass


def _infer_provider(model: str) -> str:
    model_lower = model.lower()
    if "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    if "llama" in model_lower or "mixtral" in model_lower:
        return "meta"
    if "command" in model_lower:
        return "cohere"
    return "unknown"
