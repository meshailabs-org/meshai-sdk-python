"""AutoGen auto-tracking integration.

Uses AutoGen's event logging system to track token usage per-call,
extracting the actual model from each LLM interaction.

Usage:
    from meshai import MeshAI
    from meshai.integrations.autogen import MeshAILogHandler

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="autogen")

    # Add as a logging handler
    import logging
    logging.getLogger("autogen_core").addHandler(MeshAILogHandler(client))

    # Or use the convenience function
    from meshai.integrations.autogen import track_autogen
    track_autogen(client)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


class MeshAILogHandler(logging.Handler):
    """Logging handler that captures AutoGen LLMCall events for usage tracking.

    AutoGen logs LLM calls via the standard Python logging module
    under the 'autogen_core' logger with event type 'LLMCall'.
    """

    def __init__(self, meshai: MeshAI) -> None:
        super().__init__()
        self._meshai = meshai

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # AutoGen logs LLMCall events as structured records
            if not hasattr(record, "event_type"):
                # Try parsing from message
                msg = record.getMessage()
                if "LLMCall" not in msg and "usage" not in msg.lower():
                    return
                self._try_parse_message(msg)
                return

            if getattr(record, "event_type", "") != "LLMCall":
                return

            event = getattr(record, "event_data", None)
            if event is None:
                return

            self._process_event(event)
        except Exception:
            logger.debug("Failed to track AutoGen usage", exc_info=True)

    def _process_event(self, event: Any) -> None:
        """Process an AutoGen LLMCall event."""
        # Extract from LLMCallEvent
        response = getattr(event, "response", None) or event.get("response", {}) if isinstance(event, dict) else None

        if response is None:
            return

        # Get usage
        if hasattr(response, "usage"):
            usage = response.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
        elif isinstance(response, dict):
            usage = response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
        else:
            return

        # Get model
        model = (
            getattr(response, "model", None)
            or (response.get("model") if isinstance(response, dict) else None)
            or "unknown"
        )

        provider = _infer_provider(model)

        if input_tokens or output_tokens:
            self._meshai.track_usage(
                model_provider=provider,
                model_name=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_type="autogen",
            )

    def _try_parse_message(self, msg: str) -> None:
        """Try to parse usage from a log message string."""
        try:
            data = json.loads(msg)
            if "usage" in data:
                self._process_event(data)
        except (json.JSONDecodeError, ValueError):
            pass


def track_autogen(meshai: MeshAI) -> None:
    """Convenience function to enable AutoGen usage tracking.

    Adds a MeshAILogHandler to AutoGen's event logger.
    """
    handler = MeshAILogHandler(meshai)

    # AutoGen core event logger
    autogen_logger = logging.getLogger("autogen_core")
    autogen_logger.addHandler(handler)

    # Also try the older autogen logger name
    alt_logger = logging.getLogger("autogen")
    alt_logger.addHandler(handler)

    logger.info("AutoGen usage tracking enabled")


def _infer_provider(model: str) -> str:
    """Infer LLM provider from model name."""
    model_lower = model.lower()
    if "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    if "llama" in model_lower or "mixtral" in model_lower:
        return "meta"
    return "unknown"
