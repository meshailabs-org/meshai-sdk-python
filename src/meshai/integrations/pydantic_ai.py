"""Pydantic AI auto-tracking integration.

Wraps Pydantic AI Agent.run to capture usage from RunResult.

Usage:
    from meshai import MeshAI
    from meshai.integrations.pydantic_ai import track_pydantic_ai

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="pydantic-ai")

    track_pydantic_ai(client)
    # All Pydantic AI agents now auto-track usage
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


def track_pydantic_ai(meshai: MeshAI) -> None:
    """Enable usage tracking for Pydantic AI agents.

    Patches Agent.run and Agent.run_sync to capture RunUsage after execution.
    """
    try:
        from pydantic_ai import Agent
    except ImportError:
        logger.warning("Pydantic AI not installed. Install with: pip install pydantic-ai")
        return

    if getattr(Agent, "_meshai_tracked", False):
        return

    # Patch run_sync
    if hasattr(Agent, "run_sync"):
        original_sync = Agent.run_sync

        @functools.wraps(original_sync)
        def tracked_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_sync(self, *args, **kwargs)
            _extract_usage(self, result, meshai)
            return result

        Agent.run_sync = tracked_sync

    # Patch run (async)
    if hasattr(Agent, "run"):
        original_run = Agent.run

        @functools.wraps(original_run)
        async def tracked_run(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = await original_run(self, *args, **kwargs)
            _extract_usage(self, result, meshai)
            return result

        Agent.run = tracked_run

    Agent._meshai_tracked = True  # noqa: SLF001
    logger.info("Pydantic AI usage tracking enabled")


def _extract_usage(agent: Any, result: Any, meshai: MeshAI) -> None:
    """Extract usage from a Pydantic AI RunResult."""
    try:
        usage = getattr(result, "usage", None)
        if usage is None:
            # Try result.usage()
            usage_fn = getattr(result, "usage", None)
            if callable(usage_fn):
                usage = usage_fn()

        if usage is None:
            return

        input_tokens = getattr(usage, "request_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "response_tokens", 0) or getattr(usage, "output_tokens", 0) or 0

        # Extract model from agent
        model = getattr(agent, "model", None)
        model_name = "unknown"
        if model:
            if hasattr(model, "model_name"):
                model_name = model.model_name
            elif isinstance(model, str):
                model_name = model

        provider = _infer_provider(model_name)

        if input_tokens or output_tokens:
            meshai.track_usage(
                model_provider=provider,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_type="pydantic-ai",
            )
    except Exception:
        logger.debug("Failed to track Pydantic AI usage", exc_info=True)


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
    return "unknown"
