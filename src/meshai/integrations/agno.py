"""Agno (formerly Phidata) auto-tracking integration.

Wraps Agno's Agent to track token usage from each run.

Usage:
    from meshai import MeshAI
    from meshai.integrations.agno import track_agno

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="agno")

    track_agno(client)
    # Agno agents now auto-track usage via monkey-patching
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


def track_agno(meshai: MeshAI) -> None:
    """Enable usage tracking for Agno agents.

    Patches agno.Agent.run to capture token usage after each execution.
    """
    try:
        from agno.agent import Agent
    except ImportError:
        logger.warning("Agno not installed. Install with: pip install agno")
        return

    if getattr(Agent, "_meshai_tracked", False):
        return

    original_run = Agent.run

    @functools.wraps(original_run)
    def tracked_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        response = original_run(self, *args, **kwargs)
        try:
            # Extract model from agent config
            model = getattr(self, "model", None)
            model_name = "unknown"
            provider = "unknown"

            if model:
                if hasattr(model, "id"):
                    model_name = model.id
                elif isinstance(model, str):
                    model_name = model

                if hasattr(model, "provider"):
                    provider = model.provider
                else:
                    provider = _infer_provider(model_name)

            # Extract usage from response
            if hasattr(response, "metrics") and response.metrics:
                metrics = response.metrics
                input_tokens = getattr(metrics, "input_tokens", 0) or 0
                output_tokens = getattr(metrics, "output_tokens", 0) or 0
            elif hasattr(response, "usage"):
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
            else:
                input_tokens = 0
                output_tokens = 0

            if input_tokens or output_tokens:
                meshai.track_usage(
                    model_provider=provider,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_type="agno",
                )
        except Exception:
            logger.debug("Failed to track Agno usage", exc_info=True)
        return response

    Agent.run = tracked_run
    Agent._meshai_tracked = True  # noqa: SLF001
    logger.info("Agno usage tracking enabled")


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
