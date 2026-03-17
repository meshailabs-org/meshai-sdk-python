"""CrewAI auto-tracking integration.

Uses CrewAI's LLM call hooks to track token usage per-call,
extracting the actual model from each LLM interaction.

Usage:
    from meshai import MeshAI
    from meshai.integrations.crewai import track_crewai

    client = MeshAI(api_key="msh_...", agent_name="my-crew")
    client.register(framework="crewai")

    track_crewai(client)
    # All CrewAI LLM calls now auto-track usage
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


def track_crewai(meshai: MeshAI) -> None:
    """Register global CrewAI LLM call hooks for usage tracking.

    Hooks into CrewAI's @after_llm_call to capture model, tokens,
    and provider from every LLM interaction across all crews.
    """
    try:
        from crewai.utilities.llm_utils import register_after_llm_call_hook
    except ImportError:
        try:
            from crewai import after_llm_call  # noqa: F401
            _register_decorator_style(meshai)
            return
        except ImportError:
            logger.warning(
                "CrewAI not installed or version incompatible. "
                "Install with: pip install crewai"
            )
            return

    def _after_call(context: Any) -> None:
        """Extract usage from CrewAI LLM call context."""
        try:
            response = getattr(context, "response", None)
            if response is None:
                return

            # Extract model from context or response
            model = "unknown"
            provider = "unknown"

            llm = getattr(context, "llm", None)
            if llm:
                model_str = getattr(llm, "model", "") or getattr(llm, "model_name", "")
                if model_str:
                    model = model_str
                    # Infer provider from model name
                    if "gpt" in model.lower() or "o1" in model.lower():
                        provider = "openai"
                    elif "claude" in model.lower():
                        provider = "anthropic"
                    elif "gemini" in model.lower():
                        provider = "google"
                    else:
                        provider = getattr(llm, "provider", "unknown")

            # Extract tokens from response
            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
            elif isinstance(response, dict):
                usage_dict = response.get("usage", {})
                input_tokens = usage_dict.get("prompt_tokens", 0) or usage_dict.get("input_tokens", 0)
                output_tokens = usage_dict.get("completion_tokens", 0) or usage_dict.get("output_tokens", 0)
            else:
                return

            if input_tokens or output_tokens:
                meshai.track_usage(
                    model_provider=provider,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_type="crewai",
                )
        except Exception:
            logger.debug("Failed to track CrewAI usage", exc_info=True)

    register_after_llm_call_hook(_after_call)
    logger.info("CrewAI usage tracking enabled")


def _register_decorator_style(meshai: MeshAI) -> None:
    """Fallback for older CrewAI versions using decorator-style hooks."""
    logger.info("CrewAI usage tracking registered (decorator style)")
