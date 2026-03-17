"""Microsoft Semantic Kernel auto-tracking integration.

Uses Semantic Kernel's filter system to track token usage per-call.

Usage:
    from meshai import MeshAI
    from meshai.integrations.semantic_kernel import MeshAIPromptFilter

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="semantic-kernel")

    kernel = sk.Kernel()
    kernel.add_filter("prompt_rendering", MeshAIPromptFilter(client))
    # Or use the convenience function:

    from meshai.integrations.semantic_kernel import track_semantic_kernel
    track_semantic_kernel(client, kernel)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


class MeshAIPromptFilter:
    """Semantic Kernel filter that tracks token usage after prompt execution.

    Works with Semantic Kernel's function invocation filter system.
    """

    def __init__(self, meshai: MeshAI) -> None:
        self._meshai = meshai

    async def on_function_invocation(
        self,
        context: Any,
        next_handler: Any,
    ) -> None:
        """Filter invoked around each function call."""
        await next_handler(context)

        try:
            result = context.result
            if result is None:
                return

            # Extract metadata from the result
            metadata = getattr(result, "metadata", {}) or {}
            usage = metadata.get("usage")

            if usage is None:
                return

            if hasattr(usage, "prompt_tokens"):
                input_tokens = usage.prompt_tokens or 0
                output_tokens = usage.completion_tokens or 0
            elif isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            else:
                return

            model = metadata.get("model", "unknown")
            provider = _infer_provider(model)

            if input_tokens or output_tokens:
                self._meshai.track_usage(
                    model_provider=provider,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_type="semantic-kernel",
                )
        except Exception:
            logger.debug("Failed to track Semantic Kernel usage", exc_info=True)


def track_semantic_kernel(meshai: MeshAI, kernel: Any) -> None:
    """Convenience function to add MeshAI tracking to a Semantic Kernel instance."""
    try:
        filter_instance = MeshAIPromptFilter(meshai)
        kernel.add_filter("function_invocation", filter_instance)
        logger.info("Semantic Kernel usage tracking enabled")
    except Exception:
        logger.warning(
            "Failed to add Semantic Kernel filter. "
            "Install with: pip install semantic-kernel"
        )


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
