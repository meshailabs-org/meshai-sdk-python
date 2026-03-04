"""OpenAI auto-tracking wrapper.

Usage:
    from meshai import MeshAI
    from meshai.integrations.openai import wrap_openai
    import openai

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="custom", model_provider="openai", model_name="gpt-4o")

    tracked_client = wrap_openai(openai.OpenAI(), meshai=client)
    # All chat completions now auto-track token usage
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


def wrap_openai(openai_client: Any, meshai: MeshAI) -> Any:
    """Wrap an OpenAI client to auto-track token usage.

    Returns the same client with patched chat.completions.create.
    """
    original_create = openai_client.chat.completions.create

    @functools.wraps(original_create)
    def tracked_create(*args: Any, **kwargs: Any) -> Any:
        response = original_create(*args, **kwargs)

        try:
            if hasattr(response, "usage") and response.usage:
                model = getattr(response, "model", kwargs.get("model", "unknown"))
                meshai.track_usage(
                    model_provider="openai",
                    model_name=model,
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                    request_type="chat.completions",
                )
        except Exception:
            logger.debug("Failed to track OpenAI usage", exc_info=True)

        return response

    openai_client.chat.completions.create = tracked_create
    return openai_client
