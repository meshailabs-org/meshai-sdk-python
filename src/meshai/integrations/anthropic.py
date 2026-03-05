"""Anthropic auto-tracking wrapper.

Usage:
    from meshai import MeshAI
    from meshai.integrations.anthropic import wrap_anthropic
    import anthropic

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="custom", model_provider="anthropic", model_name="claude-sonnet-4-6")

    tracked_client = wrap_anthropic(anthropic.Anthropic(), meshai=client)
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")


_WRAPPED_ATTR = "_meshai_wrapped"


def wrap_anthropic(anthropic_client: Any, meshai: MeshAI) -> Any:
    """Wrap an Anthropic client to auto-track token usage.

    Returns the same client with patched messages.create.
    """
    if getattr(anthropic_client, _WRAPPED_ATTR, False):
        logger.debug("Anthropic client already wrapped — skipping")
        return anthropic_client

    original_create = anthropic_client.messages.create

    @functools.wraps(original_create)
    def tracked_create(*args: Any, **kwargs: Any) -> Any:
        response = original_create(*args, **kwargs)

        try:
            if hasattr(response, "usage") and response.usage:
                model = getattr(response, "model", kwargs.get("model", "unknown"))
                meshai.track_usage(
                    model_provider="anthropic",
                    model_name=model,
                    input_tokens=response.usage.input_tokens or 0,
                    output_tokens=response.usage.output_tokens or 0,
                    request_type="messages.create",
                )
        except Exception:
            logger.debug("Failed to track Anthropic usage", exc_info=True)

        return response

    anthropic_client.messages.create = tracked_create
    anthropic_client._meshai_wrapped = True
    return anthropic_client
