"""AWS Bedrock auto-tracking wrapper.

Wraps boto3's bedrock-runtime client to auto-track token usage per-call.

Usage:
    from meshai import MeshAI
    from meshai.integrations.bedrock import wrap_bedrock
    import boto3

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="custom", model_provider="bedrock")

    bedrock = boto3.client("bedrock-runtime")
    tracked = wrap_bedrock(bedrock, meshai=client)
    response = tracked.converse(modelId="anthropic.claude-3-sonnet", ...)
"""

from __future__ import annotations

import functools
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")

_WRAPPED_ATTR = "_meshai_wrapped"


def wrap_bedrock(bedrock_client: Any, meshai: MeshAI) -> Any:
    """Wrap a boto3 bedrock-runtime client to auto-track token usage."""
    if getattr(bedrock_client, _WRAPPED_ATTR, False):
        return bedrock_client

    # Wrap converse() — the preferred Bedrock API
    if hasattr(bedrock_client, "converse"):
        original_converse = bedrock_client.converse

        @functools.wraps(original_converse)
        def tracked_converse(*args: Any, **kwargs: Any) -> Any:
            response = original_converse(*args, **kwargs)
            try:
                model_id = kwargs.get("modelId", args[0] if args else "unknown")
                usage = response.get("usage", {})
                meshai.track_usage(
                    model_provider="bedrock",
                    model_name=model_id,
                    input_tokens=usage.get("inputTokens", 0),
                    output_tokens=usage.get("outputTokens", 0),
                    request_type="converse",
                )
            except Exception:
                logger.debug("Failed to track Bedrock converse usage", exc_info=True)
            return response

        bedrock_client.converse = tracked_converse

    # Wrap invoke_model() — the lower-level API
    if hasattr(bedrock_client, "invoke_model"):
        original_invoke = bedrock_client.invoke_model

        @functools.wraps(original_invoke)
        def tracked_invoke(*args: Any, **kwargs: Any) -> Any:
            response = original_invoke(*args, **kwargs)
            try:
                model_id = kwargs.get("modelId", "unknown")
                body = response.get("body")
                if body:
                    result = json.loads(body.read())
                    usage = result.get("usage", {})
                    meshai.track_usage(
                        model_provider="bedrock",
                        model_name=model_id,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        request_type="invoke_model",
                    )
            except Exception:
                logger.debug("Failed to track Bedrock invoke usage", exc_info=True)
            return response

        bedrock_client.invoke_model = tracked_invoke

    bedrock_client._meshai_wrapped = True  # noqa: SLF001
    return bedrock_client
