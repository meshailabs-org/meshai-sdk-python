"""Google Gemini auto-tracking wrapper.

Wraps the Google Gen AI SDK to auto-track token usage per-call.

Usage:
    from meshai import MeshAI
    from meshai.integrations.gemini import wrap_gemini
    from google import genai

    client = MeshAI(api_key="msh_...", agent_name="my-agent")
    client.register(framework="custom", model_provider="google")

    genai_client = genai.Client(api_key="...")
    tracked = wrap_gemini(genai_client, meshai=client)
    response = tracked.models.generate_content(model="gemini-2.5-pro", contents="Hello")
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meshai.client import MeshAI

logger = logging.getLogger("meshai")

_WRAPPED_ATTR = "_meshai_wrapped"


def wrap_gemini(genai_client: Any, meshai: MeshAI) -> Any:
    """Wrap a Google Gen AI client to auto-track token usage.

    Supports both the new unified SDK (google.genai) and the legacy SDK.
    """
    if getattr(genai_client, _WRAPPED_ATTR, False):
        return genai_client

    # New unified SDK: client.models.generate_content()
    if hasattr(genai_client, "models") and hasattr(genai_client.models, "generate_content"):
        original = genai_client.models.generate_content

        @functools.wraps(original)
        def tracked_generate(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            try:
                model = kwargs.get("model", args[0] if args else "unknown")
                if isinstance(model, str):
                    model = model.replace("models/", "")

                usage = getattr(response, "usage_metadata", None)
                if usage:
                    meshai.track_usage(
                        model_provider="google",
                        model_name=model,
                        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                        request_type="generate_content",
                    )
            except Exception:
                logger.debug("Failed to track Gemini usage", exc_info=True)
            return response

        genai_client.models.generate_content = tracked_generate

    # Legacy SDK: genai.GenerativeModel("gemini-pro").generate_content()
    elif hasattr(genai_client, "generate_content"):
        original = genai_client.generate_content

        @functools.wraps(original)
        def tracked_legacy(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            try:
                model = getattr(genai_client, "model_name", "unknown")
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    meshai.track_usage(
                        model_provider="google",
                        model_name=model,
                        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                        request_type="generate_content",
                    )
            except Exception:
                logger.debug("Failed to track Gemini usage", exc_info=True)
            return response

        genai_client.generate_content = tracked_legacy

    genai_client._meshai_wrapped = True  # noqa: SLF001
    return genai_client
