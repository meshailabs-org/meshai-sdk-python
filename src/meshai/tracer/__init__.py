"""MeshAI Tracer — OTel-native session/span telemetry (``pip install meshai-sdk[tracer]``)."""

try:
    import opentelemetry.sdk.trace  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "meshai.tracer requires the 'tracer' extra: pip install 'meshai-sdk[tracer]'"
    ) from exc

from meshai.tracer.filters import FilterConfig, FilterPipeline
from meshai.tracer.tracer import Session, SpanHandle, Tracer

__all__ = ["FilterConfig", "FilterPipeline", "Session", "SpanHandle", "Tracer"]
