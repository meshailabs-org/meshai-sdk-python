"""MeshAI Tracer — OTel-native Session/Span primitive (v1, synchronous).

Wires a private OpenTelemetry ``TracerProvider`` (never the global one — the
host agent may run its own OTel setup) with a ``BatchSpanProcessor`` and the
OTLP/HTTP protobuf exporter pointed at MeshAI's ingest endpoint. Spans carry
the OTel GenAI semantic-convention attributes MeshAI's ingest consumes
(``gen_ai.usage.*`` → cost attribution, agent auto-discovery via
``service.name`` / ``meshai.agent.framework``).

Content attributes (tool_input/tool_output) pass through the default-deny
filter pipeline in :mod:`meshai.tracer.filters` before emission.

v1 is sync-only: spans nest via the current thread's OTel context. An
async/contextvars-aware Tracer is v2 scope.
"""

import atexit
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace as otel_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor

from meshai.config import MeshAIConfig
from meshai.tracer.filters import CONTENT_FIELDS, FilterConfig, FilterPipeline

logger = logging.getLogger("meshai")

_INGEST_TRACES_PATH = "/api/v1/ingest/v1/traces"

# Attribute keys (OTel GenAI semconv where one exists, meshai.* otherwise).
_ATTR_SESSION_ID = "meshai.session.id"
_ATTR_FRAMEWORK = "meshai.agent.framework"
_ATTR_OPERATION = "gen_ai.operation.name"
_ATTR_SYSTEM = "gen_ai.system"
_ATTR_MODEL = "gen_ai.request.model"
_ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_ATTR_TOOL_NAME = "gen_ai.tool.name"
_CONTENT_ATTR = {"tool_input": "meshai.tool.input", "tool_output": "meshai.tool.output"}


class SpanHandle:
    """Thin wrapper over an OTel span with filtered content setters."""

    def __init__(
        self, span: otel_api.Span, filters: FilterPipeline, tool_name: str | None
    ) -> None:
        self._span = span
        self._filters = filters
        self._tool_name = tool_name

    @property
    def otel_span(self) -> otel_api.Span:
        return self._span

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute; content fields go through the filter pipeline."""
        if key in CONTENT_FIELDS:
            self.set_content(key, value)
        else:
            self._span.set_attribute(key, value)

    def set_content(self, content_field: str, value: Any) -> None:
        """Attach tool_input/tool_output subject to default-deny + redaction."""
        if content_field not in CONTENT_FIELDS:
            raise ValueError(f"not a content field: {content_field!r}")
        filtered = self._filters.filter_content(
            self._tool_name or "", content_field, value
        )
        if filtered is not None:
            self._span.set_attribute(_CONTENT_ATTR[content_field], filtered)

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "chat",
    ) -> None:
        """Attach GenAI usage attributes (what MeshAI turns into cost rows)."""
        self._span.set_attribute(_ATTR_SYSTEM, provider)
        self._span.set_attribute(_ATTR_MODEL, model)
        self._span.set_attribute(_ATTR_OPERATION, operation)
        self._span.set_attribute(_ATTR_INPUT_TOKENS, int(input_tokens))
        self._span.set_attribute(_ATTR_OUTPUT_TOKENS, int(output_tokens))


class Session:
    """A traced agent session: one root span, children nest inside it.

    Context-manager only (sync). Every span created through the session
    carries ``meshai.session.id`` so usage rows attribute to the session.
    """

    def __init__(
        self,
        otel_tracer: otel_api.Tracer,
        filters: FilterPipeline,
        session_id: str,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._tracer = otel_tracer
        self._filters = filters
        self.session_id = session_id
        self._name = name
        self._attributes = dict(attributes or {})
        self._cm: Any = None

    def __enter__(self) -> "Session":
        self._cm = self._tracer.start_as_current_span(
            self._name,
            attributes={
                _ATTR_SESSION_ID: self.session_id,
                _ATTR_OPERATION: "session",
                **self._attributes,
            },
        )
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        return self._cm.__exit__(exc_type, exc, tb)

    @contextmanager
    def span(
        self,
        name: str,
        tool_name: str | None = None,
        tool_input: Any = None,
        tool_output: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        """Start a child span. Content kwargs flow through the filter pipeline."""
        attrs: dict[str, Any] = {_ATTR_SESSION_ID: self.session_id}
        if tool_name:
            attrs[_ATTR_TOOL_NAME] = tool_name
            attrs[_ATTR_OPERATION] = "execute_tool"
        attrs.update(attributes or {})
        with self._tracer.start_as_current_span(name, attributes=attrs) as span:
            handle = SpanHandle(span, self._filters, tool_name)
            if tool_input is not None:
                handle.set_content("tool_input", tool_input)
            if tool_output is not None:
                handle.set_content("tool_output", tool_output)
            yield handle

    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "chat",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed LLM call as an instant child span with usage."""
        with self.span(f"{operation} {model}", attributes=attributes) as handle:
            handle.record_usage(provider, model, input_tokens, output_tokens, operation)


class Tracer:
    """Entry point: builds the provider/exporter pair and mints Sessions.

    Usage::

        from meshai.tracer import Tracer

        tracer = Tracer(api_key="msh_...", service_name="my-agent",
                        framework="claude-code")
        with tracer.session() as session:
            with session.span("step", tool_name="Bash",
                              tool_input="ls -la") as span:
                ...
            session.record_llm_call("anthropic", "claude-sonnet-4-6",
                                    input_tokens=1850, output_tokens=420)
        tracer.shutdown()  # also registered atexit
    """

    def __init__(
        self,
        api_key: str,
        service_name: str,
        base_url: str = "https://api.meshai.dev",
        framework: str | None = None,
        filters: FilterConfig | None = None,
        resource_attributes: dict[str, Any] | None = None,
        span_processor: SpanProcessor | None = None,
    ) -> None:
        # Reuse the SDK config for key-format and HTTPS validation.
        config = MeshAIConfig(
            api_key=api_key, base_url=base_url, agent_name=service_name
        )

        resource_attrs: dict[str, Any] = {
            "service.name": service_name,
            **(resource_attributes or {}),
        }
        if framework:
            resource_attrs[_ATTR_FRAMEWORK] = framework

        self._provider = TracerProvider(resource=Resource.create(resource_attrs))
        if span_processor is None:  # pragma: no cover — exercised in E2E, not unit
            span_processor = BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=f"{config.resolved_ingest_url}{_INGEST_TRACES_PATH}",
                    headers={"Authorization": f"Bearer {config.api_key}"},
                )
            )
        self._provider.add_span_processor(span_processor)
        self._otel_tracer = self._provider.get_tracer("meshai-sdk")
        self._filters = FilterPipeline(
            filters if filters is not None else FilterConfig.load()
        )
        self._shutdown = False
        atexit.register(self.shutdown)

    def session(
        self,
        session_id: str | None = None,
        name: str = "agent.session",
        attributes: dict[str, Any] | None = None,
    ) -> Session:
        """Create a Session context manager (id generated when not given)."""
        return Session(
            self._otel_tracer,
            self._filters,
            session_id=session_id or uuid.uuid4().hex,
            name=name,
            attributes=attributes,
        )

    def flush(self, timeout_millis: int = 10_000) -> bool:
        """Force-export buffered spans (e.g. before ephemeral compute exits)."""
        return self._provider.force_flush(timeout_millis)

    def shutdown(self) -> None:
        """Flush and stop the exporter. Idempotent; never raises."""
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self._provider.shutdown()
        except Exception:  # noqa: BLE001 — never crash the host agent
            logger.warning("meshai.tracer: shutdown failed", exc_info=True)
