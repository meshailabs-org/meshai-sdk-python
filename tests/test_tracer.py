"""Tests for meshai.tracer.Tracer — spans verified via in-memory export."""

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from meshai.tracer import FilterConfig, Tracer

API_KEY = "msh_" + "x" * 20


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def _tracer(exporter: InMemorySpanExporter, **kwargs) -> Tracer:
    kwargs.setdefault("filters", FilterConfig())  # deny-all, skip disk config
    return Tracer(
        api_key=API_KEY,
        service_name="test-agent",
        framework="claude-code",
        span_processor=SimpleSpanProcessor(exporter),
        **kwargs,
    )


def test_resource_carries_service_name_and_framework(exporter):
    tracer = _tracer(exporter)
    with tracer.session():
        pass
    (span,) = exporter.get_finished_spans()
    assert span.resource.attributes["service.name"] == "test-agent"
    assert span.resource.attributes["meshai.agent.framework"] == "claude-code"


def test_session_root_and_child_linkage(exporter):
    tracer = _tracer(exporter)
    with tracer.session(session_id="sess-1") as session:
        with session.span("step-1"):
            pass
    child, root = exporter.get_finished_spans()
    assert root.name == "agent.session"
    assert root.parent is None
    assert child.parent.span_id == root.context.span_id
    assert child.context.trace_id == root.context.trace_id
    # Both carry the session id for downstream attribution.
    assert root.attributes["meshai.session.id"] == "sess-1"
    assert child.attributes["meshai.session.id"] == "sess-1"


def test_record_llm_call_emits_genai_usage_attributes(exporter):
    tracer = _tracer(exporter)
    with tracer.session() as session:
        session.record_llm_call(
            "anthropic", "claude-sonnet-4-6", input_tokens=1850, output_tokens=420
        )
    llm_span = exporter.get_finished_spans()[0]
    assert llm_span.name == "chat claude-sonnet-4-6"
    assert llm_span.attributes["gen_ai.system"] == "anthropic"
    assert llm_span.attributes["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert llm_span.attributes["gen_ai.operation.name"] == "chat"
    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 1850
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 420


def test_tool_content_denied_by_default(exporter):
    tracer = _tracer(exporter)
    with tracer.session() as session:
        with session.span(
            "tool", tool_name="Bash", tool_input="cat /etc/passwd"
        ) as handle:
            handle.set_content("tool_output", "root:x:0:0")
    tool_span = exporter.get_finished_spans()[0]
    # Structural metadata flows; content does not.
    assert tool_span.attributes["gen_ai.tool.name"] == "Bash"
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert "meshai.tool.input" not in tool_span.attributes
    assert "meshai.tool.output" not in tool_span.attributes


def test_allowlisted_tool_content_is_emitted_and_redacted(exporter):
    config = FilterConfig(allow={"Bash": frozenset({"tool_input"})})
    tracer = _tracer(exporter, filters=config)
    with tracer.session() as session:
        with session.span(
            "tool",
            tool_name="Bash",
            tool_input=f"export ANTHROPIC_API_KEY=sk-ant-{'a' * 24}",
            tool_output="still denied",
        ):
            pass
    tool_span = exporter.get_finished_spans()[0]
    emitted = tool_span.attributes["meshai.tool.input"]
    assert "sk-ant-" not in emitted
    assert "[REDACTED:anthropic_api_key]" in emitted
    assert "meshai.tool.output" not in tool_span.attributes  # not allowlisted


def test_session_ends_root_span_on_exception(exporter):
    tracer = _tracer(exporter)
    with pytest.raises(RuntimeError):
        with tracer.session():
            raise RuntimeError("agent crashed")
    (root,) = exporter.get_finished_spans()
    assert root.name == "agent.session"
    assert not root.status.is_ok


def test_set_attribute_routes_content_fields_through_filters(exporter):
    tracer = _tracer(exporter)
    with tracer.session() as session:
        with session.span("tool", tool_name="Read") as handle:
            handle.set_attribute("tool_input", "secret file body")  # denied
            handle.set_attribute("custom.key", "structural")  # plain
    span = exporter.get_finished_spans()[0]
    assert "meshai.tool.input" not in span.attributes
    assert span.attributes["custom.key"] == "structural"


def test_shutdown_is_idempotent(exporter):
    tracer = _tracer(exporter)
    tracer.shutdown()
    tracer.shutdown()  # must not raise


def test_invalid_api_key_rejected(exporter):
    with pytest.raises(ValueError):
        Tracer(api_key="not-a-key", service_name="x")


def test_non_https_base_url_rejected(exporter):
    with pytest.raises(ValueError):
        Tracer(api_key=API_KEY, service_name="x", base_url="http://api.meshai.dev")


class TestIngestGatewayRouting:
    """Telemetry goes to the collector gateway; the API host does not change.

    The gateway absorbs large exports and re-chunks them before the API sees
    them, so a batch that the API would reject is delivered instead.
    """

    def test_default_config_sends_telemetry_to_the_gateway(self):
        from meshai.config import MeshAIConfig

        cfg = MeshAIConfig(api_key="msh_" + "x" * 20)
        assert cfg.resolved_ingest_url == "https://ingest.meshai.dev"
        # The API host is unchanged: registry, cost and heartbeat still go there.
        assert cfg.base_url == "https://api.meshai.dev"

    def test_explicit_ingest_url_wins(self):
        from meshai.config import MeshAIConfig

        cfg = MeshAIConfig(api_key="msh_" + "x" * 20, ingest_url="https://custom.example")
        assert cfg.resolved_ingest_url == "https://custom.example"

    def test_self_hosted_base_url_keeps_its_own_telemetry(self):
        """Regression guard. If a self-hoster's base_url did not carry through,
        their spans would be shipped to MeshAI's gateway instead of staying on
        their own infrastructure - a data-egress bug, not a routing detail."""
        from meshai.config import MeshAIConfig

        cfg = MeshAIConfig(api_key="msh_" + "x" * 20, base_url="https://meshai.internal.corp")
        assert cfg.resolved_ingest_url == "https://meshai.internal.corp"

    def test_localhost_development_stays_local(self):
        from meshai.config import MeshAIConfig

        cfg = MeshAIConfig(api_key="msh_" + "x" * 20, base_url="http://localhost:8080")
        assert cfg.resolved_ingest_url == "http://localhost:8080"
