"""Tests for meshai.tracer.Tracer — spans verified via in-memory export."""

import json

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
    with tracer.session(session_id="sess-1") as session, session.span("step-1"):
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


def test_record_llm_call_drops_inputs_by_default(exporter):
    tracer = _tracer(exporter)
    with tracer.session() as session:
        session.record_llm_call(
            "anthropic",
            "claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            input_messages=[{"role": "user", "content": "private prompt"}],
            system_instructions=[{"type": "text", "content": "private system"}],
            retrieval_query="private retrieval query",
        )
    attributes = exporter.get_finished_spans()[0].attributes
    assert "gen_ai.input.messages" not in attributes
    assert "gen_ai.system_instructions" not in attributes
    assert "gen_ai.retrieval.query.text" not in attributes


def test_record_llm_call_emits_redacted_standard_inputs_when_opted_in(exporter):
    tracer = _tracer(exporter, capture_inputs=True)
    with tracer.session() as session:
        session.record_llm_call(
            "anthropic",
            "claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            input_messages=[{"role": "user", "content": f"key sk-ant-{'a' * 24}"}],
            system_instructions=[{"type": "text", "content": "Be concise"}],
            retrieval_query="find account test@example.com",
        )
    attributes = exporter.get_finished_spans()[0].attributes
    messages = json.loads(attributes["gen_ai.input.messages"])
    assert messages[0]["role"] == "user"
    assert "sk-ant-" not in messages[0]["content"]
    assert "[REDACTED:anthropic_api_key]" in messages[0]["content"]
    assert json.loads(attributes["gen_ai.system_instructions"])[0]["content"] == "Be concise"
    assert attributes["gen_ai.retrieval.query.text"] == "find account test@example.com"


def test_input_serialization_failure_never_breaks_host_agent(exporter):
    cyclic = []
    cyclic.append(cyclic)
    tracer = _tracer(exporter, capture_inputs=True)
    with tracer.session() as session:
        session.record_llm_call(
            "openai",
            "gpt-4o",
            input_tokens=1,
            output_tokens=1,
            input_messages=cyclic,
        )
    attributes = exporter.get_finished_spans()[0].attributes
    assert "gen_ai.input.messages" not in attributes


def test_generic_attribute_path_cannot_bypass_input_opt_in(exporter):
    tracer = _tracer(exporter)
    with tracer.session() as session, session.span(
        "llm",
        attributes={
            "gen_ai.input.messages": [{"role": "user", "content": "private"}],
            "gen_ai.retrieval.query.text": "private query",
            "custom.structural": "kept",
        },
    ):
        pass
    attributes = exporter.get_finished_spans()[0].attributes
    assert "gen_ai.input.messages" not in attributes
    assert "gen_ai.retrieval.query.text" not in attributes
    assert attributes["custom.structural"] == "kept"


def test_set_attribute_applies_input_opt_in_and_redaction(exporter):
    tracer = _tracer(exporter, capture_inputs=True)
    with tracer.session() as session, session.span("llm") as handle:
        handle.set_attribute(
            "gen_ai.input.messages",
            [{"role": "user", "content": f"sk-ant-{'b' * 24}"}],
        )
    messages = json.loads(exporter.get_finished_spans()[0].attributes["gen_ai.input.messages"])
    assert messages[0]["content"] == "[REDACTED:anthropic_api_key]"


def test_tool_content_denied_by_default(exporter):
    tracer = _tracer(exporter)
    with tracer.session() as session, session.span(
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
    with tracer.session() as session, session.span(
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
    with pytest.raises(RuntimeError), tracer.session():
        raise RuntimeError("agent crashed")
    (root,) = exporter.get_finished_spans()
    assert root.name == "agent.session"
    assert not root.status.is_ok


def test_set_attribute_routes_content_fields_through_filters(exporter):
    tracer = _tracer(exporter)
    with tracer.session() as session, session.span("tool", tool_name="Read") as handle:
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
