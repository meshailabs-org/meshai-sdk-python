"""Tests for SDK query methods (agents, cost, anomalies)."""

import httpx
import respx

from meshai import MeshAI

BASE = "https://api.meshai.dev/api/v1"


def _client() -> MeshAI:
    return MeshAI(api_key="msh_test1234abcdef5678", agent_name="test")


def _ok(data):
    return httpx.Response(200, json={"success": True, "data": data, "error": None, "meta": None})


# --- Agent Queries ---


@respx.mock
def test_list_agents():
    respx.get(f"{BASE}/agents").mock(return_value=_ok([{"id": "01A", "name": "agent-1"}]))
    c = _client()
    result = c.list_agents(status="healthy", page=1)
    assert result["success"] is True
    assert len(result["data"]) == 1
    c.shutdown()


@respx.mock
def test_get_agent():
    respx.get(f"{BASE}/agents/01AGENT").mock(return_value=_ok({"id": "01AGENT", "name": "test"}))
    c = _client()
    result = c.get_agent("01AGENT")
    assert result["data"]["id"] == "01AGENT"
    c.shutdown()


@respx.mock
def test_update_agent():
    respx.patch(f"{BASE}/agents/01AGENT").mock(
        return_value=_ok({"id": "01AGENT", "description": "updated"})
    )
    c = _client()
    result = c.update_agent("01AGENT", description="updated")
    assert result["data"]["description"] == "updated"
    c.shutdown()


@respx.mock
def test_delete_agent():
    respx.delete(f"{BASE}/agents/01AGENT").mock(return_value=httpx.Response(204))
    c = _client()
    result = c.delete_agent("01AGENT")
    assert result["success"] is True
    c.shutdown()


# --- Cost Intelligence ---


@respx.mock
def test_get_cost_summary():
    respx.get(f"{BASE}/cost/summary").mock(
        return_value=_ok({"total_cost_usd": "45.50", "total_tokens": 100000})
    )
    c = _client()
    result = c.get_cost_summary(start="2026-03-01T00:00:00Z")
    assert result["data"]["total_cost_usd"] == "45.50"
    c.shutdown()


@respx.mock
def test_get_cost_by_agent():
    respx.get(f"{BASE}/cost/by-agent").mock(
        return_value=_ok([{"label": "agent-1", "cost_usd": "20.00"}])
    )
    c = _client()
    result = c.get_cost_by_agent()
    assert len(result["data"]) == 1
    c.shutdown()


@respx.mock
def test_get_cost_by_model():
    respx.get(f"{BASE}/cost/by-model").mock(
        return_value=_ok([{"label": "gpt-4o", "cost_usd": "30.00"}])
    )
    c = _client()
    result = c.get_cost_by_model()
    assert result["data"][0]["label"] == "gpt-4o"
    c.shutdown()


# --- Anomaly Detection ---


@respx.mock
def test_list_anomalies():
    respx.get(f"{BASE}/anomalies").mock(
        return_value=_ok([{"id": 1, "anomaly_type": "cost_spike", "severity": "high"}])
    )
    c = _client()
    result = c.list_anomalies(severity="high")
    assert result["data"][0]["severity"] == "high"
    c.shutdown()


@respx.mock
def test_get_anomaly():
    respx.get(f"{BASE}/anomalies/42").mock(
        return_value=_ok({"id": 42, "title": "Cost spike detected"})
    )
    c = _client()
    result = c.get_anomaly(42)
    assert result["data"]["id"] == 42
    c.shutdown()


@respx.mock
def test_acknowledge_anomaly():
    respx.post(f"{BASE}/anomalies/42/acknowledge").mock(
        return_value=_ok({"id": 42, "acknowledged_at": "2026-03-17T00:00:00Z"})
    )
    c = _client()
    result = c.acknowledge_anomaly(42)
    assert result["data"]["acknowledged_at"] is not None
    c.shutdown()


@respx.mock
def test_resolve_anomaly():
    respx.post(f"{BASE}/anomalies/42/resolve").mock(
        return_value=_ok({"id": 42, "resolved_at": "2026-03-17T00:00:00Z"})
    )
    c = _client()
    result = c.resolve_anomaly(42)
    assert result["data"]["resolved_at"] is not None
    c.shutdown()


@respx.mock
def test_get_anomaly_summary():
    respx.get(f"{BASE}/anomalies/summary").mock(
        return_value=_ok({"total_active": 5, "by_type": {"cost_spike": 3}})
    )
    c = _client()
    result = c.get_anomaly_summary()
    assert result["data"]["total_active"] == 5
    c.shutdown()
