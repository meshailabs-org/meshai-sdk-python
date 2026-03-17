"""Tests for SDK compliance methods (readiness, FRIA, incidents, billing)."""

import httpx
import respx

from meshai import MeshAI

BASE = "https://api.meshai.dev/api/v1"


def _client() -> MeshAI:
    return MeshAI(api_key="msh_test1234abcdef5678", agent_name="test")


def _ok(data):
    return httpx.Response(200, json={"success": True, "data": data, "error": None, "meta": None})


# --- Compliance ---


@respx.mock
def test_get_readiness_score():
    respx.get(f"{BASE}/compliance/readiness").mock(
        return_value=_ok({
            "score": 85,
            "max_score": 120,
            "components": [
                {"name": "Audit Trail Active", "score": 20, "max_score": 20, "status": "pass"},
                {"name": "Risk Classification", "score": 15, "max_score": 25, "status": "partial"},
            ],
        })
    )
    c = _client()
    result = c.get_readiness_score()
    assert result["data"]["score"] == 85
    assert len(result["data"]["components"]) == 2
    c.shutdown()


@respx.mock
def test_get_fria():
    respx.get(f"{BASE}/agents/01AGENT/fria").mock(
        return_value=_ok({
            "agent_id": "01AGENT",
            "agent_name": "test-agent",
            "risk_level": "high",
            "sections": [{"title": "1. System Description", "fields": []}],
        })
    )
    c = _client()
    result = c.get_fria("01AGENT")
    assert result["data"]["risk_level"] == "high"
    assert len(result["data"]["sections"]) == 1
    c.shutdown()


@respx.mock
def test_get_transparency_card():
    respx.get(f"{BASE}/agents/01AGENT/transparency-card").mock(
        return_value=_ok({
            "agent_id": "01AGENT",
            "agent_name": "test-agent",
            "identity": {"name": "test-agent", "framework": "crewai"},
            "risk_classification": {"level": "high"},
        })
    )
    c = _client()
    result = c.get_transparency_card("01AGENT")
    assert result["data"]["identity"]["framework"] == "crewai"
    c.shutdown()


# --- Incidents ---


@respx.mock
def test_create_incident():
    respx.post(f"{BASE}/incidents").mock(
        return_value=httpx.Response(
            201,
            json={
                "success": True,
                "data": {
                    "id": 1,
                    "agent_id": "01AGENT",
                    "title": "Data leak",
                    "severity": "critical",
                    "status": "reported",
                    "notification_deadline": "2026-04-01T00:00:00Z",
                },
                "error": None,
                "meta": None,
            },
        )
    )
    c = _client()
    result = c.create_incident(
        agent_id="01AGENT",
        title="Data leak",
        description="Agent exposed PII",
        severity="critical",
        reported_by="security-team",
        is_widespread=True,
    )
    assert result["data"]["severity"] == "critical"
    assert result["data"]["notification_deadline"] is not None
    c.shutdown()


@respx.mock
def test_list_incidents():
    respx.get(f"{BASE}/incidents").mock(
        return_value=_ok([{"id": 1, "status": "reported"}, {"id": 2, "status": "resolved"}])
    )
    c = _client()
    result = c.list_incidents(status="reported")
    assert len(result["data"]) == 2
    c.shutdown()


@respx.mock
def test_update_incident():
    respx.patch(f"{BASE}/incidents/1").mock(
        return_value=_ok({
            "id": 1,
            "status": "resolved",
            "root_cause": "Model hallucination",
            "authority_notified": True,
        })
    )
    c = _client()
    result = c.update_incident(
        1,
        status="resolved",
        root_cause="Model hallucination",
        authority_notified=True,
    )
    assert result["data"]["status"] == "resolved"
    assert result["data"]["authority_notified"] is True
    c.shutdown()


# --- Billing ---


@respx.mock
def test_get_billing_info():
    respx.get(f"{BASE}/billing").mock(
        return_value=_ok({
            "plan": "professional",
            "price_usd": 799,
            "max_agents": 100,
            "current_agents": 42,
            "at_limit": False,
        })
    )
    c = _client()
    result = c.get_billing_info()
    assert result["data"]["plan"] == "professional"
    assert result["data"]["at_limit"] is False
    c.shutdown()


# --- Error Handling ---


@respx.mock
def test_api_error_returns_error_dict():
    """SDK should never raise — returns error dict on failure."""
    respx.get(f"{BASE}/agents").mock(return_value=httpx.Response(500, json={"error": "Internal error"}))
    c = _client()
    result = c.list_agents()
    # Should not raise, should return parsed error
    assert "error" in result or result.get("success") is False
    c.shutdown()


@respx.mock
def test_network_error_returns_error_dict():
    """Network errors should be caught and returned as error dict."""
    respx.get(f"{BASE}/agents").mock(side_effect=httpx.ConnectError("Connection refused"))
    c = _client()
    result = c.list_agents()
    assert result["success"] is False
    assert "error" in result
    c.shutdown()
