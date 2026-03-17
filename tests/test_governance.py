"""Tests for SDK governance methods (audit, risk, policies, approvals)."""

import httpx
import respx

from meshai import MeshAI

BASE = "https://api.meshai.dev/api/v1"


def _client() -> MeshAI:
    return MeshAI(api_key="msh_test1234abcdef5678", agent_name="test")


def _ok(data):
    return httpx.Response(200, json={"success": True, "data": data, "error": None, "meta": None})


# --- Audit Trail ---


@respx.mock
def test_list_audit_events():
    respx.get(f"{BASE}/audit-trail").mock(
        return_value=_ok([{"id": 1, "event_type": "agent.registered"}])
    )
    c = _client()
    result = c.list_audit_events(event_type="agent.registered")
    assert result["data"][0]["event_type"] == "agent.registered"
    c.shutdown()


@respx.mock
def test_get_audit_event():
    respx.get(f"{BASE}/audit-trail/1").mock(
        return_value=_ok({"id": 1, "event_type": "agent.registered", "description": "Agent created"})
    )
    c = _client()
    result = c.get_audit_event(1)
    assert result["data"]["id"] == 1
    c.shutdown()


# --- Risk Classification ---


@respx.mock
def test_classify_risk():
    respx.post(f"{BASE}/agents/01AGENT/risk-classification").mock(
        return_value=httpx.Response(
            201,
            json={"success": True, "data": {"id": 1, "risk_level": "high"}, "error": None, "meta": None},
        )
    )
    c = _client()
    result = c.classify_risk(
        agent_id="01AGENT",
        risk_level="high",
        justification="Handles PII in production environment",
        assessed_by="security-team",
    )
    assert result["data"]["risk_level"] == "high"
    c.shutdown()


@respx.mock
def test_get_risk_classification():
    respx.get(f"{BASE}/agents/01AGENT/risk-classification").mock(
        return_value=_ok({"agent_id": "01AGENT", "risk_level": "high"})
    )
    c = _client()
    result = c.get_risk_classification("01AGENT")
    assert result["data"]["risk_level"] == "high"
    c.shutdown()


@respx.mock
def test_get_risk_suggestion():
    respx.get(f"{BASE}/agents/01AGENT/risk-suggestion").mock(
        return_value=_ok({
            "suggested_level": "high",
            "confidence": 0.72,
            "factors": ["Handles PII", "Production"],
        })
    )
    c = _client()
    result = c.get_risk_suggestion("01AGENT")
    assert result["data"]["suggested_level"] == "high"
    assert result["data"]["confidence"] > 0.5
    c.shutdown()


@respx.mock
def test_list_risk_classifications():
    respx.get(f"{BASE}/risk-classifications").mock(
        return_value=_ok([{"agent_id": "01A", "risk_level": "minimal"}])
    )
    c = _client()
    result = c.list_risk_classifications(risk_level="minimal")
    assert len(result["data"]) == 1
    c.shutdown()


# --- Policies ---


@respx.mock
def test_create_policy():
    respx.post(f"{BASE}/policies").mock(
        return_value=httpx.Response(
            201,
            json={
                "success": True,
                "data": {"id": 1, "name": "GPT only", "policy_type": "model_allowlist"},
                "error": None,
                "meta": None,
            },
        )
    )
    c = _client()
    result = c.create_policy(
        name="GPT only",
        policy_type="model_allowlist",
        rules={"allowed_models": ["gpt-4o"]},
        conditions={"environments": ["production"]},
    )
    assert result["data"]["name"] == "GPT only"
    c.shutdown()


@respx.mock
def test_list_policies():
    respx.get(f"{BASE}/policies").mock(
        return_value=_ok([{"id": 1, "name": "GPT only", "enabled": True}])
    )
    c = _client()
    result = c.list_policies(enabled=True)
    assert len(result["data"]) == 1
    c.shutdown()


@respx.mock
def test_get_policy():
    respx.get(f"{BASE}/policies/1").mock(
        return_value=_ok({"id": 1, "name": "GPT only", "rules": {"allowed_models": ["gpt-4o"]}})
    )
    c = _client()
    result = c.get_policy(1)
    assert result["data"]["id"] == 1
    c.shutdown()


@respx.mock
def test_update_policy():
    respx.patch(f"{BASE}/policies/1").mock(
        return_value=_ok({"id": 1, "enabled": False})
    )
    c = _client()
    result = c.update_policy(1, enabled=False)
    assert result["data"]["enabled"] is False
    c.shutdown()


@respx.mock
def test_delete_policy():
    respx.delete(f"{BASE}/policies/1").mock(return_value=httpx.Response(204))
    c = _client()
    result = c.delete_policy(1)
    assert result["success"] is True
    c.shutdown()


@respx.mock
def test_evaluate_policies():
    respx.post(f"{BASE}/policies/evaluate").mock(
        return_value=_ok([
            {"policy_id": 1, "policy_name": "GPT only", "result": "pass", "reason": None},
            {"policy_id": 2, "policy_name": "No Bedrock", "result": "pass", "reason": None},
        ])
    )
    c = _client()
    result = c.evaluate_policies(agent_id="01AGENT", provider="openai", model="gpt-4o")
    assert len(result["data"]) == 2
    assert all(r["result"] == "pass" for r in result["data"])
    c.shutdown()


# --- Approvals ---


@respx.mock
def test_list_approvals():
    respx.get(f"{BASE}/approvals").mock(
        return_value=_ok([{"id": 1, "status": "pending", "agent_id": "01AGENT"}])
    )
    c = _client()
    result = c.list_approvals(status="pending")
    assert result["data"][0]["status"] == "pending"
    c.shutdown()


@respx.mock
def test_get_pending_count():
    respx.get(f"{BASE}/approvals/pending/count").mock(
        return_value=_ok({"count": 3})
    )
    c = _client()
    result = c.get_pending_count()
    assert result["data"]["count"] == 3
    c.shutdown()


@respx.mock
def test_decide_approval():
    respx.post(f"{BASE}/approvals/1/decide").mock(
        return_value=_ok({"request": {"id": 1, "status": "approved"}, "decision": {"decision": "approved"}})
    )
    c = _client()
    result = c.decide_approval(request_id=1, decision="approved", reviewer_id="admin", reason="Looks good")
    assert result["data"]["request"]["status"] == "approved"
    c.shutdown()
