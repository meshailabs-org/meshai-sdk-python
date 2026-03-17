"""Tests for Transport methods (GET, POST, PATCH, DELETE)."""

import httpx
import respx

from meshai.config import MeshAIConfig
from meshai.transport import Transport

BASE = "https://api.meshai.dev"


def _transport() -> Transport:
    config = MeshAIConfig(api_key="msh_test1234abcdef5678", base_url=BASE)
    return Transport(config)


@respx.mock
def test_get_success():
    respx.get(f"{BASE}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"success": True, "data": []})
    )
    t = _transport()
    result = t.get("/agents")
    assert result["success"] is True
    t.close()


@respx.mock
def test_post_success():
    respx.post(f"{BASE}/api/v1/agents").mock(
        return_value=httpx.Response(201, json={"success": True, "data": {"id": "01"}})
    )
    t = _transport()
    result = t.post("/agents", {"name": "test"})
    assert result["success"] is True
    t.close()


@respx.mock
def test_patch_success():
    respx.patch(f"{BASE}/api/v1/agents/01").mock(
        return_value=httpx.Response(200, json={"success": True, "data": {"id": "01"}})
    )
    t = _transport()
    result = t.patch("/agents/01", {"name": "updated"})
    assert result["success"] is True
    t.close()


@respx.mock
def test_delete_204():
    respx.delete(f"{BASE}/api/v1/agents/01").mock(return_value=httpx.Response(204))
    t = _transport()
    result = t.delete("/agents/01")
    assert result["success"] is True
    t.close()


@respx.mock
def test_get_network_error():
    respx.get(f"{BASE}/api/v1/agents").mock(side_effect=httpx.ConnectError("refused"))
    t = _transport()
    result = t.get("/agents")
    assert result["success"] is False
    t.close()


@respx.mock
def test_post_retry_on_500():
    route = respx.post(f"{BASE}/api/v1/agents")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json={"success": True, "data": {}}),
    ]
    config = MeshAIConfig(
        api_key="msh_test1234abcdef5678",
        base_url=BASE,
        max_retries=2,
        retry_backoff_seconds=0.01,
    )
    t = Transport(config)
    result = t.post("/agents", {"name": "test"})
    assert result["success"] is True
    assert route.call_count == 2
    t.close()


@respx.mock
def test_post_non_json_response():
    respx.post(f"{BASE}/api/v1/agents").mock(
        return_value=httpx.Response(200, text="not json")
    )
    t = _transport()
    result = t.post("/agents", {"name": "test"})
    assert result["success"] is False
    assert "non-JSON" in result["error"]
    t.close()
