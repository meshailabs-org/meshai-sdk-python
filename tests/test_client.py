"""Tests for the MeshAI client."""

import httpx
import pytest
import respx

from meshai import MeshAI


@respx.mock
def test_register_agent():
    respx.post("https://api.meshai.dev/api/v1/agents").mock(
        return_value=httpx.Response(
            201,
            json={
                "success": True,
                "data": {
                    "id": "01TESTID",
                    "name": "test-agent",
                    "status": "unknown",
                    "framework": "crewai",
                    "model_provider": "openai",
                    "model_name": "gpt-4o",
                    "environment": "production",
                    "description": None,
                    "team_id": None,
                    "tags": None,
                    "metadata": None,
                    "last_heartbeat_at": None,
                    "registered_at": "2026-03-04T00:00:00Z",
                },
            },
        )
    )

    client = MeshAI(api_key="msh_test1234", agent_name="test-agent")
    result = client.register(framework="crewai", model_provider="openai", model_name="gpt-4o")

    assert result["success"] is True
    assert client.agent_id == "01TESTID"
    client.shutdown()


@respx.mock
def test_track_usage_without_register():
    """track_usage should warn and not crash if agent not registered."""
    client = MeshAI(api_key="msh_test1234", agent_name="test-agent")
    # Should not raise
    client.track_usage(
        model_provider="openai",
        model_name="gpt-4o",
        input_tokens=100,
        output_tokens=50,
    )
    client.shutdown()


def test_register_requires_name():
    client = MeshAI(api_key="msh_test1234")
    with pytest.raises(ValueError, match="agent_name is required"):
        client.register()
    client.shutdown()
