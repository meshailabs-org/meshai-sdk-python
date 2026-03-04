"""Main MeshAI SDK client."""

import atexit
import logging
import threading
from typing import Any

from meshai.batcher import Batcher
from meshai.config import MeshAIConfig
from meshai.transport import Transport

logger = logging.getLogger("meshai")


class MeshAI:
    """MeshAI SDK client for agent telemetry.

    Usage:
        from meshai import MeshAI

        client = MeshAI(api_key="msh_...", agent_name="my-agent")
        client.register(framework="crewai", model_provider="openai", model_name="gpt-4o")
        client.start_heartbeat(interval_seconds=60)
        client.track_usage(model_provider="openai", model_name="gpt-4o",
                          input_tokens=1500, output_tokens=800)
    """

    def __init__(
        self,
        api_key: str,
        agent_name: str = "",
        base_url: str = "https://api.meshai.dev",
        environment: str = "production",
        **kwargs: Any,
    ) -> None:
        self._config = MeshAIConfig(
            api_key=api_key,
            base_url=base_url,
            agent_name=agent_name,
            environment=environment,
            **kwargs,
        )
        self._transport = Transport(self._config)
        self._agent_id: str | None = None
        self._heartbeat_thread: threading.Timer | None = None
        self._heartbeat_running = False

        # Batchers for telemetry
        self._heartbeat_batcher = Batcher(
            flush_fn=self._flush_heartbeats,
            batch_size=self._config.batch_size,
            flush_interval=self._config.flush_interval_seconds,
        )
        self._usage_batcher = Batcher(
            flush_fn=self._flush_usages,
            batch_size=self._config.batch_size,
            flush_interval=self._config.flush_interval_seconds,
        )

        atexit.register(self.shutdown)

    @property
    def agent_id(self) -> str | None:
        return self._agent_id

    def register(
        self,
        name: str | None = None,
        framework: str | None = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        team: str | None = None,
        description: str | None = None,
        tags: dict | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Register this agent with the MeshAI control plane.

        Returns the API response. Sets self.agent_id on success.
        """
        agent_name = name or self._config.agent_name
        if not agent_name:
            raise ValueError("agent_name is required (pass to constructor or register())")

        payload: dict[str, Any] = {
            "name": agent_name,
            "environment": self._config.environment,
        }
        if framework:
            payload["framework"] = framework
        if model_provider:
            payload["model_provider"] = model_provider
        if model_name:
            payload["model_name"] = model_name
        if team:
            payload["team_id"] = team
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        result = self._transport.post("/agents", payload)
        if result.get("success") and result.get("data"):
            self._agent_id = result["data"]["id"]
            logger.info("Agent registered: %s (id=%s)", agent_name, self._agent_id)
        return result

    def heartbeat(
        self,
        status: str = "healthy",
        latency_ms: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Send a single heartbeat. Buffered for batching."""
        if not self._agent_id:
            logger.warning("Cannot send heartbeat: agent not registered")
            return

        event: dict[str, Any] = {
            "agent_id": self._agent_id,
            "status": status,
        }
        if latency_ms is not None:
            event["latency_ms"] = latency_ms
        if metadata:
            event["metadata"] = metadata

        self._heartbeat_batcher.add(event)

    def start_heartbeat(
        self,
        interval_seconds: float | None = None,
        status: str = "healthy",
    ) -> None:
        """Start automatic background heartbeat."""
        if self._heartbeat_running:
            return

        interval = interval_seconds or self._config.heartbeat_interval_seconds
        self._heartbeat_running = True

        def _send_heartbeat() -> None:
            if not self._heartbeat_running:
                return
            self.heartbeat(status=status)
            self._heartbeat_thread = threading.Timer(interval, _send_heartbeat)
            self._heartbeat_thread.daemon = True
            self._heartbeat_thread.start()

        _send_heartbeat()
        logger.info("Background heartbeat started (interval=%ss)", interval)

    def stop_heartbeat(self) -> None:
        """Stop automatic background heartbeat."""
        self._heartbeat_running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.cancel()
            self._heartbeat_thread = None

    def track_usage(
        self,
        model_provider: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Track token usage. Buffered for batching."""
        if not self._agent_id:
            logger.warning("Cannot track usage: agent not registered")
            return

        event: dict[str, Any] = {
            "agent_id": self._agent_id,
            "model_provider": model_provider,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if request_type:
            event["request_type"] = request_type
        if session_id:
            event["session_id"] = session_id

        self._usage_batcher.add(event)

    def _flush_heartbeats(self, batch: list[dict[str, Any]]) -> None:
        if len(batch) == 1:
            self._transport.post("/telemetry/heartbeat", batch[0])
        else:
            self._transport.post("/telemetry/heartbeats", {"heartbeats": batch})

    def _flush_usages(self, batch: list[dict[str, Any]]) -> None:
        if len(batch) == 1:
            self._transport.post("/telemetry/usage", batch[0])
        else:
            self._transport.post("/telemetry/usages", {"usages": batch})

    def shutdown(self) -> None:
        """Flush all pending events and stop background threads."""
        self.stop_heartbeat()
        self._heartbeat_batcher.shutdown()
        self._usage_batcher.shutdown()
        self._transport.close()
