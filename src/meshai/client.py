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
        self._heartbeat_lock = threading.Lock()

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
        with self._heartbeat_lock:
            if self._heartbeat_running:
                return

            interval = interval_seconds or self._config.heartbeat_interval_seconds
            self._heartbeat_running = True

        def _send_heartbeat() -> None:
            with self._heartbeat_lock:
                if not self._heartbeat_running:
                    return
            self.heartbeat(status=status)
            with self._heartbeat_lock:
                if self._heartbeat_running:
                    self._heartbeat_thread = threading.Timer(interval, _send_heartbeat)
                    self._heartbeat_thread.daemon = True
                    self._heartbeat_thread.start()

        _send_heartbeat()
        logger.info("Background heartbeat started (interval=%ss)", interval)

    def stop_heartbeat(self) -> None:
        """Stop automatic background heartbeat."""
        with self._heartbeat_lock:
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
        if not model_provider or not model_name:
            logger.warning("track_usage: model_provider and model_name are required")
            return
        if input_tokens < 0 or output_tokens < 0:
            logger.warning("track_usage: token counts must be non-negative")
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

    # --- Agent Queries ---

    def list_agents(self, **params: Any) -> dict[str, Any]:
        """List registered agents. Supports: page, limit, status, framework, team_id, search."""
        return self._transport.get("/agents", params=params or None)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        """Get a single agent by ID."""
        return self._transport.get(f"/agents/{agent_id}")

    def update_agent(self, agent_id: str, **fields: Any) -> dict[str, Any]:
        """Update agent fields (name, description, framework, etc.)."""
        return self._transport.patch(f"/agents/{agent_id}", {k: v for k, v in fields.items() if v is not None})

    def delete_agent(self, agent_id: str) -> dict[str, Any]:
        """Soft-delete an agent."""
        return self._transport.delete(f"/agents/{agent_id}")

    # --- Cost Intelligence ---

    def get_cost_summary(self, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """Get cost summary for the tenant."""
        params: dict[str, str] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._transport.get("/cost/summary", params=params or None)

    def get_cost_by_agent(self, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """Get cost breakdown by agent."""
        params: dict[str, str] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._transport.get("/cost/by-agent", params=params or None)

    def get_cost_by_model(self, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """Get cost breakdown by model."""
        params: dict[str, str] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._transport.get("/cost/by-model", params=params or None)

    # --- Anomaly Detection ---

    def list_anomalies(self, **params: Any) -> dict[str, Any]:
        """List active anomalies. Supports: anomaly_type, severity, page, limit."""
        return self._transport.get("/anomalies", params=params or None)

    def get_anomaly(self, event_id: int) -> dict[str, Any]:
        """Get a single anomaly event."""
        return self._transport.get(f"/anomalies/{event_id}")

    def acknowledge_anomaly(self, event_id: int) -> dict[str, Any]:
        """Acknowledge an anomaly event."""
        return self._transport.post(f"/anomalies/{event_id}/acknowledge", {})

    def resolve_anomaly(self, event_id: int) -> dict[str, Any]:
        """Resolve an anomaly event."""
        return self._transport.post(f"/anomalies/{event_id}/resolve", {})

    def get_anomaly_summary(self, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """Get anomaly summary (counts by type/severity)."""
        params: dict[str, str] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._transport.get("/anomalies/summary", params=params or None)

    # --- Governance: Audit Trail ---

    def list_audit_events(self, **params: Any) -> dict[str, Any]:
        """List audit trail events. Supports: event_type, actor_type, resource_type, start, end, page, limit."""
        return self._transport.get("/audit-trail", params=params or None)

    def get_audit_event(self, event_id: int) -> dict[str, Any]:
        """Get a single audit event."""
        return self._transport.get(f"/audit-trail/{event_id}")

    # --- Governance: Risk Classification ---

    def classify_risk(
        self,
        agent_id: str,
        risk_level: str,
        justification: str,
        assessed_by: str,
        domain_tags: list[str] | None = None,
        ai_act_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Classify an agent's risk level (minimal/limited/high/unacceptable)."""
        payload: dict[str, Any] = {
            "risk_level": risk_level,
            "justification": justification,
            "assessed_by": assessed_by,
        }
        if domain_tags:
            payload["domain_tags"] = domain_tags
        if ai_act_categories:
            payload["ai_act_categories"] = ai_act_categories
        return self._transport.post(f"/agents/{agent_id}/risk-classification", payload)

    def get_risk_classification(self, agent_id: str) -> dict[str, Any]:
        """Get current risk classification for an agent."""
        return self._transport.get(f"/agents/{agent_id}/risk-classification")

    def get_risk_suggestion(self, agent_id: str) -> dict[str, Any]:
        """Get AI-assisted risk level suggestion based on agent metadata."""
        return self._transport.get(f"/agents/{agent_id}/risk-suggestion")

    def list_risk_classifications(self, **params: Any) -> dict[str, Any]:
        """List all risk classifications. Supports: risk_level, page, limit."""
        return self._transport.get("/risk-classifications", params=params or None)

    # --- Governance: Policies ---

    def create_policy(
        self,
        name: str,
        policy_type: str,
        rules: dict[str, Any],
        enabled: bool = True,
        priority: int = 100,
        conditions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a governance policy."""
        payload: dict[str, Any] = {
            "name": name,
            "policy_type": policy_type,
            "rules": rules,
            "enabled": enabled,
            "priority": priority,
        }
        if conditions:
            payload["conditions"] = conditions
        return self._transport.post("/policies", payload)

    def list_policies(self, **params: Any) -> dict[str, Any]:
        """List governance policies. Supports: policy_type, enabled, page, limit."""
        return self._transport.get("/policies", params=params or None)

    def get_policy(self, policy_id: int) -> dict[str, Any]:
        """Get a single policy."""
        return self._transport.get(f"/policies/{policy_id}")

    def update_policy(self, policy_id: int, **fields: Any) -> dict[str, Any]:
        """Update policy fields (name, enabled, priority, conditions, rules)."""
        return self._transport.patch(f"/policies/{policy_id}", {k: v for k, v in fields.items() if v is not None})

    def delete_policy(self, policy_id: int) -> dict[str, Any]:
        """Delete a governance policy."""
        return self._transport.delete(f"/policies/{policy_id}")

    def evaluate_policies(
        self,
        agent_id: str,
        provider: str,
        model: str,
        team_id: str | None = None,
        environment: str = "production",
    ) -> dict[str, Any]:
        """Dry-run evaluate all policies against a request context."""
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "provider": provider,
            "model": model,
            "environment": environment,
        }
        if team_id:
            payload["team_id"] = team_id
        return self._transport.post("/policies/evaluate", payload)

    # --- Governance: Approvals ---

    def list_approvals(self, **params: Any) -> dict[str, Any]:
        """List approval requests. Supports: status, agent_id, page, limit."""
        return self._transport.get("/approvals", params=params or None)

    def get_pending_count(self) -> dict[str, Any]:
        """Get count of pending approval requests."""
        return self._transport.get("/approvals/pending/count")

    def decide_approval(
        self,
        request_id: int,
        decision: str,
        reviewer_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Approve or deny an approval request."""
        payload: dict[str, Any] = {
            "decision": decision,
            "reviewer_id": reviewer_id,
        }
        if reason:
            payload["reason"] = reason
        return self._transport.post(f"/approvals/{request_id}/decide", payload)

    # --- Compliance ---

    def get_readiness_score(self) -> dict[str, Any]:
        """Get EU AI Act compliance readiness score (0-120)."""
        return self._transport.get("/compliance/readiness")

    def get_fria(self, agent_id: str) -> dict[str, Any]:
        """Get auto-generated FRIA template for an agent."""
        return self._transport.get(f"/agents/{agent_id}/fria")

    def get_transparency_card(self, agent_id: str) -> dict[str, Any]:
        """Get auto-generated transparency card for an agent."""
        return self._transport.get(f"/agents/{agent_id}/transparency-card")

    # --- Incidents ---

    def create_incident(
        self,
        agent_id: str,
        title: str,
        description: str,
        severity: str,
        reported_by: str,
        is_widespread: bool = False,
        anomaly_event_id: int | None = None,
    ) -> dict[str, Any]:
        """Report a serious incident (Article 73)."""
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "title": title,
            "description": description,
            "severity": severity,
            "reported_by": reported_by,
            "is_widespread": is_widespread,
        }
        if anomaly_event_id:
            payload["anomaly_event_id"] = anomaly_event_id
        return self._transport.post("/incidents", payload)

    def list_incidents(self, **params: Any) -> dict[str, Any]:
        """List incident reports. Supports: status, page, limit."""
        return self._transport.get("/incidents", params=params or None)

    def update_incident(self, incident_id: int, **fields: Any) -> dict[str, Any]:
        """Update incident (status, root_cause, corrective_actions, authority_notified)."""
        return self._transport.patch(f"/incidents/{incident_id}", {k: v for k, v in fields.items() if v is not None})

    # --- Billing ---

    def get_billing_info(self) -> dict[str, Any]:
        """Get current plan and agent usage."""
        return self._transport.get("/billing")
