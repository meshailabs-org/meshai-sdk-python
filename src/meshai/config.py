"""SDK configuration."""

import re
from dataclasses import dataclass, field

_LOCALHOST_RE = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d+)?(/.*)?$")

_DEFAULT_API_URL = "https://api.meshai.dev"
_DEFAULT_INGEST_URL = "https://ingest.meshai.dev"


@dataclass(frozen=True)
class MeshAIConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.meshai.dev"
    # Telemetry goes to a dedicated collector gateway, not the API host.
    # The gateway absorbs large exports and re-chunks them before they reach
    # the API, so a big batch is delivered rather than rejected. Leave as None
    # to get the right behaviour automatically; see ``resolved_ingest_url``.
    ingest_url: str | None = None
    agent_name: str = ""
    environment: str = "production"
    timeout_seconds: float = 10.0
    # Batching
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    # Heartbeat
    heartbeat_interval_seconds: float = 60.0
    # Retry
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.startswith("msh_") or len(self.api_key) < 16:
            raise ValueError("Invalid API key format")
        if not self.base_url.startswith("https://") and not _LOCALHOST_RE.match(
            self.base_url
        ):
            raise ValueError("base_url must use HTTPS (except localhost for development)")

    @property
    def resolved_ingest_url(self) -> str:
        """Base URL for OTLP export.

        Explicit ``ingest_url`` always wins. Otherwise, only the DEFAULT
        production host is redirected to the gateway: a caller who pointed
        ``base_url`` at localhost or a self-hosted deployment must keep their
        telemetry going there, not to MeshAI's gateway. Getting that backwards
        would silently ship a self-hoster's spans off their own infrastructure.
        """
        if self.ingest_url:
            return self.ingest_url
        if self.base_url == _DEFAULT_API_URL:
            return _DEFAULT_INGEST_URL
        return self.base_url

    def __repr__(self) -> str:
        key_preview = self.api_key[:8] + "..." if self.api_key else ""
        return (
            f"MeshAIConfig(api_key='{key_preview}', base_url='{self.base_url}', "
            f"agent_name='{self.agent_name}', environment='{self.environment}')"
        )
