"""SDK configuration."""

import re
from dataclasses import dataclass, field

_LOCALHOST_RE = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d+)?(/.*)?$")


@dataclass(frozen=True)
class MeshAIConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.meshai.dev"
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
        if not self.base_url.startswith("https://"):
            if not _LOCALHOST_RE.match(self.base_url):
                raise ValueError("base_url must use HTTPS (except localhost for development)")

    def __repr__(self) -> str:
        key_preview = self.api_key[:8] + "..." if self.api_key else ""
        return (
            f"MeshAIConfig(api_key='{key_preview}', base_url='{self.base_url}', "
            f"agent_name='{self.agent_name}', environment='{self.environment}')"
        )
