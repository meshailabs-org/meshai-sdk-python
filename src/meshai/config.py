"""SDK configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MeshAIConfig:
    api_key: str
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
        if not self.api_key or not self.api_key.startswith("msh_"):
            raise ValueError("api_key must start with 'msh_'")
        if not self.base_url.startswith("https://"):
            if self.base_url.startswith("http://localhost") or self.base_url.startswith("http://127.0.0.1"):
                pass  # Allow localhost for development
            else:
                raise ValueError("base_url must use HTTPS (except localhost for development)")
