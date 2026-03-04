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
