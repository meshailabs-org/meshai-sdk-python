"""HTTP transport layer with retry logic."""

import logging
import time
from typing import Any

import httpx

from meshai.config import MeshAIConfig

logger = logging.getLogger("meshai")


class Transport:
    """Handles HTTP communication with the MeshAI API."""

    def __init__(self, config: MeshAIConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"meshai-python/0.1.0",
            },
            timeout=config.timeout_seconds,
        )

    def post(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        """POST with retry logic. Never raises — returns error dict on failure."""
        last_error = None
        for attempt in range(self._config.max_retries):
            try:
                response = self._client.post(f"/api/v1{path}", json=json)
                if response.status_code < 500:
                    return response.json()
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as e:
                last_error = str(e)

            if attempt < self._config.max_retries - 1:
                backoff = self._config.retry_backoff_seconds * (2**attempt)
                time.sleep(backoff)

        logger.warning("MeshAI API request failed after %d retries: %s", self._config.max_retries, last_error)
        return {"success": False, "error": last_error}

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET request. Never raises."""
        try:
            response = self._client.get(f"/api/v1{path}", params=params)
            return response.json()
        except httpx.HTTPError as e:
            logger.warning("MeshAI API GET failed: %s", e)
            return {"success": False, "error": str(e)}

    def close(self) -> None:
        self._client.close()
