"""HTTP transport layer with retry logic."""

import logging
import random
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
                "User-Agent": "meshai-python/0.1.0",
            },
            timeout=config.timeout_seconds,
            verify=True,
            follow_redirects=False,
        )

    def _safe_parse(self, response: httpx.Response, path: str) -> dict[str, Any]:
        """Parse JSON response safely — never raises."""
        try:
            return response.json()
        except Exception:
            return {"success": False, "error": f"HTTP {response.status_code}: non-JSON response from {path}"}

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter to avoid synchronized retry storms."""
        base = self._config.retry_backoff_seconds * (2**attempt)
        return base + random.uniform(0, base * 0.25)

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    def post(
        self, path: str, json: dict[str, Any], idempotent: bool = False
    ) -> dict[str, Any]:
        """POST. Never raises — returns error dict on failure.

        Non-idempotent POSTs (the default) are NOT retried on 5xx or network
        errors: the request may have already committed server-side, so a blind
        retry would duplicate the effect (e.g. double-counted usage, duplicate
        agents/incidents). A 429 is always safe to retry (the request was
        rejected, not processed) and honors Retry-After. Pass idempotent=True for
        endpoints that are safe to repeat.
        """
        last_error = None
        for attempt in range(self._config.max_retries):
            try:
                response = self._client.post(f"/api/v1{path}", json=json)
                status = response.status_code
                if status == 429:
                    last_error = "HTTP 429"
                    if attempt < self._config.max_retries - 1:
                        wait = self._parse_retry_after(
                            response.headers.get("retry-after")
                        )
                        time.sleep(wait if wait is not None else self._backoff(attempt))
                        continue
                    return self._safe_parse(response, path)
                if status < 500:
                    return self._safe_parse(response, path)
                last_error = f"HTTP {status}"
                if not idempotent:
                    return self._safe_parse(response, path)
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = f"{type(e).__name__} on attempt {attempt + 1}"
                if not idempotent:
                    return {"success": False, "error": last_error}

            if attempt < self._config.max_retries - 1:
                time.sleep(self._backoff(attempt))

        logger.warning(
            "MeshAI API request failed after %d retries: %s",
            self._config.max_retries, last_error,
        )
        return {"success": False, "error": last_error}

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET request. Never raises."""
        try:
            response = self._client.get(f"/api/v1{path}", params=params)
            return self._safe_parse(response, path)
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("MeshAI API GET %s failed: %s", path, type(e).__name__)
            return {"success": False, "error": f"{type(e).__name__}: request to {path} failed"}

    def close(self) -> None:
        self._client.close()

    def patch(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        """PATCH request. Never raises."""
        try:
            response = self._client.patch(f"/api/v1{path}", json=json)
            return self._safe_parse(response, path)
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("MeshAI API PATCH %s failed: %s", path, type(e).__name__)
            return {"success": False, "error": f"{type(e).__name__}: request to {path} failed"}

    def delete(self, path: str) -> dict[str, Any]:
        """DELETE request. Never raises."""
        try:
            response = self._client.delete(f"/api/v1{path}")
            if response.status_code == 204:
                return {"success": True}
            return self._safe_parse(response, path)
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("MeshAI API DELETE %s failed: %s", path, type(e).__name__)
            return {"success": False, "error": f"{type(e).__name__}: request to {path} failed"}
