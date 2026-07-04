"""Content filter pipeline for the Tracer.

Default-deny: tool_input/tool_output content is DROPPED unless the user
explicitly allowlists it per tool in ``~/.config/meshai/filters.yaml``.
Structural metadata (tool name, timing, token counts) always flows.

Allowlisted content still passes through secret-redaction patterns before
emission. Redaction runs under a per-pattern timeout (``regex`` module) as a
ReDoS guard and fails CLOSED: on timeout the value is replaced by a
placeholder, never emitted raw.

This pipeline is shared by every MeshAI connector (claude-code, and later
Codex/Cursor/Aider) — one security-critical codebase to audit.

Config file shape::

    tools:
      Bash:
        allow: [tool_input, tool_output]
      Read:
        allow: [tool_input]
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import regex

logger = logging.getLogger("meshai")

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "meshai" / "filters.yaml"

#: Content fields subject to default-deny. Everything else is structural.
CONTENT_FIELDS = frozenset({"tool_input", "tool_output"})

#: Per-pattern, per-value regex budget. Generous for honest inputs; a
#: crafted ReDoS payload trips it and the value fails closed.
PATTERN_TIMEOUT_SECONDS = 0.25

TIMEOUT_PLACEHOLDER = json.dumps({"filtered": True, "reason": "filter_timeout"})

# Built-in secret patterns, applied to every allowlisted value. Sourced from
# detect-secrets/trufflehog signatures plus AI-ecosystem keys. Order matters
# only for placeholder naming (all matches redact); keep specific prefixes
# before generic ones so the label is accurate.
BUILTIN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("anthropic_api_key", r"\bsk-ant-[A-Za-z0-9_\-]{16,}"),
    ("openai_api_key", r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"),
    ("meshai_api_key", r"\bmsh_[A-Za-z0-9_\-]{8,}"),
    ("aws_access_key_id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    (
        "github_token",
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"
        r"|\bgithub_pat_[A-Za-z0-9_]{20,}",
    ),
    ("slack_token", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{30,}"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
    (
        "private_key_block",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
        r"[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
    ),
    ("gcp_service_account", r"\"private_key_id\"\s*:\s*\"[0-9a-f]{8,}\""),
    ("bearer_header", r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{16,}"),
    ("basic_auth_url", r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@"),
    (
        "credential_assignment",
        r"(?i)\b(?:password|passwd|secret|api_key|apikey|access_token|"
        r"auth_token)\b\s*[:=]\s*['\"]?\S{6,}",
    ),
)


@dataclass(frozen=True)
class FilterConfig:
    """Per-tool allowlist of content fields. Immutable; default is deny-all."""

    allow: dict[str, frozenset[str]] = field(default_factory=dict)

    def allows(self, tool_name: str, content_field: str) -> bool:
        return content_field in self.allow.get(tool_name, frozenset())

    @classmethod
    def load(cls, path: Path | None = None) -> "FilterConfig":
        """Load ``filters.yaml``. Any problem fails CLOSED to deny-all.

        A missing file is the normal case (default-deny needs no config);
        a malformed file is logged and treated as absent — never crash the
        host agent, never widen the allowlist on error.
        """
        config_path = path or DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()
        try:
            import yaml  # noqa: PLC0415 — optional dep, only needed with a config file

            raw = yaml.safe_load(config_path.read_text()) or {}
            tools = raw.get("tools") or {}
            allow = {
                str(tool): frozenset(
                    str(f) for f in (spec or {}).get("allow", []) if str(f) in CONTENT_FIELDS
                )
                for tool, spec in tools.items()
            }
            return cls(allow=allow)
        except Exception:  # noqa: BLE001 — fail closed on any parse/read error
            logger.warning(
                "meshai.tracer: could not parse %s; content filtering stays "
                "default-deny", config_path, exc_info=True,
            )
            return cls()


class FilterPipeline:
    """Applies the allowlist decision, then secret redaction, fail-closed."""

    def __init__(
        self,
        config: FilterConfig | None = None,
        patterns: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        self._config = config or FilterConfig()
        self._patterns: list[tuple[str, regex.Pattern]] = [
            (name, regex.compile(pat)) for name, pat in (patterns or BUILTIN_PATTERNS)
        ]

    def filter_content(
        self, tool_name: str, content_field: str, value: object
    ) -> str | None:
        """Return the emittable value for a content field, or None to drop it.

        None (not-allowlisted) means the attribute is simply not emitted —
        structural metadata only, per the default-deny posture.
        """
        if not self._config.allows(tool_name, content_field):
            return None
        return self.redact(_coerce(value))

    def redact(self, value: str) -> str:
        """Scrub secrets from an allowlisted value; fail CLOSED on timeout."""
        for name, pattern in self._patterns:
            try:
                value = pattern.sub(
                    f"[REDACTED:{name}]", value, timeout=PATTERN_TIMEOUT_SECONDS
                )
            except TimeoutError:
                # regex.TimeoutError subclasses TimeoutError. A timeout means
                # we could not prove the value clean — drop it entirely.
                logger.warning(
                    "meshai.tracer: filter pattern %r timed out; value dropped "
                    "(fail-closed)", name,
                )
                return TIMEOUT_PLACEHOLDER
        return value


def _coerce(value: object) -> str:
    """Stringify non-string content so redaction sees the full payload."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)
