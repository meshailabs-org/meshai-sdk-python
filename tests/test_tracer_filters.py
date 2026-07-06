"""Tests for the tracer filter pipeline: default-deny, redaction, fail-closed."""

import json

import pytest

from meshai.tracer.filters import (
    TIMEOUT_PLACEHOLDER,
    FilterConfig,
    FilterPipeline,
)

ALLOW_BASH_INPUT = FilterConfig(allow={"Bash": frozenset({"tool_input"})})


# --- default-deny allowlist ------------------------------------------------


def test_default_config_denies_everything():
    pipeline = FilterPipeline()
    assert pipeline.filter_content("Bash", "tool_input", "ls") is None
    assert pipeline.filter_content("Bash", "tool_output", "x") is None


def test_allowlist_is_per_tool_and_per_field():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    assert pipeline.filter_content("Bash", "tool_input", "ls -la") == "ls -la"
    assert pipeline.filter_content("Bash", "tool_output", "x") is None
    assert pipeline.filter_content("Read", "tool_input", "x") is None


def test_non_string_content_is_json_coerced_then_redacted():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    payload = {"cmd": "deploy", "env": {"ANTHROPIC_API_KEY": "sk-ant-" + "k" * 24}}
    result = pipeline.filter_content("Bash", "tool_input", payload)
    assert "sk-ant-" not in result
    assert "deploy" in result


# --- secret redaction corpus (seed of the T9 adversarial corpus) ------------

CORPUS = [
    ("anthropic_api_key", f"key=sk-ant-api03-{'a' * 24}"),
    ("openai_api_key", f"OPENAI_API_KEY=sk-proj-{'b' * 24}"),
    ("openai_api_key", f"sk-{'c' * 32}"),
    ("meshai_api_key", "Authorization: msh_0123456789abcdef"),
    ("aws_access_key_id", "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"),
    ("aws_access_key_id", "ASIAIOSFODNN7EXAMPLE is temporary"),
    ("github_token", f"ghp_{'D' * 36}"),
    ("github_token", f"github_pat_{'e' * 22}_{'f' * 20}"),
    ("slack_token", "xoxb-1234567890-abcdefghijk"),
    ("google_api_key", f"AIza{'G' * 35}"),
    (
        "jwt",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c",
    ),
    (
        "private_key_block",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END RSA PRIVATE KEY-----",
    ),
    (
        "private_key_block",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaA==",  # truncated paste
    ),
    ("gcp_service_account", '{"private_key_id": "0123456789abcdef0123"}'),
    ("bearer_header", "curl -H 'Authorization: Bearer abcdef0123456789TOKEN'"),
    ("basic_auth_url", "postgres://meshai:s3cretpw@db.internal:5432/prod"),
    ("credential_assignment", "password = hunter2hunter2"),
    ("credential_assignment", "API_KEY: 'zzzzzzzzzzzz'"),
    ("credential_assignment", "access_token=abcdef123456"),
]


@pytest.mark.parametrize(("expected_label", "sample"), CORPUS)
def test_corpus_secret_is_redacted(expected_label, sample):
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    result = pipeline.filter_content("Bash", "tool_input", sample)
    assert f"[REDACTED:{expected_label}]" in result, result


def test_clean_text_passes_untouched():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    clean = "pytest tests/ -q && git status --short"
    assert pipeline.filter_content("Bash", "tool_input", clean) == clean


def test_multiple_secrets_in_one_value_all_redacted():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    value = f"a=sk-ant-{'a' * 20} b=AKIAIOSFODNN7EXAMPLE"
    result = pipeline.filter_content("Bash", "tool_input", value)
    assert "sk-ant-" not in result
    assert "AKIA" not in result


# --- hostile-QA evasion (T9): homoglyph, base64-wrap, attribute-split -------


def test_homoglyph_cyrillic_secret_redacted():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    # CYRILLIC SMALL A (U+0430) inside the key prefix.
    sample = f"key=sk-аnt-{'a' * 24}"
    result = pipeline.filter_content("Bash", "tool_input", sample)
    assert "[REDACTED:anthropic_api_key]" in result
    assert "a" * 24 not in result
    assert result.startswith("key=")  # surrounding text preserved


def test_homoglyph_fullwidth_secret_redacted():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    # Fullwidth 's', 'k' and hyphen (ｓｋ－) spelling the prefix.
    sample = f"ｓｋ－ant-{'b' * 24} deployed"
    result = pipeline.filter_content("Bash", "tool_input", sample)
    assert "[REDACTED:anthropic_api_key]" in result
    assert "b" * 24 not in result
    assert result.endswith(" deployed")


def test_base64_wrapped_secret_redacted():
    import base64

    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    encoded = base64.b64encode(
        f"export ANTHROPIC_API_KEY=sk-ant-{'c' * 24}".encode()
    ).decode()
    sample = f"echo {encoded} | base64 -d | sh"
    result = pipeline.filter_content("Bash", "tool_input", sample)
    assert encoded not in result
    assert "[REDACTED:base64_anthropic_api_key]" in result
    assert result.startswith("echo ") and result.endswith(" | base64 -d | sh")


def test_base64_urlsafe_wrapped_secret_redacted():
    import base64

    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    encoded = base64.urlsafe_b64encode(
        f"aws_access_key_id = AKIAIOSFODNN7EXAMPLE >>{'!' * 8}".encode()
    ).decode()
    result = pipeline.filter_content("Bash", "tool_input", f"run {encoded}")
    assert encoded not in result
    assert "[REDACTED:base64_aws_access_key_id]" in result


def test_benign_base64_passes_untouched():
    import base64

    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    encoded = base64.b64encode(b"just a plain sentence with no secrets").decode()
    sample = f"payload={encoded}"
    assert pipeline.filter_content("Bash", "tool_input", sample) == sample


def test_long_non_base64_run_passes_untouched():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    sample = f"sha256:{'deadbeef' * 8}"
    assert pipeline.filter_content("Bash", "tool_input", sample) == sample


def test_cross_attribute_split_is_controlled_by_allowlist():
    """A secret split ACROSS attributes is out of redaction's reach by
    construction — each fragment alone matches nothing. The control for this
    class is the default-deny allowlist: fragments leak nothing unless the
    user explicitly opted the field in. This test pins that boundary."""
    pipeline = FilterPipeline()  # default deny-all
    assert pipeline.filter_content("Bash", "tool_input", "sk-ant-") is None
    assert pipeline.filter_content("Bash", "tool_output", "abcd1234" * 3) is None
    # Opted-in, the bare fragment passes — documented residual risk.
    opted = FilterPipeline(ALLOW_BASH_INPUT)
    assert pipeline.filter_content("Bash", "tool_output", "sk-ant-") is None
    assert opted.filter_content("Bash", "tool_input", "sk-ant-") == "sk-ant-"


def test_overlapping_matches_merge_and_preserve_surroundings():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    value = f"pre sk-ant-{'d' * 24} mid AKIAIOSFODNN7EXAMPLE post"
    result = pipeline.filter_content("Bash", "tool_input", value)
    assert result.startswith("pre ") and result.endswith(" post")
    assert " mid " in result
    assert result.count("[REDACTED:") == 2


# --- fail-closed on pattern timeout (D5 eng) --------------------------------


class _TimingOutPattern:
    """Duck-types regex.Pattern but always exceeds the budget.

    The regex module raises the builtin TimeoutError on budget exhaustion.
    """

    def finditer(self, value, timeout=None):
        raise TimeoutError("regex timeout")

    def search(self, value, timeout=None):
        raise TimeoutError("regex timeout")


def test_pattern_timeout_fails_closed_with_placeholder():
    pipeline = FilterPipeline(ALLOW_BASH_INPUT)
    pipeline._patterns = [("evil", _TimingOutPattern())]
    result = pipeline.filter_content("Bash", "tool_input", "anything")
    assert result == TIMEOUT_PLACEHOLDER
    assert json.loads(result) == {"filtered": True, "reason": "filter_timeout"}


# --- config loading ----------------------------------------------------------


def test_load_missing_file_is_deny_all(tmp_path):
    config = FilterConfig.load(tmp_path / "nope.yaml")
    assert config.allow == {}


def test_load_parses_tool_allowlists(tmp_path):
    path = tmp_path / "filters.yaml"
    path.write_text(
        "tools:\n"
        "  Bash:\n"
        "    allow: [tool_input, tool_output]\n"
        "  Read:\n"
        "    allow: [tool_input]\n"
    )
    config = FilterConfig.load(path)
    assert config.allows("Bash", "tool_input")
    assert config.allows("Bash", "tool_output")
    assert config.allows("Read", "tool_input")
    assert not config.allows("Read", "tool_output")


def test_load_ignores_unknown_fields(tmp_path):
    path = tmp_path / "filters.yaml"
    path.write_text("tools:\n  Bash:\n    allow: [tool_input, everything]\n")
    config = FilterConfig.load(path)
    assert config.allow["Bash"] == frozenset({"tool_input"})


def test_load_malformed_yaml_fails_closed(tmp_path):
    path = tmp_path / "filters.yaml"
    path.write_text("tools: [unclosed")
    config = FilterConfig.load(path)
    assert config.allow == {}
