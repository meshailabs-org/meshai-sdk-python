# AGENTS.md — meshai-sdk-python

This file guides AI coding agents (Claude Code, Codex, and others) working in this repository. CLAUDE.md is a symlink to this file.

## Overview
Python SDK for the MeshAI Agent Control Plane. Lightweight, zero-config agent telemetry.

## Architecture
```
src/meshai/
  __init__.py       — Exports MeshAI
  client.py         — Main SDK client (register, heartbeat, track_usage)
  config.py         — Immutable config dataclass
  transport.py      — HTTP layer with retry (httpx)
  batcher.py        — Thread-safe buffered batch flusher
  tracer/
    tracer.py       — OTel-native Tracer: Session/Span context managers (sync, v1)
    filters.py      — Default-deny content filter + secret redaction (fail-closed)
  integrations/
    openai.py       — Auto-tracking wrapper for OpenAI
    anthropic.py    — Auto-tracking wrapper for Anthropic
```

## Key Design Principles
- **Never crash the host agent** — all SDK errors are caught and logged
- **Buffered batching** — events flush every 5s or 100 events
- **Background heartbeat** — daemon thread, auto-stops on shutdown
- **Graceful shutdown** — atexit handler flushes remaining events
- **Minimal dependencies** — only httpx (Tracer deps live behind the `tracer` extra)

## Commands
```bash
pip install -e ".[dev]"
pytest -v
ruff check src/ tests/
```

## API Target
- Base URL: https://api.meshai.dev/api/v1
- Auth: Bearer token (msh_xxx)
