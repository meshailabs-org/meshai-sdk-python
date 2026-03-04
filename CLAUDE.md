# CLAUDE.md — meshai-sdk-python

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
  integrations/
    openai.py       — Auto-tracking wrapper for OpenAI
    anthropic.py    — Auto-tracking wrapper for Anthropic
```

## Key Design Principles
- **Never crash the host agent** — all SDK errors are caught and logged
- **Buffered batching** — events flush every 5s or 100 events
- **Background heartbeat** — daemon thread, auto-stops on shutdown
- **Graceful shutdown** — atexit handler flushes remaining events
- **Minimal dependencies** — only httpx

## Commands
```bash
pip install -e ".[dev]"
pytest -v
ruff check src/ tests/
```

## API Target
- Base URL: https://api.meshai.dev/api/v1
- Auth: Bearer token (msh_xxx)
