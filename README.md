# MeshAI Python SDK

Agent telemetry for the [MeshAI Agent Control Plane](https://meshai.dev). Register agents, send heartbeats, and track token usage with zero-config batching.

## Install

```bash
pip install meshai-sdk
```

With OpenAI or Anthropic auto-tracking:

```bash
pip install meshai-sdk[openai]
pip install meshai-sdk[anthropic]
```

## Quick Start

```python
from meshai import MeshAI

client = MeshAI(api_key="msh_...", agent_name="my-agent")
client.register(framework="crewai", model_provider="openai", model_name="gpt-4o")

# Automatic heartbeats every 60s
client.start_heartbeat()

# Track token usage (buffered, batched automatically)
client.track_usage(
    model_provider="openai",
    model_name="gpt-4o",
    input_tokens=1500,
    output_tokens=800,
)

# Graceful shutdown (also registered via atexit)
client.shutdown()
```

## Auto-Tracking Integrations

### OpenAI

```python
from meshai import MeshAI
from meshai.integrations.openai import wrap_openai
import openai

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(model_provider="openai", model_name="gpt-4o")

# Wrap the OpenAI client — all completions auto-track usage
oai = wrap_openai(openai.OpenAI(), meshai=meshai)
response = oai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Anthropic

```python
from meshai import MeshAI
from meshai.integrations.anthropic import wrap_anthropic
import anthropic

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(model_provider="anthropic", model_name="claude-sonnet-4-6")

ant = wrap_anthropic(anthropic.Anthropic(), meshai=meshai)
response = ant.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Configuration

```python
client = MeshAI(
    api_key="msh_...",              # Required, must start with msh_
    agent_name="my-agent",          # Agent name (or pass to register())
    base_url="https://api.meshai.dev",  # API endpoint
    environment="production",       # production, staging, dev
    batch_size=100,                 # Events per batch
    flush_interval_seconds=5.0,     # Seconds between auto-flushes
    heartbeat_interval_seconds=60,  # Background heartbeat interval
    max_retries=3,                  # Retry count on failure
    timeout_seconds=10.0,           # HTTP request timeout
)
```

## Design Principles

- **Never crashes the host** -- all SDK errors are caught and logged
- **Buffered batching** -- events flush every 5s or 100 events
- **Background heartbeat** -- daemon thread, auto-stops on shutdown
- **Minimal dependencies** -- only `httpx`

## License

MIT
