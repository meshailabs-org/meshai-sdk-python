# MeshAI Python SDK

Python client for the [MeshAI Agent Control Plane](https://meshai.dev). Register agents, send telemetry, query anomalies, manage governance policies, and track EU AI Act compliance.

## Install

```bash
pip install meshai-sdk
```

With framework auto-tracking:

```bash
pip install meshai-sdk[openai]      # OpenAI auto-tracking
pip install meshai-sdk[anthropic]   # Anthropic auto-tracking
pip install meshai-sdk[crewai]      # CrewAI auto-tracking
pip install meshai-sdk[langchain]   # LangChain/LangGraph auto-tracking
pip install meshai-sdk[autogen]     # AutoGen auto-tracking
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

oai = wrap_openai(openai.OpenAI(), meshai=meshai)
response = oai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
# Usage automatically tracked!
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

### CrewAI

```python
from meshai import MeshAI
from meshai.integrations.crewai import track_crewai

meshai = MeshAI(api_key="msh_...", agent_name="my-crew")
meshai.register(framework="crewai")

# Enable global tracking — all crews auto-track usage
track_crewai(meshai)

# Run your crew as normal — model extracted from each LLM call
crew.kickoff()
```

### LangChain / LangGraph

```python
from meshai import MeshAI
from meshai.integrations.langchain import MeshAICallbackHandler
from langchain_openai import ChatOpenAI

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="langchain")

handler = MeshAICallbackHandler(meshai)

# Use with any LangChain model — model extracted automatically
llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

# Or with LangGraph
config = {"callbacks": [handler]}
result = graph.stream(input, config=config)
```

### AutoGen

```python
from meshai import MeshAI
from meshai.integrations.autogen import track_autogen

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="autogen")

# Enable global tracking
track_autogen(meshai)

# Run agents as normal — all LLM calls tracked
```

## Agent Queries

```python
# List all agents
agents = client.list_agents(status="healthy", page=1, limit=50)

# Get single agent
agent = client.get_agent("01AGENT_ID_HERE")

# Update agent
client.update_agent("01AGENT_ID", description="Updated description")

# Delete agent (soft delete)
client.delete_agent("01AGENT_ID")
```

## Cost Intelligence

```python
# Cost summary
summary = client.get_cost_summary(start="2026-03-01T00:00:00Z", end="2026-03-17T00:00:00Z")

# Breakdown by agent or model
by_agent = client.get_cost_by_agent()
by_model = client.get_cost_by_model()
```

## Anomaly Detection

```python
# List active anomalies
anomalies = client.list_anomalies(severity="critical")

# Get summary
summary = client.get_anomaly_summary()

# Acknowledge or resolve
client.acknowledge_anomaly(event_id=42)
client.resolve_anomaly(event_id=42)
```

## Governance

### Risk Classification

```python
# AI-assisted risk suggestion
suggestion = client.get_risk_suggestion("01AGENT_ID")

# Classify agent risk (EU AI Act Article 6)
client.classify_risk(
    agent_id="01AGENT_ID",
    risk_level="high",
    justification="Handles PII in production",
    assessed_by="security-team",
)

# Get classification
risk = client.get_risk_classification("01AGENT_ID")
```

### Policies

```python
# Create a policy
client.create_policy(
    name="Production models only",
    policy_type="model_allowlist",
    rules={"allowed_models": ["gpt-4o", "claude-3-sonnet"]},
    conditions={"environments": ["production"]},
)

# List policies
policies = client.list_policies(enabled=True)

# Dry-run evaluate
results = client.evaluate_policies(
    agent_id="01AGENT_ID",
    provider="openai",
    model="gpt-4o",
)

# Update or delete
client.update_policy(policy_id=1, enabled=False)
client.delete_policy(policy_id=1)
```

### Approvals (HITL)

```python
# Check pending approvals
count = client.get_pending_count()

# List pending
pending = client.list_approvals(status="pending")

# Approve or deny
client.decide_approval(
    request_id=1,
    decision="approved",
    reviewer_id="admin",
    reason="Reviewed and approved",
)
```

## Compliance (EU AI Act)

```python
# Readiness score (0-120)
readiness = client.get_readiness_score()

# FRIA template (Article 27)
fria = client.get_fria("01AGENT_ID")

# Transparency card
card = client.get_transparency_card("01AGENT_ID")
```

## Incident Reporting (Article 73)

```python
# Report incident
client.create_incident(
    agent_id="01AGENT_ID",
    title="Data leak detected",
    description="Agent exposed PII in response",
    severity="critical",
    reported_by="security-team",
    is_widespread=False,  # True = 2-day deadline, False = 15-day
)

# List and update
incidents = client.list_incidents(status="reported")
client.update_incident(
    incident_id=1,
    root_cause="Model hallucination",
    corrective_actions="Added PII filter policy",
    authority_notified=True,
)
```

## Billing

```python
# Current plan and agent usage
billing = client.get_billing_info()
# Returns: {plan, price_usd, max_agents, current_agents, at_limit}
```

## Configuration

```python
client = MeshAI(
    api_key="msh_...",              # Required
    agent_name="my-agent",          # Agent name (or pass to register())
    base_url="https://api.meshai.dev",
    environment="production",       # production, staging, dev
    batch_size=100,                 # Events per batch
    flush_interval_seconds=5.0,     # Auto-flush interval
    heartbeat_interval_seconds=60,  # Background heartbeat interval
    max_retries=3,                  # Retry count on failure
    timeout_seconds=10.0,           # HTTP request timeout
)
```

## Design Principles

- **Never crashes the host** — all SDK errors are caught and logged
- **Buffered batching** — events flush every 5s or 100 events
- **Background heartbeat** — daemon thread, auto-stops on shutdown
- **Minimal dependencies** — only `httpx`

## License

MIT
