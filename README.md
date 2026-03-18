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
pip install meshai-sdk[gemini]           # Google Gemini
pip install meshai-sdk[bedrock]          # AWS Bedrock
pip install meshai-sdk[llamaindex]       # LlamaIndex
pip install meshai-sdk[agno]             # Agno (ex-Phidata)
pip install meshai-sdk[pydantic-ai]      # Pydantic AI
pip install meshai-sdk[semantic-kernel]  # Microsoft Semantic Kernel
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

### Google Gemini

```python
from meshai import MeshAI
from meshai.integrations.gemini import wrap_gemini
from google import genai

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="custom", model_provider="google")

client = genai.Client(api_key="...")
tracked = wrap_gemini(client, meshai=meshai)
response = tracked.models.generate_content(model="gemini-2.5-pro", contents="Hello")
```

### AWS Bedrock

```python
from meshai import MeshAI
from meshai.integrations.bedrock import wrap_bedrock
import boto3

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="custom", model_provider="bedrock")

bedrock = boto3.client("bedrock-runtime")
tracked = wrap_bedrock(bedrock, meshai=meshai)
response = tracked.converse(modelId="anthropic.claude-3-sonnet", messages=[...])
```

### LlamaIndex

```python
from meshai import MeshAI
from meshai.integrations.llamaindex import MeshAILlamaHandler
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="llamaindex")

handler = MeshAILlamaHandler(meshai)
Settings.callback_manager = CallbackManager([handler])
# All LlamaIndex LLM calls now auto-track usage
```

### Agno

```python
from meshai import MeshAI
from meshai.integrations.agno import track_agno

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="agno")

track_agno(meshai)
# All Agno agents now auto-track usage
```

### Pydantic AI

```python
from meshai import MeshAI
from meshai.integrations.pydantic_ai import track_pydantic_ai

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="pydantic-ai")

track_pydantic_ai(meshai)
# All Pydantic AI agents now auto-track usage
```

### Semantic Kernel

```python
from meshai import MeshAI
from meshai.integrations.semantic_kernel import track_semantic_kernel
import semantic_kernel as sk

meshai = MeshAI(api_key="msh_...", agent_name="my-agent")
meshai.register(framework="semantic-kernel")

kernel = sk.Kernel()
track_semantic_kernel(meshai, kernel)
# All Semantic Kernel function calls now auto-track usage
```


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

## Kill Switch

```python
# Block an agent immediately (enforced at proxy layer)
client.block_agent(
    agent_id="01AGENT_ID",
    reason="Anomalous behavior detected — cost spike 10x above baseline",
)

# Unblock when resolved
client.unblock_agent(agent_id="01AGENT_ID")
```

## Agent Relationships

```python
# Get an agent's model/provider dependencies
relationships = client.get_agent_relationships("01AGENT_ID")

# Get the full organization-wide relationship graph (nodes + edges)
graph = client.get_relationship_graph()
# Returns: {nodes: [...], edges: [...]} — ready for D3.js visualization
```

## ABAC (Agent Owners)

```python
# Assign an owner with permissions
client.assign_owner(
    agent_id="01AGENT_ID",
    owner_type="team",
    owner_id="ml-platform-team",
    owner_name="ML Platform Team",
    permissions={"can_invoke": True, "can_configure": True, "can_delete": False},
)

# List owners of an agent
owners = client.list_agent_owners("01AGENT_ID")

# List agents owned by a specific owner
agents = client.list_owner_agents("ml-platform-team")

# Remove an owner
client.remove_owner(agent_id="01AGENT_ID", owner_id=1)
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
