# LLM Toolkit

A Python toolkit that provides a combined interface for interacting with multiple Large Language Model (LLM) providers, including Anthropic Claude, OpenAI GPT, and Google Gemini.

The purpose of this project is to simplify working with multiple AI providers by hiding provider-specific API details behind a single interface. Rather than learning different SDKs, request formats, response structures, and streaming implementations for each provider, users can interact with all supported models through one consistent API while automatically tracking token usage and estimated cost.

This project was built as a reusable foundation for future AI applications, allowing new projects to focus on functionality rather than repeatedly implementing API integrations (allowing for the use of the three main LLMs)

---

## Features

- Unified API for Claude, GPT, and Gemini
- Standardized request and response formats
- Streaming response support
- Token usage tracking
- Cost estimation based on model pricing
- Command-line interface (CLI)
- Architecture for adding future additional providers

---

## How It Works

The toolkit follows a simple workflow:

```text
User
 ↓
CLI or Python Application
 ↓
ask()
 ↓
Provider Adapter
 ↓
Claude / GPT / Gemini
```

A user submits a prompt through either Python code or the command line. The toolkit routes the request through a provider-specific adapter, which formats the request according to that provider's API requirements. The response is then converted into a standardized format before being returned to the user.

This design allows applications to switch between providers without changing business logic.

---

## Repository Structure

```text
llm-toolkit/
│
├── __init__.py
├── adapters.py
├── core.py
├── constants.py
├── cli.py
└── tests/
```

### `__init__.py`

Acts as the package entry point.

This file exposes the main `ask()` function, allowing users to write:

```python
from llm_toolkit import ask
```

instead of importing directly from internal modules.

---

### `adapters.py`

Contains provider-specific implementations for:

- Anthropic Claude
- OpenAI GPT
- Google Gemini

Each provider exposes a different API, message format, response structure, and streaming implementation. The adapter layer handles these differences and converts all responses into a standardized format.

This file also defines the core data structures used throughout the project:

#### Message

Represents a chat message.

```python
Message(
    role="user",
    content="What is DNA?"
)
```

#### Usage

Stores token counts.

```python
Usage(
    input_tokens=100,
    output_tokens=50
)
```

#### Response

Represents a normalized response returned by any provider.

```python
Response(
    provider="gpt",
    model="gpt-4o-mini",
    text="DNA is...",
    usage=...
)
```

The adapter pattern allows future projects to work with a single response format regardless of which provider generated the output.

---

### `core.py`

Contains the primary business logic of the toolkit.

#### `ask()`

The main entry point used throughout the project.

Example:

```python
response = ask(
    prompt="Explain DNA replication",
    provider="gpt"
)
```

The function:

1. Validates user input
2. Selects a default model if one is not provided
3. Creates a standardized message object
4. Routes the request to the correct provider adapter
5. Returns a standardized response

#### `ask_all()`

Sends the same prompt to Claude, GPT, and Gemini and returns all responses along with total execution time.

#### `cost_cents()`

Calculates estimated API cost using:

- Input token count
- Output token count
- Model pricing information

---

### `constants.py`

Stores model pricing information.

Example:

```python
PRICES = {
    "gpt-4o-mini": {
        "input_per_million_usd": 0.15,
        "output_per_million_usd": 0.60
    }
}
```

Keeping pricing information separate allows costs to be updated without modifying business logic.

---

### `cli.py`

Implements the Command Line Interface (CLI).

A CLI allows users to interact with the toolkit directly from the terminal without writing Python code.

Examples:

```bash
python -m llm_toolkit.cli "What is DNA?"
```

```bash
python -m llm_toolkit.cli "Explain attention" --all
```

```bash
python -m llm_toolkit.cli "Explain transformers" --provider claude --stream
```

The CLI handles:

- Reading command-line arguments
- Calling the toolkit
- Printing responses
- Displaying token usage
- Displaying estimated cost
- Displaying execution time

---

## Supported Providers

### Anthropic Claude

Uses the Anthropic Python SDK.

Supported through:

```python
from anthropic import Anthropic
```

Example models:

- Claude Haiku
- Claude Sonnet
- Claude Opus

---

### OpenAI GPT

Uses the OpenAI Python SDK.

Supported through:

```python
from openai import OpenAI
```

Example models:

- GPT-4o
- GPT-4o-mini

---

### Google Gemini

Uses Google's Gemini SDK.

Supported through:

```python
from google import genai
```

Example models:

- Gemini Flash
- Gemini Pro

---

## Example Usage

### Python

```python
from llm_toolkit import ask

response = ask(
    prompt="Explain attention in two sentences.",
    provider="gpt"
)

print(response.text)
```

### Compare All Providers

```bash
python -m llm_toolkit.cli "Explain attention in two sentences" --all
```

### Streaming Responses

```bash
python -m llm_toolkit.cli "Explain transformers" --provider claude --stream
```

---

## Cost Tracking

The toolkit automatically tracks token usage and estimates API costs using provider pricing information.

Example output:

```text
[gpt:gpt-4o-mini]

Attention allows transformers to focus on relevant parts of the input sequence when generating each token.

usage: in=25 out=40 cost(cents) total=0.000278
```

This makes it easy to compare providers not only by response quality but also by cost.

---

## Why Build This?

Every LLM provider exposes a different API, response structure, streaming implementation, and usage reporting format.

Without this toolkit, every new AI project would require writing separate integrations for Claude, GPT, and Gemini.

This project provides a common abstraction layer that allows future applications to switch providers with minimal code changes while maintaining consistent behavior and cost tracking.

It serves as a reusable foundation for future projects involving:

- AI agents
- Retrieval-augmented generation (RAG)
- Evaluation frameworks
- Multi-model comparisons
- LLM-powered applications

---
---

## License

MIT License
