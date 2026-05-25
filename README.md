# llm-toolkit

Local Python package for querying Anthropic, OpenAI, and Gemini behind one interface.

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e ".[providers]"
```

## Environment

Copy the example file and set API keys:

```bash
cp .env.example .env
```

## CLI Usage

Single provider:

```bash
llm-toolkit "Explain retrieval augmented generation" --provider gpt --stream
```

All providers:

```bash
llm-toolkit "Summarize transformer attention in 2 bullets" --all
```

## Python Usage

```python
from llm_toolkit import ask

response = ask("Hello!", provider="gpt", model="gpt-4o-mini")
print(response.text)
print(response.usage)
```

## Terminal Screenshot

_Add screenshot here_
