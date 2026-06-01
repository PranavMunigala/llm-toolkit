# Constants for llm-toolkit, including model pricing information and other shared constants.

PRICES = {
    "claude-3-5-haiku-latest": {"input_per_million_usd": 1.00, "output_per_million_usd": 5.00},
    "claude-3-5-sonnet-latest": {"input_per_million_usd": 3.00, "output_per_million_usd": 15.00},
    "gpt-4o-mini": {"input_per_million_usd": 0.15, "output_per_million_usd": 0.60},
    "gpt-4o": {"input_per_million_usd": 2.50, "output_per_million_usd": 10.00},
    "gemini-1.5-flash": {"input_per_million_usd": 0.35, "output_per_million_usd": 1.05},
    "gemini-1.5-pro": {"input_per_million_usd": 3.50, "output_per_million_usd": 10.50},
}
