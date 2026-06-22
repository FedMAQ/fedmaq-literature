"""LLM client and utility helpers using OpenRouter."""

from __future__ import annotations

import os
import dotenv
from openai import OpenAI

# Load environment variables from .env if present
dotenv.load_dotenv()

DEFAULT_SUMMARIZE_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_SYNTHESIS_MODEL = "deepseek/deepseek-v4-pro"


def get_llm_client() -> OpenAI:
    """Return an OpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please create a '.env' file in the fedmaq-literature workspace "
            "and set OPENROUTER_API_KEY=your_key_here."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        # OpenRouter optional headers
        default_headers={
            "HTTP-Referer": "https://github.com/BSMSCS-Thesis-Hub/fedmaq-literature",
            "X-Title": "FedMAQ Literature RAG Tool",
        },
    )


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    model: str = DEFAULT_SUMMARIZE_MODEL,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> str:
    """Generate completion using OpenRouter LLM."""
    client = get_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
