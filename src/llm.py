"""
llm.py — shared OpenRouter (OpenAI-compatible) client wrapper.

Centralizes client creation, retries, and structured-JSON output so every
pipeline module (schema mapping, persona labeling, chat) calls the LLM the same
way. Uses the `openai` SDK pointed at OpenRouter, running
`meta-llama/llama-3.3-70b-instruct` by default.

Structured output strategy: Llama via OpenRouter does not expose Gemini-style
enum schemas, so we use JSON-object mode, embed the target JSON schema in the
prompt, then validate the response with pydantic and retry on failure.
"""

from __future__ import annotations

import json
import time

from openai import OpenAI
from pydantic import TypeAdapter, ValidationError

from config import (
    LLM_MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_APP_URL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    has_api_key,
)

# Lazily-created singleton client (avoids constructing it at import time).
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a cached OpenRouter client, or raise if no API key is configured."""
    global _client
    if not has_api_key():
        raise RuntimeError(
            "No OpenRouter API key configured. Copy .env.template to .env and set "
            "OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)."
        )
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": OPENROUTER_APP_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )
    return _client


def generate_structured(
    prompt: str,
    response_schema,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    system_instruction: str | None = None,
):
    """
    Call the LLM and return parsed output validated against `response_schema`.

    `response_schema` is any type pydantic can validate (a BaseModel subclass,
    `list[Model]`, etc.). The JSON schema is embedded in the prompt and the
    response is requested in JSON-object mode, then validated with pydantic.
    Retries transient API errors, JSON-decode errors, and validation errors up
    to LLM_MAX_RETRIES.
    """
    client = get_client()
    adapter = TypeAdapter(response_schema)
    schema_json = json.dumps(adapter.json_schema())

    system = (system_instruction + "\n\n") if system_instruction else ""
    system += (
        "You are a precise data assistant. Respond with ONLY a single valid JSON "
        "value that conforms to this JSON Schema. Do not include markdown, code "
        "fences, or any prose.\n\nJSON Schema:\n" + schema_json
    )

    last_err: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model or OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            data = json.loads(_strip_fences(text))
            return adapter.validate_python(data)
        except (json.JSONDecodeError, ValidationError) as err:
            last_err = err  # malformed/non-conforming output -> retry
        except Exception as err:  # transient API errors -> retry with backoff
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"LLM structured call failed after {LLM_MAX_RETRIES} attempts: {last_err}"
    )


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    system_instruction: str | None = None,
) -> str:
    """Call the LLM and return plain text (used by the chat explainer in Phase 9)."""
    client = get_client()
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model or OPENROUTER_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as err:
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"LLM text call failed after {LLM_MAX_RETRIES} attempts: {last_err}"
    )


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences a model may add despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()
