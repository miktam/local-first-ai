"""
ollama_client.py — Minimal Ollama wrapper.

Trust zone: this is the LLM boundary. Anything passed to `generate()` is
exposed to the model. Callers must ensure the prompt contains only
anonymized data — never raw names, never the vocab store, never the salt.

The default `think=False` is deliberate. The blog's earlier benchmark
showed thinking mode adds 5-15x token overhead on simple tasks and can
run away catastrophically. Summarisation and probing don't need it.
"""

import requests
from typing import Optional


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4-think:26b"


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    think: bool = False,
    host: str = DEFAULT_HOST,
    timeout: int = 300,
) -> str:
    """Send a prompt to Ollama and return the response text."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"think": think},
    }
    if system:
        body["system"] = system

    r = requests.post(f"{host}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def healthcheck(host: str = DEFAULT_HOST, timeout: int = 5) -> bool:
    """Return True if Ollama is reachable. Useful as a pre-flight check."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False
