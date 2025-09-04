from __future__ import annotations
from typing import Optional, Dict, Any
import time


def create_response(client,
                    model: str,
                    instructions: str,
                    input_text: str,
                    effort: str = "minimal",
                    previous_response_id: Optional[str] = None,
                    # retries is retained for compatibility but ignored when infinite_retries=True
                    retries: int = 5,
                    initial_delay: float = 3.0,
                    max_delay: float = 120.0,
                    verbose: bool = True,
                    max_output_tokens: Optional[int] = None,
                    infinite_retries: bool = True):
    """Wrapper for OpenAI Responses API with optional reasoning + chaining.

    - Adds reasoning only for gpt-5, gpt-5-mini, gpt-5-nano.
    - Includes previous_response_id if provided for conversational continuity.
    """
    params: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
    }
    # Only set max_output_tokens if caller explicitly provided it.
    if isinstance(max_output_tokens, int) and max_output_tokens > 0:
        params["max_output_tokens"] = max_output_tokens
    if model in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        params["reasoning"] = {"effort": effort}
    if previous_response_id:
        params["previous_response_id"] = previous_response_id
    attempt = 0
    while True:
        try:
            return client.responses.create(**params)
        except Exception as e:
            msg = str(e)
            if verbose:
                print(f"[rate-limit] error: {msg}")
            is_rate = False
            msg_lower = msg.lower()
            # Heuristics to detect rate limit (TPM/QPS)
            if ("rate limit" in msg_lower) or ("tokens per min" in msg_lower) or ("429" in msg_lower):
                is_rate = True
            # New/legacy SDKs may expose properties
            try:
                status = getattr(e, "status_code", None)
                if status == 429:
                    is_rate = True
            except Exception:
                pass

            # Only loop on rate limits; otherwise, raise immediately
            if not is_rate:
                raise
            # If finite retries requested and exceeded, raise
            if (not infinite_retries) and (attempt >= retries):
                raise

            # Simple linear backoff: 10s, 20s, 30s ... capped at 60s
            total = int(min(max_delay, 10 * (attempt + 1)))
            if verbose:
                for s in range(total, 0, -1):
                    print(f"[rate-limit] retrying in {s}s...", end="\r", flush=True)
                    time.sleep(1)
                # clear the line after countdown
                print(" " * 40, end="\r", flush=True)
            else:
                time.sleep(total)
            attempt += 1


def extract_usage(resp) -> Optional[Dict[str, int]]:
    """Best-effort extraction of usage tokens from a Responses API result.

    Returns a dict: {"input_tokens": int, "output_tokens": int, "total_tokens": int}
    or None if unavailable.
    """
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        # usage may be an object with attributes or a plain dict
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", input_tokens)
            output_tokens = usage.get("output_tokens", output_tokens)
            total_tokens = usage.get("total_tokens", total_tokens)
        # compute total if not provided
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens) + int(output_tokens)
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }
    except Exception:
        return None
