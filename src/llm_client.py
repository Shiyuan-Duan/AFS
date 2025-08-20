from __future__ import annotations

import os, json, typing as t
from dataclasses import dataclass
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL".replace("-", "_")) or os.getenv("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID,
)

@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict]
    raw: dict

def _select_api_style(model: str) -> str:
    """Choose API style by model name.
    - Use Responses API for models that support reasoning (gpt-5, gpt-5-mini, gpt-5-nano, thinking variants).
    - Use Chat Completions for chat-only variants (e.g., gpt-5-chat-latest) and other non-reasoning models.
    """
    m = (model or "").lower()
    if "chat" in m:
        return "chat"
    if any(k in m for k in ["gpt-5-nano", "gpt-5-mini"]) or re.match(r"^gpt-5(?!-chat)", m) or any(k in m for k in ["thinking", "reason"]):
        return "responses"
    return "chat"

def _supports_reasoning_param(model: str) -> bool:
    """Return True if the model supports the Responses API 'reasoning' parameter.
    - True: gpt-5, gpt-5-mini, gpt-5-nano (and 'thinking'/'reason' variants)
    - False: chat-only variants like gpt-5-chat-latest and other non-reasoning models
    """
    m = (model or "").lower()
    if "chat" in m:
        return False
    if any(k in m for k in ["gpt-5-nano", "gpt-5-mini"]) or re.match(r"^gpt-5(?!-chat)", m) or any(k in m for k in ["thinking", "reason"]):
        return True
    return False

def _mk_messages_payload(messages: list[dict]) -> list[dict]:
    # Normalize messages to [{role, content}] where content is str
    norm = []
    for m in messages:
        norm.append({"role": m["role"], "content": m["content"]})
    return norm

# --------- Tools schema helpers ---------
def _mk_tools_for_responses(functions: list[dict]) -> list[dict]:
    """Responses API expects a flattened schema: each tool has top-level name/parameters."""
    tools = []
    for f in functions:
        tools.append({
            "type": "function",
            "name": f["name"],
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {"type": "object", "properties": {}}),
        })
    return tools

def _mk_tools_for_chat(functions: list[dict]) -> list[dict]:
    """Chat Completions expects {type:'function', function:{name,parameters}}."""
    return [{"type": "function", "function": f} for f in functions]

def _parse_tool_calls_from_chat(choice_msg) -> list[dict]:
    calls = []
    tool_calls = getattr(choice_msg, "tool_calls", None) or getattr(choice_msg, "tool_calls", None)
    if tool_calls:
        for c in tool_calls:
            try:
                calls.append({
                    "name": c.function.name,
                    "arguments": json.loads(c.function.arguments or "{}"),
                })
            except Exception:
                calls.append({
                    "name": c.function.name if hasattr(c, "function") else "unknown",
                    "arguments": c.function.arguments if hasattr(c, "function") else "{}",
                })
    return calls

def _parse_from_responses(resp) -> LLMResponse:
    """
    Parse Responses API output across SDK versions.
    Prefer .output blocks; fallback to .output_text.
    """
    text_parts = []
    tool_calls = []
    try:
        blocks = getattr(resp, "output", None) or []
        for b in blocks:
            tpe = getattr(b, "type", None)
            if tpe == "message":
                for cp in getattr(b, "content", []) or []:
                    if getattr(cp, "type", "") in ("output_text", "text"):
                        text_parts.append(getattr(cp, "text", ""))
            elif tpe == "tool_call":
                fn = getattr(b, "function", None)
                if fn:
                    try:
                        tool_calls.append({"name": fn.name, "arguments": json.loads(fn.arguments or "{}")})
                    except Exception:
                        tool_calls.append({"name": getattr(fn, "name", "unknown"), "arguments": getattr(fn, "arguments", "{}")})
        if not text_parts:
            ot = getattr(resp, "output_text", None)
            if isinstance(ot, str):
                text_parts.append(ot)
    except Exception:
        d = getattr(resp, "model_dump", lambda: {})() or {}
        if isinstance(d, dict):
            text_parts.append(d.get("output_text", ""))
    text = "\n".join([p for p in text_parts if p])
    return LLMResponse(text=text, tool_calls=tool_calls, raw=getattr(resp, "model_dump", lambda: resp)())

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_llm(
    model: str,
    messages: list[dict],
    functions: list[dict] | None = None,
    temperature: float = 0.2,
    reasoning_effort: str | None = "medium",
    max_output_tokens: int | None = 2048,
    api_style: str | None = None,
) -> LLMResponse:
    """
    Resilient wrapper:
    - Uses Responses API for reasoning models and keeps the request minimal:
      model + input + (optional) reasoning + (optional) tools[flattened].
    - Falls back to Chat Completions for non-reasoning models.
    """
    api_style = api_style or _select_api_style(model)
    msgs = _mk_messages_payload(messages)

    if api_style == "responses":
        tools = _mk_tools_for_responses(functions) if functions else None
        kwargs = dict(
            model=model,
            input=msgs,
        )
        if reasoning_effort and _supports_reasoning_param(model):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if tools:
            kwargs["tools"] = tools  # no tool_choice for Responses API

        try:
            resp = client.responses.create(**kwargs)
        except Exception:
            # Fallback: try without tools (minimal shape)
            if "tools" in kwargs:
                kwargs_min = dict(kwargs)
                kwargs_min.pop("tools", None)
                resp = client.responses.create(**kwargs_min)
            else:
                raise
        return _parse_from_responses(resp)

    # ------ Chat Completions fallback ------
    tools = _mk_tools_for_chat(functions) if functions else None
    kwargs = dict(
        model=model,
        messages=msgs,
        temperature=temperature,
    )
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0].message
    tool_calls = _parse_tool_calls_from_chat(choice)
    text = (choice.content or "").strip() if choice.content else ""
    return LLMResponse(text=text, tool_calls=tool_calls, raw=resp.model_dump() if hasattr(resp, "model_dump") else resp)
