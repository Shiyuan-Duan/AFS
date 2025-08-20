"""
Vanilla two-agent chat loop.
No tools, no file writes. Just conversation between Scientist and Assistant.
Outputs a JSON payload indicating conversation_only mode.
"""

from __future__ import annotations
import os, json, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

from dotenv import load_dotenv
from .llm_client import call_llm
from .storage import make_run_folder, append_jsonl
from .tools import (
    ToolContext,
    write_text_file,
    run_python,
    write_json_file,
    read_text_file,
    render_report as tool_render_report,
    ingest_data,
)

load_dotenv()

# --------------------------- Config ---------------------------

@dataclass
class AgentConfig:
    scientist_model: str
    assistant_model: str
    temperature: float = 0.2
    reasoning_effort: str = "medium"
    max_output_tokens: int = 1024

# --------------------------- IO ---------------------------

def load_task(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_prompts() -> Dict[str, str]:
    """Load role prompts from src/agents/, with simple defaults."""
    root = Path(__file__).parent / "agents"
    def read_or_default(name: str, default: str) -> str:
        p = root / name
        return p.read_text(encoding="utf-8") if p.exists() else default

    scientist_md = read_or_default("scientist_prompt.md", "You are the Scientist.")
    # Guide Assistant to either chat or propose an action plan in JSON.
    assistant_md = read_or_default(
        "assistant_prompt.md",
        (
            "You are the Assistant. Respond in JSON only. Either: "
            '{"type":"message","content":"<concise reply>"} OR '
            '{"type":"action_plan","plan":[{"op":"write_code","path":"code/script.py","language":"python","content":"..."},{"op":"run_python","path":"code/script.py","args":[],"timeout_sec":60,"save":[{"source":"stdout","to":"outputs/stdout.txt"}]}],"finalize":{"use":["outputs/stdout.txt"],"instructions":"Summarize results."}}'
        ),
    )
    return {"scientist": scientist_md, "assistant": assistant_md}

# --------------------------- Chat helpers ---------------------------

def system_message(content: str) -> Dict[str, str]:
    return {"role": "system", "content": content}

def user_message(content: str) -> Dict[str, str]:
    return {"role": "user", "content": content}

def assistant_message(content: str) -> Dict[str, str]:
    return {"role": "assistant", "content": content}

# --------------------------- Parsing helpers ---------------------------

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # drop the first fence line
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # drop the last fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return t

def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first top-level {...} JSON object, accounting for strings/escapes."""
    s = text
    start = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if start is None:
            if ch == '{':
                start = i
                depth = 1
                in_str = False
                esc = False
        else:
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
    return None

def _parse_assistant_json(raw: str) -> tuple[Optional[Dict[str, Any]], Optional[str], list[str]]:
    """
    Try to parse the assistant output into a JSON object. Returns (obj, used_text, warnings).
    Attempts strict json, then strips code fences, then extracts first JSON object.
    """
    warns: list[str] = []
    txt = raw.strip()
    # First try strict
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj, txt, warns
    except Exception as e:
        warns.append(f"strict_json_parse_failed: {e}")

    # Strip code fences and retry
    stripped = _strip_code_fences(txt)
    if stripped != txt:
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                warns.append("parsed_after_strip_code_fences")
                return obj, stripped, warns
        except Exception as e:
            warns.append(f"strip_fences_parse_failed: {e}")

    # Extract first JSON object
    frag = _extract_first_json_object(stripped)
    if frag:
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                warns.append("parsed_from_first_json_object")
                return obj, frag, warns
        except Exception as e:
            warns.append(f"first_object_parse_failed: {e}")

    return None, None, warns

def _validate_artifacts(run_root: Path, artifacts: list[str]) -> tuple[bool, list[Dict[str, Any]]]:
    """Basic sanity checks for outputs referenced in finalize.use.
    - file exists under run_root
    - size > 0 bytes
    - if .json: loadable and not empty (obj truthy)
    - if .csv/.txt: at least one non-empty line
    - if image (png/jpg): size > 100 bytes
    Returns (ok, details)
    """
    results: list[Dict[str, Any]] = []
    all_ok = True
    for rel in artifacts or []:
        # guard relative
        rel_s = str(rel).lstrip('/\\')
        p = (run_root / rel_s).resolve()
        ok = True
        info: Dict[str, Any] = {"path": str(p)}
        try:
            if not p.exists() or not p.is_file():
                ok = False
                info["error"] = "not_found"
            else:
                size = p.stat().st_size
                info["size"] = size
                if size <= 0:
                    ok = False
                    info["error"] = "empty_file"
                else:
                    suf = p.suffix.lower()
                    if suf == ".json":
                        try:
                            import json as _json
                            with open(p, 'r', encoding='utf-8') as f:
                                obj = _json.load(f)
                            if not obj:
                                ok = False
                                info["error"] = "empty_json"
                        except Exception as e:
                            ok = False
                            info["error"] = f"json_error: {e}"
                    elif suf in (".csv", ".tsv", ".txt"):
                        try:
                            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = [ln.strip() for ln in f.readlines()[:5]]
                            if not any(ln for ln in lines):
                                ok = False
                                info["error"] = "empty_text"
                        except Exception as e:
                            ok = False
                            info["error"] = f"text_error: {e}"
                    elif suf in (".png", ".jpg", ".jpeg"):
                        if size < 100:
                            ok = False
                            info["error"] = "image_too_small"
        except Exception as e:
            ok = False
            info["error"] = f"stat_error: {e}"
        results.append({"ok": ok, **info})
        if not ok:
            all_ok = False
    return all_ok, results

# --------------------------- Dialogue loop ---------------------------

def run_dialogue(
    task_yaml: str,
    run_name: str | None = None,
    max_steps: int = 12,
    dry_run: bool = False,
) -> str:
    """
    Vanilla chat between Scientist and Assistant. No actions or tools.
    Returns a JSON string with {mode: conversation_only, transcript: [...], meta: {...}}.
    """
    task = load_task(task_yaml)
    task_section = task.get("task", {})
    title = task_section.get("title", "study")
    goal = task_section.get("goal", "")
    deliverables = task_section.get("deliverables", [])
    constraints = task_section.get("constraints", [])
    inputs = task_section.get("inputs", {})
    dataset_root = inputs.get("dataset_root")

    prompts = load_prompts()
    cfg = AgentConfig(
        scientist_model=os.getenv("SCIENTIST_MODEL", "gpt-5-thinking"),
        assistant_model=os.getenv("ASSISTANT_MODEL", "gpt-5-nano-reason"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        reasoning_effort=os.getenv("REASONING_EFFORT", "medium"),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1024")),
    )

    # Histories per role
    scientist_history: List[Dict[str, str]] = [system_message(prompts["scientist"])]
    # Strict guardrail so Scientist does not propose plans or self-execute
    scientist_history.append(system_message(
        "Scientist Output Protocol (STRICT):\n"
        "- Do not propose to execute or draft an action plan yourself.\n"
        "- Do not ask for confirmation to proceed. Issue a direct instruction to the Assistant instead.\n"
        "- Structure: brief analysis (1–2 sentences) → imperative instruction to Assistant → one question requesting results.\n"
        "- No JSON/code/paths from you."
    ))
    assistant_history: List[Dict[str, str]] = [system_message(prompts["assistant"])]
    assistant_protocol = (
        "OUTPUT PROTOCOL (STRICT): Respond with JSON only. Either\n"
        "1) {\"type\":\"message\", \"content\":\"<concise reply>\"}  (message.content may include Markdown and fenced code blocks)\n"
        "2) {\"type\":\"action_plan\", \"plan\":[\n"
        "   {\"op\":\"write_code\",\"path\":\"code/script.py\",\"language\":\"python\",\"content\":\"import os\\nprint(\\\"hello\\\")\\n\"},\n"
        "   {\"op\":\"run_python\",\"path\":\"code/script.py\",\"args\":[],\"timeout_sec\":60,\"save\":[{\"source\":\"stdout\",\"to\":\"outputs/stdout.txt\"}]},\n"
        "   {\"op\":\"render_report\",\"title\":\"...\",\"abstract\":\"...\",\"sections\":[{\"title\":\"...\",\"text\":\"...\",\"figure_paths\":[\"outputs/fig.png\"]}],\"conclusions\":[\"...\"]},\n"
        "   {\"op\":\"ingest_data\",\"src_path\":\"/abs/path/to/dataset\",\"dest_relpath\":\"data/raw\"}\n"
        " ], \"finalize\":{\"use\":[\"outputs/stdout.txt\",\"reports/report.md\"], \"instructions\":\"Summarize results.\"}}\n"
        "JSON RULES: No text outside the JSON. Escape newlines as \\n and quotes as \\\". For write_code, put the entire code in one JSON string (no backticks)."
        " The Scientist will not provide JSON, file paths, or step lists; infer concrete steps and choose paths under code/, data/, outputs/, reports/."
        " If external data is outside the run folder, either read via the absolute dataset_root in Inputs or ingest it into data/ using ingest_data."
    )
    assistant_history.append(system_message(assistant_protocol))
    # Provide task context up front
    ctx_msg = (
        "Task context:\n"
        f"Title: {title}\n"
        f"Goal: {goal}\n"
        f"Deliverables: {deliverables}\n"
        f"Constraints: {constraints}\n"
        f"Inputs: {{'dataset_root': '{dataset_root}'}}"
    )
    assistant_history.append(system_message(ctx_msg))
    assistant_protocol = (
        "OUTPUT PROTOCOL (STRICT): Respond with JSON only. Either\n"
        "1) {\"type\":\"message\", \"content\":\"<concise reply>\"}\n"
        "2) {\"type\":\"action_plan\", \"plan\":[\n"
        "   {\"op\":\"write_code\",\"path\":\"code/script.py\",\"language\":\"python\",\"content\":\"...\"},\n"
        "   {\"op\":\"run_python\",\"path\":\"code/script.py\",\"args\":[],\"timeout_sec\":60,\"save\":[{\"source\":\"stdout\",\"to\":\"outputs/stdout.txt\"}]},\n"
        "   {\"op\":\"render_report\",\"title\":\"...\",\"abstract\":\"...\",\"sections\":[{\"title\":\"...\",\"text\":\"...\",\"figure_paths\":[\"outputs/fig.png\"]}],\"conclusions\":[\"...\"]}\n"
        " ], \"finalize\":{\"use\":[\"outputs/stdout.txt\",\"reports/report.md\"], \"instructions\":\"Summarize results.\"}}\n"
        "No markdown fences outside the JSON. The Scientist will not provide JSON, file paths, or step lists; infer the concrete steps and choose appropriate paths under code/, data/, outputs/, reports/."
    )
    assistant_history.append(system_message(assistant_protocol))
    assistant_protocol = (
        "OUTPUT PROTOCOL (STRICT): Respond with JSON only. Either\n"
        "1) {\"type\":\"message\", \"content\":\"<concise reply>\"}\n"
        "2) {\"type\":\"action_plan\", \"plan\":[\n"
        "   {\"op\":\"write_code\",\"path\":\"code/script.py\",\"language\":\"python\",\"content\":\"...\"},\n"
        "   {\"op\":\"run_python\",\"path\":\"code/script.py\",\"args\":[],\"timeout_sec\":60,\"save\":[{\"source\":\"stdout\",\"to\":\"outputs/stdout.txt\"}]},\n"
        "   {\"op\":\"render_report\",\"title\":\"...\",\"abstract\":\"...\",\"sections\":[{\"title\":\"...\",\"text\":\"...\",\"figure_paths\":[\"outputs/fig.png\"]}],\"conclusions\":[\"...\"]}\n"
        " ], \"finalize\":{\"use\":[\"outputs/stdout.txt\",\"reports/report.md\"], \"instructions\":\"Summarize results.\"}}\n"
        "No markdown fences outside the JSON."
    )
    assistant_history.append(system_message(assistant_protocol))

    kickoff = (
        "Task context:\n"
        f"Title: {title}\n"
        f"Goal: {goal}\n"
        f"Deliverables: {deliverables}\n"
        f"Constraints: {constraints}\n\n"
        "In 2–5 sentences, state the most important requirement and ask the next most important question. "
        "Do not include any JSON, code, file paths, or execution steps. The Assistant will decide whether actions are needed."
    )
    scientist_history.append(user_message(kickoff))

    transcript: List[Dict[str, str]] = []

    for step in range(max_steps):
        # Scientist speaks
        if dry_run:
            sc_text = f"[dry-run] Scientist message {step+1} about: {title}"
        else:
            sc_resp = call_llm(
                model=cfg.scientist_model,
                messages=scientist_history,
                temperature=cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
                max_output_tokens=cfg.max_output_tokens,
            )
            sc_text = (sc_resp.text or "").strip()
        transcript.append({"role": "scientist", "content": sc_text})
        # Feed to assistant
        assistant_history.append(user_message(sc_text))

        # Assistant replies
        if dry_run:
            as_text = f"[dry-run] Assistant reply {step+1} to the Scientist."
        else:
            as_resp = call_llm(
                model=cfg.assistant_model,
                messages=assistant_history,
                temperature=cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
                max_output_tokens=cfg.max_output_tokens,
            )
            as_text = (as_resp.text or "").strip()
        transcript.append({"role": "assistant", "content": as_text})
        # Feed back to scientist
        scientist_history.append(user_message(as_text))

    payload = {
        "mode": "conversation_only",
        "task": {"title": title, "goal": goal},
        "meta": {
            "scientist_model": cfg.scientist_model,
            "assistant_model": cfg.assistant_model,
            "steps": max_steps,
            "run_name": run_name,
        },
        "transcript": transcript,
    }
    return json.dumps(payload, ensure_ascii=False)


def stream_dialogue(
    task_yaml: str,
    run_name: str | None = None,
    max_steps: int = 12,
    dry_run: bool = False,
    execute_actions: bool = False,
    until_done: bool = False,
    data_root: str | None = None,
):
    """
    Generator that yields conversation events as they occur, then a final payload.
    Yields dicts like:
      {"type":"message", "step": n, "role": "scientist"|"assistant", "content": "..."}
      {"type":"final", "payload": <jsonable dict>}
    """
    task = load_task(task_yaml)
    title = task.get("task", {}).get("title", "study")
    goal = task.get("task", {}).get("goal", "")
    deliverables = task.get("task", {}).get("deliverables", [])
    constraints = task.get("task", {}).get("constraints", [])

    prompts = load_prompts()
    cfg = AgentConfig(
        scientist_model=os.getenv("SCIENTIST_MODEL", "gpt-5-thinking"),
        assistant_model=os.getenv("ASSISTANT_MODEL", "gpt-5-nano-reason"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        reasoning_effort=os.getenv("REASONING_EFFORT", "medium"),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1024")),
    )

    scientist_history: List[Dict[str, str]] = [system_message(prompts["scientist"])]
    assistant_history: List[Dict[str, str]] = [system_message(prompts["assistant"])]

    kickoff = (
        "Task context:\n"
        f"Title: {title}\n"
        f"Goal: {goal}\n"
        f"Deliverables: {deliverables}\n"
        f"Constraints: {constraints}\n\n"
        "Please start the discussion."
    )
    scientist_history.append(user_message(kickoff))
    # Encourage explicit termination when satisfied
    scientist_history.append(system_message("When the task is fully satisfied, reply with a single word: DONE."))

    transcript: List[Dict[str, str]] = []

    # Always create a run folder to log the conversation
    name_hint = run_name or ("chat-actions" if execute_actions else "chat")
    run_paths = make_run_folder("runs", name_hint=name_hint)
    ctx: ToolContext | None = None
    transcript_path = run_paths.logs / "transcript.jsonl"

    def log_event(ev: Dict[str, Any]):
        try:
            append_jsonl(transcript_path, ev)
        except Exception:
            pass

    yield {"type": "run_root", "path": str(run_paths.root)}
    log_event({"type": "run_root", "path": str(run_paths.root)})

    # Prepare ToolContext only if we plan to execute actions
    if execute_actions:
        ctx = ToolContext(
            run_root=run_paths.root,
            code_dir=run_paths.code,
            data_dir=run_paths.data,
            outputs_dir=run_paths.outputs,
            reports_dir=run_paths.reports,
            logs_dir=run_paths.logs,
        )
        # Auto-ingest data folder if provided
        if data_root:
            try:
                res = ingest_data(ctx, data_root, dest_relpath="data/")
                if res.get("ok"):
                    ev = {"type": "data_copied", "details": {k: res.get(k) for k in ("src","dest","files")}}
                    yield ev
                    log_event(ev)
                    # Inform Assistant explicitly where the dataset lives inside this run
                    assistant_history.append(system_message(
                        "Dataset is available under the run folder at 'data/'. "
                        "If your task refers to 'training2017', the full relative path is 'data/training2017'. "
                        "Prefer using list_dir and read_text_file on paths under data/."
                    ))
                else:
                    ev = {"type": "data_copy_error", "error": res.get("error", "unknown")}
                    yield ev
                    log_event(ev)
            except Exception as e:
                ev = {"type": "data_copy_error", "error": str(e)}
                yield ev
                log_event(ev)

    # Offline mode if no API key or explicit dry_run
    offline = dry_run or not bool(os.getenv("OPENAI_API_KEY"))
    if offline and not dry_run:
        ev = {"type": "warning", "message": "OPENAI_API_KEY not set; running in offline dry-run mode."}
        yield ev
        log_event(ev)

    steps_limit = max_steps if not until_done else max(max_steps, 200)
    stopped_by = None
    import re
    last_sc_msg: str | None = None
    for step in range(steps_limit):
        # Scientist speaks
        if offline:
            sc_text = f"[dry-run] Scientist message {step+1} about: {title}"
        else:
            sc_resp = call_llm(
                model=cfg.scientist_model,
                messages=scientist_history,
                temperature=cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
                max_output_tokens=cfg.max_output_tokens,
            )
            sc_text = (sc_resp.text or "").strip()
        transcript.append({"role": "scientist", "content": sc_text})
        last_sc_msg = sc_text
        ev = {"type": "message", "step": step + 1, "role": "scientist", "content": sc_text}
        yield ev
        log_event(ev)
        # Check for explicit DONE from Scientist
        if until_done and re.search(r"\bDONE\b", sc_text, flags=re.IGNORECASE):
            stopped_by = "scientist_done"
            break
        assistant_history.append(user_message(sc_text))

        # Assistant replies (may emit a conversation message or an action plan JSON)
        if offline:
            as_text = f"[dry-run] Assistant reply {step+1} to the Scientist."
            transcript.append({"role": "assistant", "content": as_text})
            ev = {"type": "message", "step": step + 1, "role": "assistant", "content": as_text}
            yield ev
            log_event(ev)
            scientist_history.append(user_message(as_text))
            continue

        as_resp = call_llm(
            model=cfg.assistant_model,
            messages=assistant_history,
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
            max_output_tokens=cfg.max_output_tokens,
        )
        raw = (as_resp.text or "").strip()

        # Try robust parse
        parsed, used_text, parse_warns = _parse_assistant_json(raw)
        looks_like_plan = ('"type"' in raw and 'action_plan' in raw) or ('op"' in raw and 'write_code' in raw)
        if parsed is None and looks_like_plan:
            # Defer to Scientist's judgment. Surface an action warning and nudge Scientist to decide.
            ev = {
                "type": "action_warning",
                "message": "Assistant output looks like an action plan but is not valid JSON.",
                "details": parse_warns,
                "raw_preview": raw[:800],
            }
            yield ev
            log_event(ev)
            scientist_history.append(system_message(
                "Assistant's last reply appears to be an action plan but was not valid JSON. "
                "Decide: If you want to execute it, instruct the Assistant to resend STRICT JSON action_plan only (no fences), "
                "and ensure write_code.content is a valid JSON string (use \\n for newlines and \\\" for quotes). "
                "Otherwise, instruct the Assistant to proceed conversationally without execution."
            ))
            # Treat this turn as no assistant message; continue to next Scientist turn
            continue

        if not parsed or parsed.get("type") == "message":
            # If we had parse warnings, surface them
            if parse_warns:
                ev = {"type": "parse_warning", "details": parse_warns, "raw_preview": raw[:400]}
                yield ev
                log_event(ev)
            content = parsed.get("content") if parsed else raw
            transcript.append({"role": "assistant", "content": content})
            ev2 = {"type": "message", "step": step + 1, "role": "assistant", "content": content}
            yield ev2
            log_event(ev2)
            scientist_history.append(user_message(content))
            # Reinforce Scientist protocol before next turn
            scientist_history.append(system_message(
                "Remember: analyze the Assistant's results, then issue a direct instruction. "
                "Do not propose to execute or ask for confirmation; direct the Assistant to act."
            ))
            continue

        # action_plan path
        plan: List[Dict[str, Any]] = parsed.get("plan") or []
        finalize: Dict[str, Any] = parsed.get("finalize") or {}
        # Heuristic guard: if Scientist asked to modify a specific file, ensure we edit that path instead of creating a new file
        try:
            edit_intent = False
            paths_requested: list[str] = []
            if last_sc_msg:
                # keywords indicating modification
                if re.search(r"\b(remove|modify|fix|delete|drop|change|patch|edit)\b", last_sc_msg, flags=re.IGNORECASE):
                    edit_intent = True
                    paths_requested = re.findall(r"\b(code/[A-Za-z0-9_./\\-]+\.py)\b", last_sc_msg)
            if edit_intent and paths_requested:
                requested_set = set(paths_requested)
                write_paths = [str(step.get("path")) for step in plan if (step.get("op") or "").lower() == "write_code"]
                if write_paths and not any(p in requested_set for p in write_paths):
                    warn = {
                        "type": "plan_path_warning",
                        "message": "Scientist requested in-place edit of a specific file; your plan writes a different path.",
                        "requested": list(requested_set),
                        "proposed": write_paths,
                    }
                    yield warn
                    log_event(warn)
                    assistant_history.append(system_message(
                        "The Scientist asked to modify specific file(s): " + ", ".join(requested_set) + ". "
                        "Do NOT create a differently named file. Resend STRICT JSON action_plan that reads the target file(s) and overwrites the SAME path(s) with the fix."
                    ))
                    # Skip executing this mismatched plan; continue to next turn
                    continue
        except Exception:
            pass
        ev = {
            "type": "plan",
            "step": step + 1,
            "plan": plan,
            "finalize": finalize,
            "note": "Assistant requested code execution. Showing plan before execution.",
        }
        yield ev
        log_event(ev)

        if not execute_actions:
            # Do not execute; feed a note back and continue conversation
            note = "Plan noted. Execution is disabled; please provide a conversational summary instead."
            transcript.append({"role": "assistant", "content": raw})
            ev = {"type": "skipped", "reason": "execute_actions is False"}
            yield ev
            log_event(ev)
            scientist_history.append(user_message(note))
            continue

        # Execute the plan in a temporary run folder
        if run_paths is None or ctx is None:
            run_paths = make_run_folder("runs", name_hint=run_name or "chat-actions")
            ctx = ToolContext(
                run_root=run_paths.root,
                code_dir=run_paths.code,
                data_dir=run_paths.data,
                outputs_dir=run_paths.outputs,
                reports_dir=run_paths.reports,
                logs_dir=run_paths.logs,
            )

        exec_log: List[Dict[str, Any]] = []
        for i, step_def in enumerate(plan, start=1):
            op = (step_def.get("op") or "").lower()
            try:
                if op == "write_code":
                    path = step_def.get("path", "code/snippet.py")
                    content = step_def.get("content", "")
                    res = write_text_file(ctx, path, content, overwrite=True)
                    exec_log.append({"ok": res.get("ok", False), "op": op, "path": res.get("path")})
                elif op == "write_text_file":
                    path = step_def.get("path")
                    content = step_def.get("content", "")
                    res = write_text_file(ctx, path, content, overwrite=True)
                    exec_log.append({"ok": res.get("ok", False), "op": op, "path": res.get("path")})
                elif op == "run_python":
                    path = step_def.get("path") or step_def.get("script")
                    args = step_def.get("args") or []
                    timeout_sec = int(step_def.get("timeout_sec", 60))
                    env_vars = step_def.get("env") or None
                    res = run_python(ctx, path=path, args=args, timeout_sec=timeout_sec, env=env_vars)
                    # Optionally save stdout/stderr
                    for save in step_def.get("save", []) or []:
                        src = (save.get("source") or "").lower()
                        to = save.get("to")
                        if src in {"stdout", "stderr"} and to:
                            content = res.get(src, "")
                            write_text_file(ctx, to, content, overwrite=True)
                    exec_log.append({
                        "ok": res.get("ok", False),
                        "op": op,
                        "path": res.get("path"),
                        "exit_code": res.get("exit_code"),
                        "stdout": res.get("stdout", "")[:4000],
                        "stderr": res.get("stderr", "")[:4000],
                    })
                elif op in ("read_text_file", "read_file"):
                    rel = step_def.get("relpath") or step_def.get("path")
                    res = read_text_file(ctx, rel)
                    content = res.get("content", "") if res.get("ok") else ""
                    exec_log.append({
                        "ok": res.get("ok", False),
                        "op": op,
                        "path": res.get("path"),
                        "bytes": len(content.encode("utf-8")) if content else 0,
                        "preview": content[:1000] if content else "",
                    })
                elif op == "list_dir":
                    rel = step_def.get("relpath") or step_def.get("path") or "."
                    res = list_dir(ctx, rel)
                    exec_log.append({
                        "ok": res.get("ok", False),
                        "op": op,
                        "path": res.get("path"),
                        "items": res.get("items", [])[:200],
                    })
                elif op == "render_report":
                    title = step_def.get("title") or "Report"
                    abstract = step_def.get("abstract") or ""
                    sections = step_def.get("sections") or []
                    figures = step_def.get("figures") or []
                    conclusions = step_def.get("conclusions") or []
                    res = tool_render_report(
                        ctx,
                        title=title,
                        abstract=abstract,
                        sections=sections,
                        figures=figures,
                        conclusions=conclusions,
                    )
                    exec_log.append({
                        "ok": res.get("ok", False),
                        "op": op,
                        "markdown": res.get("markdown"),
                        "html": res.get("html"),
                    })
                elif op == "check_env":
                    pkgs = step_def.get("packages") or None
                    from .tools import check_env as _check_env
                    res = _check_env(pkgs)
                    exec_log.append({"ok": res.get("ok", False), "op": op, "details": res.get("details", [])})
                elif op == "install_python_packages":
                    pkgs = step_def.get("packages") or []
                    upg = bool(step_def.get("upgrade", False))
                    idx = step_def.get("index_url")
                    to = int(step_def.get("timeout_sec", 600))
                    from .tools import install_python_packages as _pip_install
                    res = _pip_install(ctx, pkgs, upgrade=upg, index_url=idx, timeout_sec=to)
                    exec_log.append({
                        "ok": res.get("ok", False),
                        "op": op,
                        "exit_code": res.get("exit_code"),
                        "stdout": (res.get("stdout") or "")[:2000],
                        "stderr": (res.get("stderr") or "")[:2000],
                        "cmd": res.get("cmd"),
                    })
                else:
                    exec_log.append({"ok": False, "op": op, "error": "unsupported op"})
            except Exception as e:
                exec_log.append({"ok": False, "op": op, "error": str(e)})

        ev = {"type": "execution", "step": step + 1, "run_root": str(run_paths.root), "log": exec_log}
        yield ev
        log_event(ev)

        # Ask the Assistant to self-review before reporting to Scientist
        artifacts_used = finalize.get("use") or []
        self_review = {
            "run_root": str(run_paths.root),
            "artifacts": artifacts_used,
            "exec_log_tail": exec_log[-3:] if exec_log else [],
        }
        instructions = (
            "Before addressing the Scientist, conduct a self-review: (1) skim your code for syntax/logic errors; "
            "(2) read recent logs/artifacts (e.g., outputs/*stdout.txt, *stderr.txt, CSV/JSON) and quote the exact failing lines if any; "
            "(3) check target/label for NaNs or non-numeric types (encode deterministically), ensure X and y masks align; "
            "(4) verify referenced outputs exist and are non-empty, and metrics/plots look plausible. "
            "If anything is off, respond with STRICT JSON {\\\"type\\\":\\\"action_plan\\\"} to fix and re-run (use read_text_file to inspect files if needed). "
            "Otherwise, respond with STRICT JSON {\\\"type\\\":\\\"message\\\", \\\"content\\\":\\\"...\\\"} that (a) summarizes at a PhD level what you executed and key results, (b) explicitly lists which steps succeeded vs failed, and (c) quotes key stderr lines for any failures. No text outside JSON."
        )
        assistant_history.append(user_message("SELF-REVIEW CONTEXT:\n" + json.dumps(self_review, ensure_ascii=False) + "\n" + instructions))
        as_post = call_llm(
            model=cfg.assistant_model,
            messages=assistant_history,
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
            max_output_tokens=cfg.max_output_tokens,
        )
        final_raw = (as_post.text or "").strip()
        final_parsed, _, _ = _parse_assistant_json(final_raw)
        if final_parsed and final_parsed.get("type") == "action_plan":
            # Execute corrective plan immediately (do not bounce back to Scientist first)
            for attempt in range(2):
                plan2: List[Dict[str, Any]] = final_parsed.get("plan") or []
                finalize2: Dict[str, Any] = final_parsed.get("finalize") or {}
                yield {"type": "plan", "step": step + 1, "plan": plan2, "finalize": finalize2, "note": "Self-review corrective plan."}
                exec_log2: List[Dict[str, Any]] = []
                for i2, step_def2 in enumerate(plan2, start=1):
                    op2 = (step_def2.get("op") or "").lower()
                    try:
                        if op2 == "write_code":
                            path2 = step_def2.get("path", "code/snippet.py")
                            content2 = step_def2.get("content", "")
                            res2 = write_text_file(ctx, path2, content2, overwrite=True)
                            exec_log2.append({"ok": res2.get("ok", False), "op": op2, "path": res2.get("path")})
                        elif op2 == "write_text_file":
                            path2 = step_def2.get("path")
                            content2 = step_def2.get("content", "")
                            res2 = write_text_file(ctx, path2, content2, overwrite=True)
                            exec_log2.append({"ok": res2.get("ok", False), "op": op2, "path": res2.get("path")})
                        elif op2 == "run_python":
                            path2 = step_def2.get("path") or step_def2.get("script")
                            args2 = step_def2.get("args") or []
                            timeout2 = int(step_def2.get("timeout_sec", 60))
                            env2 = step_def2.get("env") or None
                            res2 = run_python(ctx, path=path2, args=args2, timeout_sec=timeout2, env=env2)
                            for save2 in step_def2.get("save", []) or []:
                                src2 = (save2.get("source") or "").lower()
                                to2 = save2.get("to")
                                if src2 in {"stdout", "stderr"} and to2:
                                    content_save = res2.get(src2, "")
                                    write_text_file(ctx, to2, content_save, overwrite=True)
                            exec_log2.append({
                                "ok": res2.get("ok", False),
                                "op": op2,
                                "path": res2.get("path"),
                                "exit_code": res2.get("exit_code"),
                                "stdout": res2.get("stdout", "")[:4000],
                                "stderr": res2.get("stderr", "")[:4000],
                            })
                        elif op2 in ("read_text_file", "read_file"):
                            rel2 = step_def2.get("relpath") or step_def2.get("path")
                            res2 = read_text_file(ctx, rel2)
                            content2 = res2.get("content", "") if res2.get("ok") else ""
                            exec_log2.append({
                                "ok": res2.get("ok", False),
                                "op": op2,
                                "path": res2.get("path"),
                                "bytes": len(content2.encode("utf-8")) if content2 else 0,
                                "preview": content2[:1000] if content2 else "",
                            })
                        elif op2 == "list_dir":
                            rel2 = step_def2.get("relpath") or step_def2.get("path") or "."
                            res2 = list_dir(ctx, rel2)
                            exec_log2.append({
                                "ok": res2.get("ok", False),
                                "op": op2,
                                "path": res2.get("path"),
                                "items": res2.get("items", [])[:200],
                            })
                        elif op2 == "render_report":
                            title2 = step_def2.get("title") or "Report"
                            abstract2 = step_def2.get("abstract") or ""
                            sections2 = step_def2.get("sections") or []
                            figures2 = step_def2.get("figures") or []
                            conclusions2 = step_def2.get("conclusions") or []
                            res2 = tool_render_report(ctx, title=title2, abstract=abstract2, sections=sections2, figures=figures2, conclusions=conclusions2)
                            exec_log2.append({"ok": res2.get("ok", False), "op": op2, "markdown": res2.get("markdown"), "html": res2.get("html")})
                        elif op2 == "check_env":
                            pkgs2 = step_def2.get("packages") or None
                            from .tools import check_env as _check_env2
                            res2 = _check_env2(pkgs2)
                            exec_log2.append({"ok": res2.get("ok", False), "op": op2, "details": res2.get("details", [])})
                        elif op2 == "install_python_packages":
                            pkgs2 = step_def2.get("packages") or []
                            upg2 = bool(step_def2.get("upgrade", False))
                            idx2 = step_def2.get("index_url")
                            to2 = int(step_def2.get("timeout_sec", 600))
                            from .tools import install_python_packages as _pip_install2
                            res2 = _pip_install2(ctx, pkgs2, upgrade=upg2, index_url=idx2, timeout_sec=to2)
                            exec_log2.append({
                                "ok": res2.get("ok", False),
                                "op": op2,
                                "exit_code": res2.get("exit_code"),
                                "stdout": (res2.get("stdout") or "")[:2000],
                                "stderr": (res2.get("stderr") or "")[:2000],
                                "cmd": res2.get("cmd"),
                            })
                        elif op2 == "ingest_data":
                            srcp2 = step_def2.get("src_path")
                            dstrel2 = step_def2.get("dest_relpath", "data/")
                            res2 = ingest_data(ctx, srcp2, dest_relpath=dstrel2)
                            exec_log2.append({"ok": res2.get("ok", False), "op": op2, "src": res2.get("src"), "dest": res2.get("dest")})
                        else:
                            exec_log2.append({"ok": False, "op": op2, "error": "unsupported op"})
                    except Exception as e:
                        exec_log2.append({"ok": False, "op": op2, "error": str(e)})
                evx = {"type": "execution", "step": step + 1, "run_root": str(run_paths.root), "log": exec_log2}
                yield evx
                log_event(evx)
                # Ask for another self-review immediately
                self_review2 = {"run_root": str(run_paths.root), "artifacts": finalize2.get("use") or [], "exec_log_tail": exec_log2[-3:] if exec_log2 else []}
                instructions2 = (
                    "Self-review again. If anything remains off, reply with STRICT JSON {\\\"type\\\":\\\"action_plan\\\"} to fix; "
                    "otherwise reply with STRICT JSON {\\\"type\\\":\\\"message\\\", \\\"content\\\":\\\"...\\\"} that (a) summarizes results at a PhD level, (b) explicitly lists which steps succeeded vs failed, and (c) quotes key stderr lines for any failures."
                )
                assistant_history.append(user_message("SELF-REVIEW CONTEXT:\n" + json.dumps(self_review2, ensure_ascii=False) + "\n" + instructions2))
                as_post2 = call_llm(
                    model=cfg.assistant_model,
                    messages=assistant_history,
                    temperature=cfg.temperature,
                    reasoning_effort=cfg.reasoning_effort,
                    max_output_tokens=cfg.max_output_tokens,
                )
                final_raw2 = (as_post2.text or "").strip()
                fp2, _, _ = _parse_assistant_json(final_raw2)
                if fp2 and fp2.get("type") == "action_plan":
                    final_parsed = fp2
                    continue
                # Got a final message
                final_msg2 = fp2.get("content") if isinstance(fp2, dict) else final_raw2
                transcript.append({"role": "assistant", "content": final_msg2})
                evm2 = {"type": "message", "step": step + 1, "role": "assistant", "content": final_msg2}
                yield evm2
                log_event(evm2)
                scientist_history.append(user_message(final_msg2))
                # Add run summary and explicit review instruction for the Scientist
                run_summary2 = {
                    "exec_log_tail": exec_log2[-3:] if exec_log2 else [],
                    "artifacts": finalize2.get("use") or [],
                }
                scientist_history.append(system_message(
                    "Review the latest run outputs and logs (summary below). Identify concrete errors, missing files, empty or implausible outputs, or design issues. "
                    "Then give a directive with explicit fix recommendations for the Assistant to act on next.\n" +
                    json.dumps(run_summary2, ensure_ascii=False)
                ))
                break
            continue
        final_msg = final_parsed.get("content") if isinstance(final_parsed, dict) else final_raw
        transcript.append({"role": "assistant", "content": final_msg})
        evm = {"type": "message", "step": step + 1, "role": "assistant", "content": final_msg}
        yield evm
        log_event(evm)
        scientist_history.append(user_message(final_msg))
        # Provide a concise run summary and instruct the Scientist to review and recommend fixes
        run_summary = {
            "exec_log_tail": exec_log[-3:] if exec_log else [],
            "artifacts": artifacts_used,
        }
        scientist_history.append(system_message(
            "Review the latest run outputs and logs (summary below). Identify concrete errors, missing files, empty or implausible outputs, or design issues. "
            "Then give a directive with explicit fix recommendations for the Assistant to act on next.\n" +
            json.dumps(run_summary, ensure_ascii=False)
        ))
        if until_done:
            scientist_history.append(user_message("If the above results satisfy the task, reply with DONE. Otherwise, ask the next most important question."))

    if stopped_by is None:
        stopped_by = "step_limit"

    payload = {
        "mode": "conversation_with_optional_actions" if execute_actions else "conversation_only",
        "task": {"title": title, "goal": goal},
        "meta": {
            "scientist_model": cfg.scientist_model,
            "assistant_model": cfg.assistant_model,
            "steps": max_steps,
            "run_name": run_name,
            "until_done": until_done,
            "stopped_by": stopped_by,
        },
        "transcript": transcript,
    }
    ev_final = {"type": "final", "payload": payload}
    yield ev_final
    log_event(ev_final)

# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Scientist ↔ Assistant vanilla chat loop")
    parser.add_argument("task_yaml", help="Path to a YAML task file")
    parser.add_argument("--rounds", type=int, default=6, help="Max chat rounds (default: 6)")
    args = parser.parse_args()

    try:
        final_payload = None
        for evt in stream_dialogue(args.task_yaml, max_steps=args.rounds):
            if evt.get("type") == "message":
                role = evt["role"]
                print(f"[{role} step {evt['step']}]\n{evt['content']}\n")
            elif evt.get("type") == "final":
                final_payload = json.dumps(evt["payload"], ensure_ascii=False)
        if final_payload:
            print(final_payload)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
