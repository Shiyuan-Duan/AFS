import json
import re
from pathlib import Path
from typing import Optional, Dict

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

try:
    from .llm_utils import create_response, extract_usage
    from .orchestrator import Orchestrator, extract_first_json
    from .run_utils import append_jsonl, save_checkpoint
except Exception:
    from llm_utils import create_response, extract_usage
    from orchestrator import Orchestrator, extract_first_json
    from run_utils import append_jsonl, save_checkpoint


PI_INSTRUCTIONS = (
    "You are the Principal Investigator (PI).\n"
    "- Describe the task comprehensively and give a clear, actionable instruction to the student.\n"
    "- Be specific; include exactly one directive. Do NOT list multiple TODOs at once.\n"
    "- Do NOT instruct creating run folders (e.g., no 'runs/...'); the orchestrator already created a run.\n"
    "- Refer only to paths under these prefixes: code/, outputs/, data/, logs/.\n"
    "- If the Student reports syntax errors or code that fails to run, briefly re-iterate the task goal and instruct the Student to rewrite the code cleanly (avoid minor patches to broken code).\n"
    "- Reminder: scripts under code/ should reference the dataset via 'data/[TASK_SPECIFIC_DATASET]' relative paths."
)

STUDENT_INSTRUCTIONS = (
    "You are the Student.\n"
    "- Follow the PI's instruction and provide a concise answer (2–5 sentences).\n"
    "- Be direct and practical; avoid filler.\n"
    "- If a required code/data/output file is missing under the allowed prefixes (code/, outputs/, data/, logs/), prefer creating it deterministically (write_file + run_code) instead of spending time locating it.\n"
    "- When writing code under code/, refer to dataset files using 'data/[TASK_SPECIFIC_DATASET]' relative paths."
)


def _is_explicit_done(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.match(r"^\s*DONE\s*[.!?\"]?\s*$", t, flags=re.IGNORECASE))


def run_conversation_loop(
    *,
    client,
    console: Console,
    orch: Orchestrator,
    run_paths: Dict[str, Path],
    transcript_path: Path,
    task_text: str,
    task_path: str,
    start_turn: int,
    rounds: int,
    pi_model: str,
    student_model: str,
    pi_prev_id: Optional[str] = None,
    student_prev_id: Optional[str] = None,
    last_student: Optional[str] = None,
) -> None:
    console.print(Panel(f"PI_MODEL={pi_model}\nSTUDENT_MODEL={student_model}\nRun: {run_paths['root']}", title="Models", border_style="blue"))

    for turn in range(start_turn, start_turn + rounds):
        # 1) PI: describe/critique + instruct
        if last_student:
            pi_input = (
                "ROLE REMINDER (PI):\n" + PI_INSTRUCTIONS +
                "\n\nTask (YAML content):\n" + task_text +
                "\n\nContext: Student's previous answer to review and build upon:\n" + last_student +
                f"\n\nPI: Provide a concise directive for turn {turn}."
            )
        else:
            pi_input = (
                "ROLE REMINDER (PI):\n" + PI_INSTRUCTIONS +
                "\n\nTask (YAML content):\n" + task_text +
                f"\n\nPI: Describe the task comprehensively and instruct the student how to begin (turn {turn})."
            )
        usage = None
        try:
            _pi_prev = pi_prev_id
            pi_resp = create_response(client, pi_model, PI_INSTRUCTIONS, pi_input, effort="minimal", previous_response_id=_pi_prev)
            pi_text = (pi_resp.output_text or "").strip()
            pi_prev_id = getattr(pi_resp, "id", None)
            usage = extract_usage(pi_resp)
            append_jsonl(transcript_path, {"type": "llm_call", "who": "PI", "turn": turn, "previous_response_id": _pi_prev, "response_id": pi_prev_id, "usage": usage})
        except Exception as ex:
            err = str(ex)
            console.print(Panel(f"LLM error in PI:\n{err}", title="Rate limit / LLM Error", border_style="red"))
            append_jsonl(transcript_path, {"type": "llm_error", "turn": turn, "who": "PI", "error": err})
            pi_text = "[rate-limited] PI could not respond this turn; will retry next turn."
        if turn == 1:
            console.rule("PI ↔ Student")
        console.print(Panel(Markdown(pi_text), title=f"PI · turn {turn}", border_style="cyan"))
        # Show token usage if available
        try:
            if usage:
                console.print(Panel(f"in: {usage.get('input_tokens', 0)} · out: {usage.get('output_tokens', 0)} · total: {usage.get('total_tokens', 0)}", title="PI · tokens", border_style="cyan"))
        except Exception:
            pass
        append_jsonl(transcript_path, {"type": "message", "role": "pi", "turn": turn, "content": pi_text})
        if _is_explicit_done(pi_text):
            console.rule("PI signaled DONE — ending loop")
            append_jsonl(transcript_path, {"type": "done", "by": "pi", "turn": turn})
            break

        # 2) Student: phase=plan (STRICT JSON only)
        tmpl_path = Path(__file__).parent / "templates" / "student_plan_prompt.txt"
        try:
            plan_template = tmpl_path.read_text(encoding="utf-8")
        except Exception:
            plan_template = (
                "You are the Student. Reply with STRICT JSON only.\n"
                "{\"phase\":\"plan\",\"goal\":\"...\",\"rationale\":\"...\",\"steps\":[],\"quality_checks\":[]}\n"
            )
        plan_prompt = (
            "ROLE REMINDER (Student):\n" + STUDENT_INSTRUCTIONS +
            "\n\n" + plan_template + "\n\nPI instruction:\n" + pi_text
        )
        usage_plan = None
        try:
            _st_prev = student_prev_id
            student_plan_resp = create_response(client, student_model, STUDENT_INSTRUCTIONS, plan_prompt, effort="minimal", previous_response_id=_st_prev)
            student_prev_id = getattr(student_plan_resp, "id", None)
            student_plan_text = (student_plan_resp.output_text or "").strip()
            usage_plan = extract_usage(student_plan_resp)
            append_jsonl(transcript_path, {"type": "llm_call", "who": "Student-plan", "turn": turn, "previous_response_id": _st_prev, "response_id": student_prev_id, "usage": usage_plan})
        except Exception as ex:
            err = str(ex)
            console.rule("Student plan unavailable due to rate limit — skipping execution this turn")
            console.print(Panel(f"LLM error in Student plan:\n{err}", title="Rate limit / LLM Error", border_style="red"))
            append_jsonl(transcript_path, {"type": "plan_skipped", "turn": turn, "reason": "rate_limit", "error": err})
            continue
        plan_obj = extract_first_json(student_plan_text) or {}
        console.print(Panel(Markdown("```json\n" + json.dumps(plan_obj, ensure_ascii=False, indent=2) + "\n```"), title="Student · plan", border_style="magenta"))
        try:
            if usage_plan:
                console.print(Panel(f"in: {usage_plan.get('input_tokens', 0)} · out: {usage_plan.get('output_tokens', 0)} · total: {usage_plan.get('total_tokens', 0)}", title="Student-plan · tokens", border_style="magenta"))
        except Exception:
            pass
        append_jsonl(transcript_path, {"type": "student_plan", "turn": turn, "raw": student_plan_text, "parsed": plan_obj})

        # 3) Orchestrator executes plan
        exec_summary = orch.execute_plan(plan_obj)
        console.print(Panel(Markdown("```json\n" + json.dumps(exec_summary, ensure_ascii=False, indent=2) + "\n```"), title="Execution Results", border_style="green"))
        append_jsonl(transcript_path, {"type": "execution", "turn": turn, "summary": exec_summary})

        # 4) Student: phase=reflect → produce report markdown
        reflect_prompt = (
            "ROLE REMINDER (Student):\n" + STUDENT_INSTRUCTIONS +
            "\n\nYou are the Student. Reply with STRICT JSON only.\n"
            "Phase: reflect. Analyze the execution results and produce a comprehensive Markdown report explaining what you did and the outcomes.\n"
            "Schema: {\"phase\":\"reflect\",\"analysis\":\"...\",\"report_md\":\"...\"}\n"
            "When embedding outputs like directory listings or CSV/text content, TRUNCATE to a concise preview (e.g., first 20–40 lines or 1000–2000 chars) and summarize shape/counts instead of pasting full content.\n"
            "Respond with a single JSON object and nothing else.\n\nPlan JSON:\n" + json.dumps(plan_obj, ensure_ascii=False) +
            "\n\nExecution Results JSON:\n" + json.dumps(exec_summary, ensure_ascii=False)
        )
        usage_reflect = None
        try:
            _st_prev2 = student_prev_id
            student_reflect_resp = create_response(client, student_model, STUDENT_INSTRUCTIONS, reflect_prompt, effort="minimal", previous_response_id=_st_prev2)
            student_prev_id = getattr(student_reflect_resp, "id", None)
            reflect_raw = (student_reflect_resp.output_text or "").strip()
            usage_reflect = extract_usage(student_reflect_resp)
            append_jsonl(transcript_path, {"type": "llm_call", "who": "Student-reflect", "turn": turn, "previous_response_id": _st_prev2, "response_id": student_prev_id, "usage": usage_reflect})
        except Exception as ex:
            err = str(ex)
            console.rule("Student reflect unavailable due to rate limit — continuing to next turn")
            console.print(Panel(f"LLM error in Student reflect:\n{err}", title="Rate limit / LLM Error", border_style="red"))
            append_jsonl(transcript_path, {"type": "reflect_skipped", "turn": turn, "reason": "rate_limit", "error": err})
            continue
        reflect_obj = extract_first_json(reflect_raw) or {}
        report_md = reflect_obj.get('report_md') or ""
        # Fallback if reflect payload is missing/invalid or lacks report_md
        if (not isinstance(reflect_obj, dict)) or (reflect_obj.get('phase','').lower() != 'reflect') or (not report_md.strip()):
            try:
                steps = plan_obj.get('steps') or []
                qcs = exec_summary.get('quality_checks') or []
                ok_qc = sum(1 for q in qcs if q.get('ok'))
                total_qc = len(qcs)
                first_ops = ", ".join((s.get('op') or 'unknown') for s in steps[:5])
                report_md = (
                    "## Reflect (fallback)\n\n"
                    "The model returned an invalid reflect payload. Generating a concise report from the last plan and execution summary.\n\n"
                    f"- Steps run: {len(steps)} (first ops: {first_ops})\n"
                    f"- QC passed: {ok_qc}/{total_qc}\n"
                    "- Key outputs: see transcript entries for read_file previews and metrics.\n\n"
                    "### Next considerations\n"
                    "- If external data is required but unavailable, either place it under `data/...` or proceed with available synthetic/sample data to validate the pipeline.\n"
                )
                append_jsonl(transcript_path, {"type": "reflect_fallback", "turn": turn, "reason": "invalid_or_missing_reflect", "raw": reflect_raw})
            except Exception:
                report_md = reflect_raw or "(no report)"
        console.print(Panel(Markdown(report_md), title="Student · report", border_style="magenta"))
        try:
            if usage_reflect:
                console.print(Panel(f"in: {usage_reflect.get('input_tokens', 0)} · out: {usage_reflect.get('output_tokens', 0)} · total: {usage_reflect.get('total_tokens', 0)}", title="Student-reflect · tokens", border_style="magenta"))
        except Exception:
            pass
        append_jsonl(transcript_path, {"type": "student_reflect", "turn": turn, "raw": reflect_raw, "parsed": reflect_obj})
        last_student = report_md

        # Save checkpoint at end of turn
        try:
            save_checkpoint(run_paths["root"], {
                "turn": turn,
                "pi_prev_id": pi_prev_id,
                "student_prev_id": student_prev_id,
                "last_student": last_student,
                "task_path": task_path,
                "run_root": str(run_paths["root"]),
            })
        except Exception:
            pass

    console.rule("Conversation complete")
    append_jsonl(transcript_path, {"type": "final", "note": "conversation_complete"})
