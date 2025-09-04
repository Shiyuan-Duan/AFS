import argparse
import os
from typing import Optional
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
try:
    from .run_utils import make_run_folder, copy_data_to_run, append_jsonl, load_checkpoint
except Exception:
    from run_utils import make_run_folder, copy_data_to_run, append_jsonl, load_checkpoint
try:
    from .orchestrator import Orchestrator
    from .agent_loop import run_conversation_loop
except Exception:
    from orchestrator import Orchestrator
    from agent_loop import run_conversation_loop


load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="PI↔Student loop over a YAML task (3 iterations)")
    parser.add_argument(
        "--task",
        default="configs/training2017_task_en.yaml",
        help="Path to YAML task file (default: configs/training2017_task_en.yaml)"
    )
    parser.add_argument("--rounds", type=int, default=200, help="Maximum PI→Student iterations (default: 200)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a run folder or checkpoint.json path")
    args = parser.parse_args()

    client = OpenAI()
    console = Console()

    # Set up run folder (new or resume)
    run_paths = None
    start_turn = 1
    pi_prev_id: Optional[str] = None
    student_prev_id: Optional[str] = None
    last_student: Optional[str] = None
    task_path = args.task
    if args.resume:
        # Load checkpoint
        try:
            ck = load_checkpoint(Path(args.resume))
        except Exception as e:
            print(f"[error] failed to load checkpoint: {e}")
            return
        run_root = Path(ck.get("run_root") or Path(args.resume)).resolve()
        run_paths = {
            "root": run_root,
            "code": run_root / "code",
            "data": run_root / "data",
            "outputs": run_root / "outputs",
            "logs": run_root / "logs",
        }
        start_turn = int(ck.get("turn", 1)) + 1
        pi_prev_id = ck.get("pi_prev_id")
        student_prev_id = ck.get("student_prev_id")
        last_student = ck.get("last_student")
        task_path = ck.get("task_path") or args.task
        transcript_path = run_paths["logs"] / "transcript.jsonl"
        append_jsonl(transcript_path, {"type": "resume", "from": str(Path(args.resume)), "run_root": str(run_root), "start_turn": start_turn})
        orch = Orchestrator(run_root)
        # Ensure task text is available on resume
        try:
            with open(task_path, "r", encoding="utf-8") as f:
                task_text = f.read()
        except Exception as e:
            print(f"[error] failed to read task file on resume: {e}")
            return
    else:
        # New run
        try:
            with open(task_path, "r", encoding="utf-8") as f:
                task_text = f.read()
        except Exception as e:
            print(f"[error] failed to read task file: {e}")
            return
        run_paths = make_run_folder("runs", name_hint="chat")
        try:
            _ = copy_data_to_run("data", run_paths["data"])  # copy dataset into run/data
        except Exception:
            pass
        transcript_path = run_paths["logs"] / "transcript.jsonl"
        append_jsonl(transcript_path, {"type": "run_root", "path": str(run_paths['root'])})
        orch = Orchestrator(run_paths["root"])  # execute relative to run folder

    # Prepare loop config
    rounds = max(1, args.rounds)
    pi_model = os.getenv("PI_MODEL", "gpt-4.1-mini")
    student_model = os.getenv("STUDENT_MODEL", "gpt-4.1-mini")
    # Initialize state if not resuming
    if not args.resume:
        last_student = None
        pi_prev_id = None
        student_prev_id = None

    # Hand off to conversation loop
    run_conversation_loop(
        client=client,
        console=console,
        orch=orch,
        run_paths=run_paths,
        transcript_path=transcript_path,
        task_text=task_text,
        task_path=task_path,
        start_turn=start_turn,
        rounds=rounds,
        pi_model=pi_model,
        student_model=student_model,
        pi_prev_id=pi_prev_id,
        student_prev_id=student_prev_id,
        last_student=last_student,
    )


if __name__ == "__main__":
    main()
