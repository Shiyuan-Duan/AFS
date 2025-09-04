from __future__ import annotations
import shutil
from pathlib import Path
from datetime import datetime
import os
import json
from typing import Dict


def make_run_folder(base: str = "runs", name_hint: str = "chat") -> Dict[str, Path]:
    base_path = Path(base).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = base_path / f"{ts}-{name_hint}"
    code = run_root / "code"
    data = run_root / "data"
    outputs = run_root / "outputs"
    logs = run_root / "logs"
    for p in (run_root, code, data, outputs, logs):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": run_root, "code": code, "data": data, "outputs": outputs, "logs": logs}


def copy_data_to_run(src: str | Path, dest_data: str | Path) -> int:
    srcp = Path(src).resolve()
    destp = Path(dest_data).resolve()
    if not srcp.exists() or not srcp.is_dir():
        return 0
    files_copied = 0
    for root, dirs, files in os.walk(srcp):  # type: ignore[name-defined]
        # Delayed import to avoid global dependency in module namespace
        pass
    import os as _os
    for root, dirs, files in _os.walk(srcp):
        rel = Path(root).relative_to(srcp)
        target_dir = destp / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for fn in files:
            shutil.copy2(str(Path(root) / fn), str(target_dir / fn))
            files_copied += 1
    return files_copied


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_checkpoint(run_root: Path, state: dict) -> Path:
    """Write a checkpoint JSON under logs/checkpoint.json and return its path."""
    ck = run_root / "logs" / "checkpoint.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    with open(ck, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return ck


def load_checkpoint(path_or_run: Path) -> dict:
    """Load a checkpoint given either a checkpoint file path or a run root directory."""
    p = Path(path_or_run).resolve()
    if p.is_dir():
        p = p / "logs" / "checkpoint.json"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
