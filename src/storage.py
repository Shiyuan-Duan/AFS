
from __future__ import annotations
import os, json, time, re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunPaths:
    root: Path
    code: Path
    data: Path
    outputs: Path
    reports: Path
    logs: Path

def slugify(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9\-]+', '-', s.strip().lower()).strip('-')
    return re.sub(r'-+', '-', s)[:40] or "run"

def make_run_folder(base: str | Path, name_hint: str | None = None) -> RunPaths:
    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = slugify(name_hint or "study")
    base_dir = Path(base).expanduser().resolve()
    root = base_dir / f"{ts}-{slug}"
    for d in ("code", "data", "outputs", "reports", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root, code=root/"code", data=root/"data",
                    outputs=root/"outputs", reports=root/"reports", logs=root/"logs")

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")
