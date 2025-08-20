from __future__ import annotations
import os, sys, re, json, subprocess, textwrap, shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import importlib
from jinja2 import Environment, FileSystemLoader, select_autoescape
import shutil
from markdown import markdown

@dataclass
class ToolContext:
    run_root: Path
    code_dir: Path
    data_dir: Path
    outputs_dir: Path
    reports_dir: Path
    logs_dir: Path

ALLOWED_PREFIXES = ("code/", "data/", "outputs/", "reports/")

def _safe_join(root: Path, relpath: str) -> Path:
    relpath = relpath.strip().lstrip("/\\")
    if not any(relpath.startswith(p) for p in ALLOWED_PREFIXES):
        raise ValueError("relpath must start with code/, data/, outputs/, or reports/")
    p = (root / relpath).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise ValueError("Path escapes run directory")
    return p

def write_text_file(ctx: ToolContext, relpath: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
    p = _safe_join(ctx.run_root, relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        return {"ok": False, "path": str(p), "msg": "exists and overwrite=False"}
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8"))}

def write_json_file(ctx: ToolContext, relpath: str, obj: Any, overwrite: bool = True) -> Dict[str, Any]:
    p = _safe_join(ctx.run_root, relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        return {"ok": False, "path": str(p), "msg": "exists and overwrite=False"}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return {"ok": True, "path": str(p)}

def read_text_file(ctx: ToolContext, relpath: str) -> Dict[str, Any]:
    p = _safe_join(ctx.run_root, relpath)
    if not p.exists():
        return {"ok": False, "msg": "not found", "path": str(p)}
    with open(p, "r", encoding="utf-8") as f:
        return {"ok": True, "path": str(p), "content": f.read()}

def list_dir(ctx: ToolContext, relpath: str = ".") -> Dict[str, Any]:
    relpath = relpath.strip().lstrip("/\\")
    p = (ctx.run_root / relpath).resolve()
    if not str(p).startswith(str(ctx.run_root.resolve())):
        return {"ok": False, "msg": "path escapes run directory", "path": str(p)}
    if not p.exists():
        return {"ok": False, "msg": "not found", "path": str(p)}
    items = []
    for child in sorted(p.iterdir()):
        items.append({"name": child.name, "is_dir": child.is_dir(), "size": child.stat().st_size})
    return {"ok": True, "path": str(p), "items": items}

def make_dir(ctx: ToolContext, relpath: str) -> Dict[str, Any]:
    p = _safe_join(ctx.run_root, relpath)
    p.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": str(p)}

def ingest_data(ctx: ToolContext, src_path: str, dest_relpath: str = "data/dataset") -> Dict[str, Any]:
    """
    Copy a local file or directory from src_path into the current run under dest_relpath.
    Useful when the dataset lives outside the run folder. Read-only from src.
    """
    try:
        src = Path(src_path).expanduser().resolve()
        if not src.exists():
            return {"ok": False, "error": f"src not found: {src}"}
        dest = _safe_join(ctx.run_root, dest_relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        files_copied = 0
        if src.is_file():
            shutil.copy2(str(src), str(dest))
            files_copied = 1
        else:
            # Copy directory tree
            if dest.exists() and any(dest.iterdir()):
                # merge copy
                for root, dirs, files in os.walk(src):
                    rel = Path(root).relative_to(src)
                    target_dir = dest / rel
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for fn in files:
                        shutil.copy2(str(Path(root)/fn), str(target_dir/fn))
                        files_copied += 1
            else:
                shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
                # best-effort count
                files_copied = sum(1 for _ in dest.rglob('*') if _.is_file())
        return {"ok": True, "src": str(src), "dest": str(dest), "files": files_copied}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def modify_text_file(ctx: ToolContext, relpath: str, edits: list[dict]) -> Dict[str, Any]:
    """
    Apply simple text edits to a file under the run folder.
    Each edit is an object like {find: "...", replace: "...", regex: bool?}.
    Returns counts per edit and writes back to the same path.
    """
    p = _safe_join(ctx.run_root, relpath)
    if not p.exists():
        return {"ok": False, "path": str(p), "error": "not found"}
    try:
        original = p.read_text(encoding="utf-8")
    except Exception as e:
        return {"ok": False, "path": str(p), "error": f"read_error: {e}"}

    import re as _re
    text = original
    results: list[dict] = []
    total_changes = 0
    for e in edits or []:
        find = str(e.get("find", ""))
        repl = str(e.get("replace", ""))
        use_regex = bool(e.get("regex", False))
        count = 0
        if not find:
            results.append({"find": find, "replace": repl, "changed": 0, "error": "empty find"})
            continue
        try:
            if use_regex:
                text, count = _re.subn(find, repl, text)
            else:
                count = text.count(find)
                if count:
                    text = text.replace(find, repl)
            results.append({"find": find, "replace": repl, "changed": int(count), "regex": use_regex})
            total_changes += int(count)
        except Exception as ex:
            results.append({"find": find, "replace": repl, "changed": 0, "error": str(ex)})

    try:
        if total_changes > 0:
            p.write_text(text, encoding="utf-8")
        return {"ok": True, "path": str(p), "edits": results, "total_changes": total_changes}
    except Exception as e:
        return {"ok": False, "path": str(p), "error": f"write_error: {e}", "edits": results}

# --- Safe execution helper ---
def _disallowed(code: str) -> str | None:
    bad = ["pip install", "subprocess.", "os.system(", "import pip", "check_call(", "Popen("]
    for b in bad:
        if b in code:
            return b
    return None

def run_python(ctx: ToolContext, code: str | None = None, filename_hint: str = "snippet.py",
               args: list[str] | None = None, timeout_sec: int = 60, path: str | None = None,
               env: Dict[str, str] | None = None) -> Dict[str, Any]:
    """
    Two modes:
    - path=code/xxx.py : run an existing file (preferred, safe)
    - code=...         : save to code/filename_hint then run (inline execution). We block dangerous patterns.
    """
    if path:
        p = _safe_join(ctx.run_root, path)
        cmd = ["python", str(p)]
        if args:
            cmd.extend(args)
        try:
            run_env = dict(os.environ)
            if env:
                run_env.update({str(k): str(v) for k, v in env.items()})
            proc = subprocess.run(cmd, cwd=str(ctx.run_root), capture_output=True, text=True, timeout=timeout_sec, env=run_env)
            return {"ok": True, "path": str(p), "exit_code": proc.returncode,
                    "stdout": proc.stdout, "stderr": proc.stderr, "cmd": " ".join(shlex.quote(x) for x in cmd)}
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "path": str(p), "timeout": timeout_sec, "stdout": e.stdout, "stderr": e.stderr}

    if code is None:
        return {"ok": False, "error": "Provide either 'path' to an existing file, or inline 'code'."}
    # basic safety for inline execution
    bad = _disallowed(code)
    if bad:
        return {"ok": False, "error": f"Disallowed pattern in inline code: {bad}"}

    safe_name = filename_hint.replace("/", "_").replace("\\", "_")
    path_file = ctx.code_dir / safe_name
    with open(path_file, "w", encoding="utf-8") as f:
        f.write(code)

    cmd = ["python", str(path_file)]
    if args:
        cmd.extend(args)
    try:
        run_env = dict(os.environ)
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        proc = subprocess.run(cmd, cwd=str(ctx.run_root), capture_output=True, text=True, timeout=timeout_sec, env=run_env)
        return {"ok": True, "path": str(path_file), "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr, "cmd": " ".join(shlex.quote(x) for x in cmd)}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "path": str(path_file), "timeout": timeout_sec, "stdout": e.stdout, "stderr": e.stderr}

def render_report(ctx: ToolContext, title: str, abstract: str, sections: list[dict],
                  figures: list[dict] | None = None, conclusions: list[str] | None = None) -> Dict[str, Any]:
    env = Environment(
        loader=FileSystemLoader(str((Path(__file__).parent / "templates").resolve())),
        autoescape=select_autoescape()
    )
    tmpl = env.get_template("report.md.jinja")
    md = tmpl.render(title=title, abstract=abstract, sections=sections or [], figures=figures or [], conclusions=conclusions or [])
    md_path = ctx.reports_dir / "report.md"
    html_path = ctx.reports_dir / "report.html"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    html = "<html><head><meta charset='utf-8'><title>{}</title></head><body>{}</body></html>".format(title, markdown(md, extensions=["tables"]))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return {"ok": True, "markdown": str(md_path), "html": str(html_path)}

def install_python_packages(ctx: ToolContext, packages: list[str], upgrade: bool = False, index_url: str | None = None, timeout_sec: int = 600) -> Dict[str, Any]:
    """
    Best-effort, guarded installer. Requires ALLOW_PIP_INSTALLS=1 in environment.
    - Validates package strings against a conservative regex (name[==version]).
    - Uses current interpreter: python -m pip install [...].
    - Returns stdout/stderr and exit_code.
    """
    if os.environ.get("ALLOW_PIP_INSTALLS") != "1":
        return {"ok": False, "error": "pip installs are disabled. Set ALLOW_PIP_INSTALLS=1 or run CLI with --allow-installs."}
    if not packages:
        return {"ok": False, "error": "no packages specified"}
    pat = re.compile(r"^[A-Za-z0-9_.\-]+(==[A-Za-z0-9_.\-]+)?$")
    safe_pkgs: list[str] = []
    for p in packages:
        ps = str(p).strip()
        if not pat.match(ps):
            return {"ok": False, "error": f"invalid package spec: {ps}"}
        safe_pkgs.append(ps)
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if index_url:
        cmd.extend(["--index-url", str(index_url)])
    cmd.extend(safe_pkgs)
    try:
        proc = subprocess.run(cmd, cwd=str(ctx.run_root), capture_output=True, text=True, timeout=timeout_sec)
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "cmd": " ".join(shlex.quote(x) for x in cmd),
        }
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "timeout": timeout_sec, "stdout": e.stdout, "stderr": e.stderr}

def check_env(packages: list[str] | None = None) -> Dict[str, Any]:
    to_check = packages or ["numpy", "scipy", "pandas", "scikit-learn", "sklearn", "matplotlib", "seaborn"]
    details = []
    overall_ok = True
    for name in to_check:
        info: Dict[str, Any] = {"name": name}
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", None)
            info.update({"available": True, "version": str(ver) if ver is not None else None})
        except Exception as e:
            info.update({"available": False, "error": str(e)})
            overall_ok = False
        details.append(info)
    return {"ok": overall_ok, "details": details}

# ---- Function specs for OpenAI tool schema (kept for compatibility) ----
FUNCTION_SPECS = [
    {
        "name": "write_text_file",
        "description": "Write a UTF-8 text file within the current run folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "relpath": {"type": "string", "description": "Relative path under code/, data/, outputs/, or reports/."},
                "content": {"type": "string", "description": "Text content to write."},
                "overwrite": {"type": "boolean", "description": "Overwrite existing file if true", "default": True},
            },
            "required": ["relpath", "content"]
        }
    },
    {
        "name": "modify_text_file",
        "description": "Modify an existing text file in-place using simple find/replace edits (optionally regex).",
        "parameters": {
            "type": "object",
            "properties": {
                "relpath": {"type": "string", "description": "Target path under code/, data/, outputs/, or reports/."},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                            "regex": {"type": "boolean", "default": false}
                        },
                        "required": ["find", "replace"]
                    }
                }
            },
            "required": ["relpath", "edits"]
        }
    },
    {
        "name": "ingest_data",
        "description": "Copy a local dataset (file or directory) into the current run folder under data/.",
        "parameters": {
            "type": "object",
            "properties": {
                "src_path": {"type": "string", "description": "Absolute or relative source path on local disk."},
                "dest_relpath": {"type": "string", "description": "Destination under data/", "default": "data/dataset"}
            },
            "required": ["src_path"]
        }
    },
    {
        "name": "write_json_file",
        "description": "Write a JSON file under the run folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "relpath": {"type": "string"},
                "obj": {"type": "object"},
                "overwrite": {"type": "boolean", "default": True},
            },
            "required": ["relpath", "obj"]
        }
    },
    {
        "name": "read_text_file",
        "description": "Read a text file under the run folder.",
        "parameters": {
            "type": "object",
            "properties": {"relpath": {"type": "string"}},
            "required": ["relpath"]
        }
    },
    {
        "name": "list_dir",
        "description": "List directory contents under the run folder.",
        "parameters": {
            "type": "object",
            "properties": {"relpath": {"type": "string", "default": "."}}
        }
    },
    {
        "name": "make_dir",
        "description": "Create a directory under the run folder.",
        "parameters": {
            "type": "object",
            "properties": {"relpath": {"type": "string"}},
            "required": ["relpath"]
        }
    },
    {
        "name": "run_python",
        "description": "Execute Python. Prefer running an existing file via 'path'; inline 'code' is restricted.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to an existing file under code/ to execute."},
                "code": {"type": "string", "description": "Inline code (discouraged; disallows subprocess/pip)."},
                "filename_hint": {"type": "string", "description": "Filename to save inline code under code/", "default": "snippet.py"},
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout_sec": {"type": "integer", "default": 60},
                "env": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Extra environment variables (e.g., MPLBACKEND=Agg)."}
            },
            "required": []
        }
    },
    {
        "name": "check_env",
        "description": "Report availability and versions of common packages (numpy, scipy, pandas, sklearn, matplotlib, seaborn).",
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    {
        "name": "render_report",
        "description": "Render the final Markdown and HTML report using the saved artifacts.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "abstract": {"type": "string"},
                "sections": {"type": "array", "items": {"type": "object"}},
                "figures": {"type": "array", "items": {"type": "object"}},
                "conclusions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title", "abstract", "sections"]
        }
    },
    {
        "name": "install_python_packages",
        "description": "Install Python packages via pip in the current interpreter (guarded; requires ALLOW_PIP_INSTALLS=1).",
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {"type": "array", "items": {"type": "string"}},
                "upgrade": {"type": "boolean", "default": False},
                "index_url": {"type": "string"},
                "timeout_sec": {"type": "integer", "default": 600}
            },
            "required": ["packages"]
        }
    }
]

DISPATCH = {
    "write_text_file": write_text_file,
    "write_json_file": write_json_file,
    "read_text_file": read_text_file,
    "list_dir": list_dir,
    "make_dir": make_dir,
    "run_python": run_python,
    "render_report": render_report,
    "ingest_data": ingest_data,
    "check_env": lambda **kwargs: check_env(kwargs.get("packages")),
    "install_python_packages": install_python_packages,
}
