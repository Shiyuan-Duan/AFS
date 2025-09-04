from __future__ import annotations
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Find first top-level {...}
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
                        frag = s[start:i+1]
                        try:
                            obj = json.loads(frag)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            return None
                        break
    return None


class Orchestrator:
    def __init__(self, run_root: Optional[Path] = None) -> None:
        self.run_root = (run_root or Path.cwd()).resolve()
        # Allow standard run subfolders
        self.allowed_prefixes = ("code/", "data/", "outputs/", "logs/")

    # --- Path helpers ---
    def _normalize_rel(self, rel: str) -> str:
        s = (rel or "").strip().lstrip("/\\")
        if not s:
            return "outputs/artifact.txt"
        if not any(s.startswith(p) for p in self.allowed_prefixes):
            s = f"outputs/{s}"
        return s

    def _safe_path(self, rel: str) -> Path:
        s = self._normalize_rel(rel)
        p = (self.run_root / s).resolve()
        if not str(p).startswith(str(self.run_root)):
            raise ValueError("path escapes run directory")
        return p

    # --- Ops ---
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return {"ok": True, "op": "write_file", "path": str(p), "bytes": len(content.encode('utf-8'))}

    def read_file(self, path: str, max_bytes: int = 20000) -> Dict[str, Any]:
        p = self._safe_path(path)
        if not p.exists():
            return {"ok": False, "op": "read_file", "path": str(p), "error": "not_found"}
        data = p.read_bytes()[:max_bytes]
        text = data.decode('utf-8', errors='replace')
        return {"ok": True, "op": "read_file", "path": str(p), "bytes": len(data), "preview": text}

    def modify_file(self, path: str, find: str, replace: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        if not p.exists():
            return {"ok": False, "op": "modify_file", "path": str(p), "error": "not_found"}
        txt = p.read_text(encoding='utf-8')
        count = txt.count(find)
        txt2 = txt.replace(find, replace)
        if txt2 != txt:
            p.write_text(txt2, encoding='utf-8')
        return {"ok": True, "op": "modify_file", "path": str(p), "replacements": count}

    def delete_file(self, path: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        if not p.exists():
            return {"ok": False, "op": "delete_file", "path": str(p), "error": "not_found"}
        p.unlink()
        return {"ok": True, "op": "delete_file", "path": str(p)}

    def run_code(self, lang: str, code: Optional[str] = None, path: Optional[str] = None,
                 args: Optional[List[str]] = None, timeout_sec: int = 60) -> Dict[str, Any]:
        def _to_text(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, bytes):
                try:
                    return v.decode('utf-8', errors='replace')
                except Exception:
                    return str(v)
            return str(v) if not isinstance(v, str) else v
        try:
            if (lang or '').lower() == 'python':
                # If 'code' looks like a filepath and no explicit 'path' is provided, treat it as path
                if (not path) and code and ('\n' not in str(code)) and str(code).strip().endswith('.py'):
                    path = str(code).strip()
                    code = None
                if path and not code:
                    pp = self._safe_path(path)
                    cmd = ['python', str(pp), *(args or [])]
                    proc = subprocess.run(cmd, cwd=str(self.run_root), capture_output=True, text=True, timeout=timeout_sec)
                    return {"ok": proc.returncode == 0, "op": "run_code", "lang": "python", "cmd": " ".join(shlex.quote(x) for x in cmd), "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
                # Inline code written to a temp script under code/
                tmp = self.run_root / "code" / "_tmp_student.py"
                tmp.write_text(code or "", encoding='utf-8')
                cmd = ['python', str(tmp), *(args or [])]
                proc = subprocess.run(cmd, cwd=str(self.run_root), capture_output=True, text=True, timeout=timeout_sec)
                return {"ok": proc.returncode == 0, "op": "run_code", "lang": "python", "cmd": " ".join(shlex.quote(x) for x in cmd), "exit_code": proc.returncode, "stdout": _to_text(proc.stdout), "stderr": _to_text(proc.stderr)}
            elif (lang or '').lower() == 'bash':
                script = code or ''
                cmd = ['/bin/bash', '-lc', script]
                proc = subprocess.run(cmd, cwd=str(self.run_root), capture_output=True, text=True, timeout=timeout_sec)
                return {"ok": proc.returncode == 0, "op": "run_code", "lang": "bash", "cmd": "bash -lc '<script>'", "exit_code": proc.returncode, "stdout": _to_text(proc.stdout), "stderr": _to_text(proc.stderr)}
            else:
                return {"ok": False, "op": "run_code", "error": f"unsupported lang: {lang}"}
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "op": "run_code", "error": "timeout", "timeout_sec": timeout_sec, "stdout": _to_text(getattr(e, 'stdout', None)), "stderr": _to_text(getattr(e, 'stderr', None))}
        except Exception as ex:
            return {"ok": False, "op": "run_code", "error": str(ex)}

    # --- Orchestration ---
    def execute_plan(self, plan_obj: Dict[str, Any]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for st in plan_obj.get('steps') or []:
            if not isinstance(st, dict):
                results.append({"ok": False, "op": "unknown", "error": "invalid_step_type", "value": st})
                continue
            op = (st.get('op') or '').lower()
            try:
                if op == 'write_file':
                    results.append(self.write_file(st.get('path',''), st.get('content','')))
                elif op == 'read_file':
                    results.append(self.read_file(st.get('path',''), int(st.get('max_bytes', 20000))))
                elif op == 'modify_file':
                    results.append(self.modify_file(st.get('path',''), st.get('find',''), st.get('replace','')))
                elif op == 'delete_file':
                    results.append(self.delete_file(st.get('path','')))
                elif op == 'run_code':
                    results.append(self.run_code(st.get('lang',''), st.get('code'), st.get('path'), st.get('args') or [], int(st.get('timeout_sec',60))))
                else:
                    results.append({"ok": False, "op": op or "", "error": "unsupported_op"})
            except Exception as ex:
                results.append({"ok": False, "op": op or "", "error": str(ex)})

        qc_results: List[Dict[str, Any]] = []
        for chk in plan_obj.get('quality_checks') or []:
            try:
                if not isinstance(chk, dict):
                    qc_results.append({"check": "invalid", "ok": False, "error": "invalid_check_type", "value": chk})
                    continue
                ctype = chk.get('check')
                if ctype == 'file_exists':
                    p = (self.run_root / chk.get('path','')).resolve()
                    qc_results.append({"check": ctype, "path": str(p), "ok": p.exists() and p.is_file()})
                elif ctype == 'json_metric_min':
                    p = (self.run_root / chk.get('from','')).resolve()
                    ok = False
                    val = None
                    if p.exists():
                        try:
                            data = json.loads(p.read_text(encoding='utf-8'))
                            val = data.get(chk.get('metric'))
                            ok = (val is not None) and (float(val) >= float(chk.get('min', 0)))
                        except Exception:
                            ok = False
                    qc_results.append({"check": ctype, "from": str(p), "metric": chk.get('metric'), "min": chk.get('min'), "value": val, "ok": ok})
                else:
                    qc_results.append({"check": ctype or "", "ok": False, "error": "unsupported_check"})
            except Exception as ex:
                qc_results.append({"check": (chk.get('check') if isinstance(chk, dict) else ""), "ok": False, "error": str(ex)})

        return {"steps": results, "quality_checks": qc_results}
