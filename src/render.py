from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
            except Exception:
                # Skip malformed lines
                continue
    return events


def _md_code_block(lang: str, content: str) -> str:
    return f"```{lang}\n{content}\n```\n"


def build_markdown(events: List[Dict[str, Any]], title: str) -> str:
    md: List[str] = []
    md.append(f"# {title}")
    md.append("")
    # Legend
    md.append(
        "> Note: Colored boxes are applied in the PDF/HTML rendering. This Markdown includes role sections for PI and Student, as well as execution artifacts."
    )
    md.append("")
    for ev in events:
        etype = ev.get("type")
        if etype == "run_root":
            md.append(f"<div class='meta'>Run folder: <code>{ev.get('path','')}</code></div>")
            md.append("")
        elif etype in ("data_copied", "data_copy_note"):
            md.append(f"<div class='meta'>Data: {json.dumps(ev, ensure_ascii=False)}</div>")
            md.append("")
        elif etype == "message":
            role = ev.get("role") or ev.get("speaker") or "pi"
            content = ev.get("content", "").strip()
            turn = ev.get("turn") or ev.get("step") or ""
            if role.lower().startswith("pi") or role.lower().startswith("scientist"):
                md.append(f"<div class='pi'><div class='role'>PI · turn {turn}</div>\n\n{content}\n\n</div>")
            else:
                md.append(f"<div class='student'><div class='role'>Student · turn {turn}</div>\n\n{content}\n\n</div>")
            md.append("")
        elif etype == "student_plan":
            md.append("<div class='plan'><div class='role'>Student · plan (JSON)</div>")
            raw = ev.get("raw") or json.dumps(ev.get("parsed", {}), ensure_ascii=False, indent=2)
            # Prefer parsed pretty view
            parsed = ev.get("parsed")
            if isinstance(parsed, dict) and parsed:
                md.append(_md_code_block("json", json.dumps(parsed, ensure_ascii=False, indent=2)))
            else:
                md.append(_md_code_block("json", str(raw)))
            md.append("</div>")
            md.append("")
        elif etype == "execution":
            md.append("<div class='exec'><div class='role'>Execution Results</div>")
            summary = ev.get("summary") or {k: ev.get(k) for k in ("turn", "log", "details") if k in ev}
            md.append(_md_code_block("json", json.dumps(summary, ensure_ascii=False, indent=2)))
            md.append("</div>")
            md.append("")
        elif etype == "student_reflect":
            parsed = ev.get("parsed") or {}
            report_md = parsed.get("report_md") or ev.get("raw") or "(no report)"
            md.append("<div class='report'><div class='role'>Student · report</div>")
            md.append(report_md)
            md.append("</div>")
            md.append("")
        elif etype == "done":
            md.append("<div class='done'><strong>DONE signaled — conversation ended.</strong></div>")
            md.append("")
        elif etype == "final":
            md.append("<div class='final'><em>Final event recorded.</em></div>")
            md.append("")
        else:
            # generic fallback
            md.append(_md_code_block("json", json.dumps(ev, ensure_ascii=False, indent=2)))
            md.append("")

    return "\n".join(md).strip() + "\n"


def md_to_html(md_text: str, doc_title: str) -> str:
    # Convert Markdown to HTML with optional dependency
    try:
        import markdown  # type: ignore
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:
        # Minimal fallback: wrap as preformatted text
        body = f"<pre>{md_text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')}</pre>"

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; font-size: 0.9em; }
    .role { font-weight: 600; margin-bottom: 6px; }
    .pi { border-left: 6px solid #1f78b4; background: #f0f7ff; padding: 10px 12px; margin: 12px 0; }
    .student { border-left: 6px solid #6a3d9a; background: #f7f0ff; padding: 10px 12px; margin: 12px 0; }
    .plan { border-left: 6px solid #33a02c; background: #f0fff3; padding: 10px 12px; margin: 12px 0; }
    .exec { border-left: 6px solid #ff7f00; background: #fff7ed; padding: 10px 12px; margin: 12px 0; }
    .report { border-left: 6px solid #b15928; background: #fff3e6; padding: 10px 12px; margin: 12px 0; }
    .meta { color: #666; font-size: 0.9em; }
    .done { color: #fff; background: #2c7; padding: 6px 10px; display: inline-block; border-radius: 6px; }
    .final { color: #666; font-style: italic; }
    h1 { border-bottom: 2px solid #eee; padding-bottom: 4px; }
    """
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{doc_title}</title>
    <style>{css}</style>
  </head>
  <body>
    {body}
  </body>
</html>
"""
    return html


def html_to_pdf(html_text: str, out_pdf: Path) -> bool:
    # Try WeasyPrint first
    try:
        from weasyprint import HTML  # type: ignore
        HTML(string=html_text, base_url=str(out_pdf.parent)).write_pdf(str(out_pdf))
        return True
    except Exception:
        pass
    # Try pdfkit if wkhtmltopdf is available
    try:
        import pdfkit  # type: ignore
        pdfkit.from_string(html_text, str(out_pdf))
        return True
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser(description="Render a conversation transcript JSONL to Markdown and a colored PDF")
    ap.add_argument("--input", required=True, help="Path to transcript.jsonl")
    ap.add_argument("--output", default=None, help="Output PDF path (default: same folder/report.pdf)")
    ap.add_argument("--title", default="Conversation Report", help="Document title")
    ap.add_argument("--also-md", action="store_true", help="Also write a .md file next to the PDF")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    events = read_jsonl(in_path)
    md = build_markdown(events, args.title)

    if args.output:
        out_pdf = Path(args.output).expanduser().resolve()
    else:
        # Default to project-level example_outputs/<run_name>/report.pdf when input is under runs/<run_name>/logs/
        run_root = in_path.parent.parent if in_path.parent.name == "logs" else in_path.parent
        run_name = run_root.name
        # Try to locate project root (parent of "runs")
        runs_dir = run_root.parent
        proj_root = runs_dir.parent if runs_dir.name == "runs" else run_root.parent
        out_dir = proj_root / "example_outputs" / run_name
        out_pdf = out_dir / "report.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    html = md_to_html(md, args.title)
    ok = html_to_pdf(html, out_pdf)
    if not ok:
        # Fallback: write HTML file if PDF conversion unavailable
        out_html = out_pdf.with_suffix(".html")
        out_html.write_text(html, encoding="utf-8")
        print(f"[warn] Could not write PDF. Wrote HTML instead: {out_html}")
    else:
        print(f"[ok] Wrote PDF: {out_pdf}")

    if args.also_md:
        out_md = out_pdf.with_suffix(".md")
        out_md.write_text(md, encoding="utf-8")
        print(f"[ok] Wrote Markdown: {out_md}")


if __name__ == "__main__":
    main()
