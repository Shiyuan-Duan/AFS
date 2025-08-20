from __future__ import annotations
import os
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from .agent_loop import stream_dialogue

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def main(
    task: str = typer.Option(..., help="Path to task YAML"),
    max_steps: int = typer.Option(12, help="Maximum dialogue steps"),
    run_name: str | None = typer.Option(None, help="Suffix for run folder name"),
    dry_run: bool = typer.Option(False, help="Do not execute tools; just simulate LLM calls"),
    execute_actions: bool = typer.Option(True, help="Allow executing Assistant action plans (writes to runs/)"),
    allow_installs: bool = typer.Option(False, help="Allow Assistant to install Python packages via pip inside the run (sets ALLOW_PIP_INSTALLS=1)"),
    data_root: str = typer.Option("/Users/shiyuanduan/Documents/ai-for-science-agents/data", help="Path to your local data folder to copy into each run (when executing)"),
    until_done: bool = typer.Option(False, help="Keep going until Scientist replies DONE (safety cap applies)"),
):
    console.rule("Scientist ↔ Assistant")
    if allow_installs:
        os.environ["ALLOW_PIP_INSTALLS"] = "1"
    payload_json: str | None = None

    for evt in stream_dialogue(task_yaml=task, run_name=run_name, max_steps=max_steps, dry_run=dry_run, execute_actions=execute_actions, until_done=until_done, data_root=data_root):
        if evt.get("type") == "run_root":
            console.print(Panel(f"Artifacts will be written under: {evt['path']}", title="Run Folder", border_style="blue"))
            continue
        if evt.get("type") == "data_copied":
            details = evt.get("details", {})
            console.print(Panel(Markdown("Copied data into run folder.\n\n" + "```json\n" + json.dumps(details, indent=2, ensure_ascii=False) + "\n```"), title="Data Ingest", border_style="blue"))
            continue
        if evt.get("type") == "data_copy_error":
            console.print(Panel(Markdown(f"[red]Failed to copy data[/red]: {evt.get('error','unknown')}"), title="Data Ingest Error", border_style="red"))
            continue
        if evt.get("type") == "message":
            role = evt["role"]
            step = evt["step"]
            text = evt.get("content", "")
            title = f"{role.capitalize()} · step {step}"
            style = "cyan" if role == "scientist" else "magenta"
            console.print(Panel(Markdown(text), title=title, border_style=style))
        elif evt.get("type") == "warning":
            console.print(Panel(evt.get("message", ""), title="Warning", border_style="red"))
        elif evt.get("type") == "plan":
            step = evt.get("step")
            plan = evt.get("plan", [])
            finalize = evt.get("finalize", {})
            console.print(Panel(Markdown("Assistant proposed an action plan.\n\n" + "```json\n" + json.dumps({"plan": plan, "finalize": finalize}, indent=2, ensure_ascii=False) + "\n```"), title=f"Action Plan · step {step}", border_style="yellow"))
            if not execute_actions:
                console.print("[yellow]Execution disabled (use --execute-actions to run). The conversation will proceed without execution.[/yellow]")
        elif evt.get("type") == "parse_warning":
            console.print(Panel(Markdown("Assistant output looked like JSON but could not be parsed. Details:\n\n" + "```\n" + "\n".join([str(x) for x in evt.get("details", [])]) + "\n```\nRaw preview:\n\n" + "```\n" + (evt.get("raw_preview") or "") + "\n```"), title="Parse Warning", border_style="yellow"))
        elif evt.get("type") == "action_warning":
            console.print(Panel(Markdown("Assistant output looks like an action plan but is not valid JSON. Details:\n\n" +
                                     "```\n" + "\n".join([str(x) for x in evt.get("details", [])]) + "\n```\nRaw preview:\n\n" +
                                     "```\n" + (evt.get("raw_preview") or "") + "\n```\n" +
                                     "Scientist: please decide — either instruct the Assistant to resend STRICT JSON action_plan (no fences), "
                                     "or proceed conversationally without execution."),
                                     title="Action Plan Warning", border_style="yellow"))
        elif evt.get("type") == "execution":
            run_root = evt.get("run_root")
            log = evt.get("log", [])
            console.print(Panel(Markdown(f"Executed plan in `{run_root}`. Summary of steps: \n\n" + "```json\n" + json.dumps(log, indent=2, ensure_ascii=False) + "\n```"), title="Execution Results", border_style="green"))
        elif evt.get("type") == "final":
            payload_json = json.dumps(evt["payload"], ensure_ascii=False)

    console.rule("[bold green]Run complete[/bold green]")
    if payload_json:
        console.print_json(payload_json)

if __name__ == "__main__":
    app()
