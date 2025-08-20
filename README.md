# AI-for-Science (Two-Agent, Reasoning Models)

This repository contains a **complete**, local-first multi-agent scaffold that pairs a *Scientist* and an *Assistant* — both powered by OpenAI reasoning models (e.g., **GPT‑5 Thinking** and **gpt‑5‑nano‑reason**). The Scientist explores hypotheses, writes and executes code, saves artifacts, and **produces a final report**. The Assistant supplies code snippets, references, and materials. All artifacts from each run (code, JSON, preprocessed data, reports, and transcript) are stored in a timestamped folder under `runs/`.

> Created on 2025-08-20 05:36:09.

---

## 1) Setup

```bash
# 1) Create the environment
conda env create -f environment.yml
conda activate ai-for-science-agents

# 2) Configure secrets
cp .env.example .env
# edit .env to add your OPENAI_API_KEY and models
```

If you're using a private endpoint, set `OPENAI_BASE_URL` in `.env` accordingly.

---

## 2) Quick start

```bash
python -m src.main --task configs/example_task.yaml --max-steps 12
```

- A new run folder is created at: `runs/YYYYmmdd-HHMMSS-<slug>/`.
- Code is executed locally by the **Scientist** via a controlled Python runner tool.
- The Scientist continues until it **renders a report** (`reports/report.md` and `reports/report.html`).

> ⚠️ **Security note**: The `run_python` tool executes Python **on your machine**. Only use datasets and prompts you trust. For hardened isolation, run inside a disposable container or VM.

---

## 3) How it works (high level)

- **Two agents**:
  - **Scientist** (reasoning model): Plans, hypothesizes, requests/uses tools, writes code, saves files, generates the run report.
  - **Assistant** (reasoning/nano model): Provides code and references/materials in response to the Scientist.
- **Tools** (Function‑Calling / tool use):
  - `write_text_file`, `write_json_file`, `read_text_file`, `list_dir`, `make_dir`
  - `run_python` — executes code in the current run's sandbox and returns stdout/stderr/exit_code.
  - `render_report` — compiles a Markdown report via Jinja2 and emits HTML.
- **Storage**:
  - Every message and tool result is streamed to `transcript.jsonl`.
  - All artifacts are stored under `runs/<run-id>/`.

---

## 4) Models

Set in `.env`:
- `SCIENTIST_MODEL` — e.g. `gpt-5-thinking` (reasoning).
- `ASSISTANT_MODEL` — e.g. `gpt-5-nano-reason`.

You can swap to other models. For reasoning models, this project uses the **Responses API** with `reasoning.effort` when available, and automatically falls back to Chat Completions if necessary.

---

## 5) Example task

See `configs/example_task.yaml` for a self‑contained “AI for science” task spec. You can create multiple YAMLs (one per study).

---

## 6) CLI

```bash
python -m src.main --task <path/to/task.yaml> --max-steps 16 --run-name "my-trial"
```

Common flags:
- `--max-steps` limits dialogue turns.
- `--run-name` adds a suffix to the run folder name.
- `--dry-run` prints the plan without executing tools.

---

## 7) Files produced in a run

```
runs/<stamp>-<slug>/
  ├─ code/           # Executable scripts written by the Scientist
  ├─ data/           # Preprocessed data saved during the run
  ├─ outputs/        # Result figures/CSVs
  ├─ reports/        # report.md, report.html
  ├─ logs/           # tool logs
  ├─ transcript.jsonl
  └─ run.json        # run metadata
```

---

## 8) Extending

- Add new tools in `src/tools.py` and register them in `src/agent_loop.py`.
- Wrap external tools/APIs behind safe function‑calling definitions.
- Swap models in `.env`. If you use non‑reasoning models, the framework still works.

---

## 9) References (API docs)

- Reasoning models & best practices — OpenAI Docs
- Responses API & function calling — OpenAI Docs

See code comments for URLs.
# AFS
