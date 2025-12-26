# Hierarchical Memory (L0/L1/L2) – Corporate Finance Task Runner

This repo runs a sequence of corporate-finance tasks (NPV/IRR, WACC, beta recap, DCF, break-even, lease vs buy),
with **3 tries per task**, a **deterministic verifier**, and a **persistent hierarchical memory**:

- **L0**: raw attempt logs (prompt → tool calls → model output → verifier feedback)
- **L1**: per-task "what mattered / what went wrong / final answer"
- **L2**: skill cards per task type (formulas, conventions, pitfalls) + a compact results index

Memory persists on disk via SQLite (`memory/memory.sqlite`).

## Quickstart

```bash
pip install -r requirements.txt
export LLM_API_KEY="..."
# Optional:
export LLM_BASE_URL="https://api.openai.com/v1"   # or an OpenAI-compatible endpoint (OpenRouter, DeepSeek, etc.)
export LLM_MODEL="gpt-5-mini"                     # pick any chat model your endpoint supports
export SUMMARIZER_MODEL="gpt-5-nano"              # cheaper model for summaries (optional)

python run_experiment.py --budget 2800 --retrieval-budget 900 --working-budget 900
```

### Using DeepSeek (cheap) via their native endpoint
Set:
```bash
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"
```

### Using OpenRouter (many providers, unified billing)
Set (either `LLM_API_KEY` or `OPENROUTER_API_KEY` works):
```bash
export LLM_PROVIDER="openrouter"                  # optional, but convenient
export OPENROUTER_API_KEY="..."                   # or export LLM_API_KEY="..."
export LLM_BASE_URL="https://openrouter.ai/api/v1" # optional if LLM_PROVIDER=openrouter
export LLM_MODEL="x-ai/grok-4.1-fast"             # example; use any OpenRouter model id
# Optional app attribution headers (shows up on OpenRouter leaderboards):
export OPENROUTER_REFERER="http://localhost"      # or your site URL
export OPENROUTER_TITLE="CF-task-runner"
```

## Compare against a baseline (no memory)
```bash
python run_experiment.py --no-memory
```

Outputs:
- `runs/latest_run.jsonl`: per-attempt logs + verifier info
- `runs/summary.json`: aggregate metrics
- `memory/memory.sqlite`: persistent memory database

