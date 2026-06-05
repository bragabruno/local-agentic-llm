# Local Agentic LLM System (Mac M3 / 48 GB)

A local, on-device agentic LLM system: a LangGraph-style agent loop (tool calling, RAG,
multi-step reasoning) talking to a **local OpenAI-compatible HTTP endpoint** — no hosted
frontier API in the inference path.

Target hardware: **Apple M3, 48 GB unified memory**. MLX is the shared base layer; the
serving engine on top (default **Ollama 0.19+**) exposes `http://localhost:11434/v1`.

## Why local

This project runs **everything locally by policy** (hard-offline stance). See the Phase 0
decision record: [`docs/decisions/0000-why-local.md`](docs/decisions/0000-why-local.md).
That constraint does **not** require a model larger than 48 GB, so the build targets the
**resident 30–35B-A3B MoE** path; the large-MoE streaming path (Phase 3) stays **locked**.

## Status

Phased build — see [`docs/plan/local-agentic-llm-plan.md`](docs/plan/local-agentic-llm-plan.md)
and [`TICKETS.md`](TICKETS.md).

| Phase | What | State |
|---|---|---|
| 0 | Decision gate | ✅ Go — driver recorded, `>48 GB` = no |
| 1 | Resident MoE baseline (Ollama, Q5/Q6) | hardware-bound — see `infra/ollama/` |
| 2 | LangGraph agent layer (tools, RAG, loop) | code in `agent/` |
| 3 | Large-MoE streaming + router | 🔒 locked (Phase 0 did not justify >48 GB) |

## Layout

```
docs/        plan + decision records (ADRs)
infra/       engine config + run notes (Ollama; streaming is Phase-3-only)
agent/       LangGraph loop, tools, RAG, prompt-budget, router
eval/        smoke tests + representative end-to-end task
```

## Quickstart (resident path)

> Requires Python 3.11+. The agent code is hardware-independent; the **model** must be
> served locally first (Phase 1).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Serve a resident MoE behind an OpenAI-compatible endpoint (Phase 1, see infra/ollama/)
#    ollama serve  &  ollama run qwen3-coder:30b-a3b-q5   # [verify model tag]

# 2) Point the agent at it
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama   # placeholder; local endpoint ignores it

# 3) Run the representative end-to-end task
python -m eval.e2e_task
```

## Development

```bash
ruff check . && ruff format --check .
mypy agent eval
pytest
```

Facts tagged **[verify]** in docs/code must be confirmed against current releases before
the consuming step runs.
