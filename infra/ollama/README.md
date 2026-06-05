# Phase 1 — Resident MoE behind Ollama (run notes)

Hardware-bound steps (LLM-1.1 → LLM-1.GATE). These need the M3 and real time (multi-GB
downloads); they are **not** run in CI. Record measured numbers in
[`../../docs/decisions/0001-resident-engine.md`](../../docs/decisions/0001-resident-engine.md).

All model tags below are **[verify]** — confirm against the registry before pulling.

## 1.1 — Install / confirm Ollama 0.19+ (MLX backend)

```bash
brew install ollama        # or upgrade: brew upgrade ollama
ollama --version           # must be >= 0.19  [verify MLX serving]
ollama serve &             # exposes http://localhost:11434/v1
```

## 1.2 — Raise the Metal working-set ceiling

The default Metal working-set cap throttles large resident weights. Raise it to admit the
chosen quant + target context. **Size on total params × quant bits + KV headroom**, never
active params. See `set-wired-limit.sh`.

```bash
# Example only — compute the value for your quant; do not copy blindly.
sudo sysctl iogpu.wired_limit_mb=40960
```

## 1.3 — Pull the resident MoE at Q5 and Q6

```bash
ollama pull qwen3-coder:30b-a3b-q5   # [verify tag]
ollama pull qwen3-coder:30b-a3b-q6   # [verify tag]
ollama list                          # record on-disk sizes
```

Record resident size with `ollama ps` while the model is loaded.

## 1.4 — Measure throughput & memory

Use `--verbose` for tok/s and watch memory (`sudo powermetrics` / Activity Monitor).
Record: short-prompt tok/s, long-prompt (8–16k) tok/s + stability, peak resident memory,
KV-cache headroom. **Do not invent numbers — record measured values.**

```bash
ollama run qwen3-coder:30b-a3b-q5 --verbose "<short prompt>"
ollama run qwen3-coder:30b-a3b-q5 --verbose "<8-16k token prompt>"
```

## 1.5 — Quant sensitivity (tool-calling)

```bash
AGENT_MODEL=qwen3-coder:30b-a3b-q5 python -m eval.smoke_toolcalling
AGENT_MODEL=qwen3-coder:30b-a3b-q6 python -m eval.smoke_toolcalling
```

Pick the quant that holds tool-call formatting.

## 1.GATE — Go/No-Go

GO if: memory budget holds (~32–38 GB usable, no swap thrash), tok/s usable
interactively, tool-calling survives quant, endpoint streams `/v1/chat/completions`.
Otherwise drop to Q4 / a smaller-total MoE and re-measure — **do not** jump to Phase 3.
