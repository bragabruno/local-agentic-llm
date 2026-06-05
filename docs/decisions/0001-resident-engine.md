# ADR 0001 — Resident engine & model (Phase 1)

- **Status:** Template — fill in after LLM-1.4 / LLM-1.5 measurements
- **Date:** _TBD_
- **Tickets:** LLM-1.1 … LLM-1.6, LLM-1.GATE

> This ADR is intentionally a template. Phase 1 is hardware-bound; record **measured**
> values here (never invented) before marking the Phase 1 gate GO.

## Decision

- **Engine:** Ollama _<version>_ (MLX backend) — endpoint `http://localhost:11434/v1`
- **Model:** _<qwen3-coder:30b-a3b-qN — [verify] tag>_
- **Quant chosen:** _Q5 | Q6_ — rationale: _which held tool-calling (LLM-1.5)_
- **`iogpu.wired_limit_mb`:** _<value>_ — derived from _<total params × quant bits + KV
  headroom>_

## Measurements (LLM-1.4)

| Metric | Value |
|---|---|
| On-disk size (Q5 / Q6) | _ / _ |
| Resident size (loaded) | _ |
| Short-prompt tok/s | _ |
| Long-prompt (8–16k) tok/s | _ |
| Long-prompt stability | _ |
| Peak resident memory under load | _ |
| KV-cache headroom remaining | _ |

## Quant sensitivity (LLM-1.5)

- Q5 tool-calling: _pass/fail + notes_
- Q6 tool-calling: _pass/fail + notes_
- Selected: _Qn_

## Gate result (LLM-1.GATE)

- Memory budget within ~32–38 GB usable, no swap thrash: _yes/no_
- tok/s usable interactively: _yes/no_
- Tool-calling survives chosen quant: _yes/no_
- Endpoint streams `/v1/chat/completions`: _yes/no_
- **Decision:** _GO → Phase 2 | NO-GO → drop to Q4 / smaller MoE and re-measure_
