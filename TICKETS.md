# Implementation Tickets — Local Agentic LLM System (Mac M3 / 48 GB)

Derived from `docs/plan/local-agentic-llm-plan.md`. IDs: `LLM-<phase>.<n>`.
Status legend: `todo` / `in-progress` / `blocked` / `done`. All facts tagged **[verify]** must be confirmed before the consuming ticket starts.

---

## Phase 0 — Decision Gate (no build until gate passes)

### LLM-0.1 — Write the purpose statement · `todo`
Name the concrete workload that requires local inference over a hosted frontier API.
- **Depends:** —
- **Acceptance:** one-paragraph purpose statement committed to `docs/decisions/0000-why-local.md`, naming the driver and the workload it serves.

### LLM-0.2 — Classify the driver · `todo`
- **Depends:** LLM-0.1
- **Acceptance:** driver classified as exactly one of `data-residency` / `compliance-air-gap` / `cost-at-volume` / `hard-offline`; if none fits cleanly, the kill branch is recorded.

### LLM-0.3 — Answer "model > 48 GB required?" · `todo`
- **Depends:** LLM-0.2
- **Acceptance:** explicit yes/no recorded with justification. Default no. A yes requires a documented capability gap, not a preference. Result sets whether Phase 3 is ever unlocked.

### LLM-0.GATE — Phase 0 go/no-go · `todo`
- **Depends:** LLM-0.1, LLM-0.2, LLM-0.3
- **Acceptance (No-Go / kill):** no clean driver → ship LangGraph + hosted-frontier-API fallback; close the project. **Acceptance (Go):** clean driver recorded → proceed to Phase 1.

---

## Phase 1 — Resident MoE Baseline

### LLM-1.1 — Install/confirm Ollama 0.19+ (MLX backend) · `todo`
- **Depends:** LLM-0.GATE = Go
- **Acceptance:** Ollama 0.19+ **[verify]** running with MLX serving confirmed.

### LLM-1.2 — Raise Metal working-set ceiling · `todo`
- **Depends:** LLM-1.1
- **Acceptance:** `iogpu.wired_limit_mb` set to a documented value admitting chosen quant + target context; reasoning recorded (total params × quant bits + KV headroom).

### LLM-1.3 — Pull resident MoE at Q5 and Q6 · `todo`
- **Depends:** LLM-1.1
- **Acceptance:** one candidate (`Qwen3-Coder-30B-A3B` or `Qwen3.5 35B-A3B` **[verify]**) pulled at Q5 and Q6; on-disk + resident sizes recorded.

### LLM-1.4 — Measure throughput & memory · `todo`
- **Depends:** LLM-1.2, LLM-1.3
- **Acceptance:** recorded (measured, not invented): short-prompt tok/s, long-prompt (8–16k) tok/s + stability, peak resident memory, KV-cache headroom. No OOM at target context.

### LLM-1.5 — Quant sensitivity probe (tool-calling) · `todo`
- **Depends:** LLM-1.3
- **Acceptance:** tool-calling + code-gen smoke set run at Q5 vs Q6; degradation noted; quant that holds tool-call formatting selected.

### LLM-1.6 — Write resident-engine ADR · `todo`
- **Depends:** LLM-1.4, LLM-1.5
- **Acceptance:** ~1-page `docs/decisions/0001-resident-engine.md`: model, quant, engine, `wired_limit`, measured tok/s, memory budget, rationale.

### LLM-1.GATE — Phase 1 go/no-go · `todo`
- **Depends:** LLM-1.4, LLM-1.5, LLM-1.6
- **Acceptance (Go):** memory budget holds (~32–38 GB usable, no swap thrash), tok/s usable interactively, tool-calling survives quant, endpoint serves `/v1/chat/completions` streaming. **No-Go:** drop to Q4 / smaller MoE and re-measure — do not skip to Phase 3.

---

## Phase 2 — Agent Layer (LangGraph)

### LLM-2.1 — Wire LangGraph to local endpoint · `todo`
- **Depends:** LLM-1.GATE = Go
- **Acceptance:** OpenAI-compatible client at `http://localhost:11434/v1`; chat + streaming + tool-call round-trips confirmed.

### LLM-2.2 — Define tool schemas · `todo`
- **Depends:** LLM-2.1, LLM-1.5
- **Acceptance:** 2–3 tools (file/search, calculator or code-exec, retrieval) defined; local model emits well-formed tool calls.

### LLM-2.3 — Stand up RAG · `todo`
- **Depends:** LLM-2.1
- **Acceptance:** small embedding model **[verify]** + local vector store + ingest path + retrieval tool exposed to the agent; embedding footprint fits the shared 48 GB budget.

### LLM-2.4 — Implement multi-step loop · `todo`
- **Depends:** LLM-2.2, LLM-2.3
- **Acceptance:** planner/executor graph with tool-result feedback, stop conditions, max-step guard.

### LLM-2.5 — Context / prompt-budget management · `todo`
- **Depends:** LLM-2.4, LLM-1.4
- **Acceptance:** per-step token budget enforced against measured KV headroom (history truncate/summarize, retrieved-chunk cap); no mid-loop overflow.

### LLM-2.6 — Representative end-to-end task · `todo`
- **Depends:** LLM-2.4, LLM-2.5
- **Acceptance:** `eval/e2e_task.py` exercising ≥2 tools + RAG + ≥3 reasoning steps completes against the local endpoint with a correct, verifiable result; RAG demonstrably influences the answer.

### LLM-2.GATE — Phase 2 go/no-go · `todo`
- **Depends:** LLM-2.6
- **Acceptance (Done — 90% case):** e2e task passes, tool calls well-formed across the full loop, prompt budget holds → **project complete**. **Go → Phase 3** only if LLM-0.3 = yes AND a concrete hard sub-task exceeds the resident model. **No-Go:** fix at resident/quant/prompt layer (LLM-1.5 / LLM-2.5), not via Phase 3.

---

## Phase 3 — CONDITIONAL: Large-MoE Streaming + Router
> Unlocks only if LLM-0.3 = yes AND LLM-2.GATE found a real capability gap. PoC maturity, no releases.

### LLM-3.1 — Install mlx-flash / LM Studio flash mode · `todo` · CONDITIONAL
- **Depends:** LLM-2.GATE = Go-to-3
- **Acceptance:** flash streaming installed; **[verify]** PoC state; exact commit/build pinned.

### LLM-3.2 — Stand up large-MoE streaming endpoint · `todo` · CONDITIONAL
- **Depends:** LLM-3.1
- **Acceptance:** OpenAI-compatible endpoint for `Qwen3-Coder-480B-A35B` **[verify]**; single-token generation confirmed.

### LLM-3.3 — Prefill stress test (BLOCKER) · `todo` · CONDITIONAL
- **Depends:** LLM-3.2
- **Acceptance:** real long agent prompts (8–16k) tested for prefill OOM. **If OOM at real sizes, Phase 3 fails acceptance — do not build the router.**

### LLM-3.4 — Build the router · `todo` · CONDITIONAL
- **Depends:** LLM-3.3 = pass
- **Acceptance:** resident MoE drives all common steps; router escalates only the rare hard sub-task to a configurable target (remote frontier API preferred; streamed big-MoE only when local is mandatory). Streaming never placed in interactive/multi-user role.

### LLM-3.5 — Full architecture doc · `todo` · CONDITIONAL
- **Depends:** LLM-3.4
- **Acceptance:** `docs/architecture/system.md` covering resident path, streaming path, router policy, failure modes.

### LLM-3.GATE — Phase 3 go/no-go · `todo` · CONDITIONAL
- **Depends:** LLM-3.3, LLM-3.4, LLM-3.5
- **Acceptance (Go):** prefill blocker passes under real prompts, router policy holds. **No-Go:** prefill OOMs or PoC too unstable → route hard sub-task to remote frontier API; ship hybrid without local big-MoE; record decision.

---

## Cross-cutting tickets

### LLM-X.1 — Repo scaffold · `todo`
- **Acceptance:** directory layout from the plan created (`docs/`, `infra/`, `agent/`, `eval/`), `requirements.txt` with pinned versions (no `latest`), `README.md`.

### LLM-X.2 — Thermal monitoring during long loops · `todo`
- **Depends:** LLM-1.4
- **Acceptance:** temp/clock monitoring in place; planning tok/s uses throttled (not peak) numbers; cooldown/backoff available for sustained loops.

### LLM-X.3 — Lint / test / typecheck gate · `todo`
- **Acceptance:** ruff + pytest (`eval/`) + typecheck wired; green before any phase is marked done.
