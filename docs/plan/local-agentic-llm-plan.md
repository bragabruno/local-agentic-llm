# Build Plan — Local Agentic LLM System (Mac M3 / 48 GB)

**Target file (on approval):** `docs/plan/local-agentic-llm-plan.md`
**Status:** plan draft, pending approval. All model/version/throughput facts tagged **[verify]** must be re-confirmed against current releases before the phase that consumes them begins.

---

## Context

Goal: build a local, on-device **agentic LLM system** on a Mac M3 with 48 GB unified memory — a LangGraph-style agent loop (tool calling, RAG, multi-step reasoning) talking to a **local OpenAI-compatible HTTP endpoint**.

The expensive, irreversible decisions here are (a) committing to local inference at all, and (b) committing to a model that exceeds memory. Both are treated as **hypotheses to justify**, not givens. The plan front-loads a kill gate, makes the high-value resident path cheap and first, and gates the fragile streaming path behind an explicit Phase 0 justification.

**Hardware envelope (constant across phases):** 48 GB unified memory → ~32–38 GB practically usable for weights + KV cache after OS + the default Metal working-set ceiling (raisable via `iogpu.wired_limit_mb`). MLX is the single shared base layer; the serving engine on top is pick-one; LangGraph talks to the OpenAI-compatible endpoint regardless of engine.

**Memory-sizing rule (applies everywhere):** a resident MoE keeps **all** expert weights in memory. "Active params" is compute-per-token, **not** memory. Size every budget on **total params × quant bits**.

---

## Phase 0 — Decision Gate (non-negotiable, before any architecture or code)

**The one question, answered in writing:**

> **What specific workload requires this system to run locally rather than on a hosted frontier API?**
> Valid answers: client/data-residency, compliance/air-gap, per-token cost at real volume, or a hard offline requirement.

### Tasks
- **0.1** Write the purpose statement in `docs/plan/local-agentic-llm-plan.md` (or a linked `docs/decisions/0000-why-local.md`). One paragraph, naming the concrete driver and the workload it serves.
- **0.2** Classify the driver into exactly one of: `data-residency` / `compliance-air-gap` / `cost-at-volume` / `hard-offline`. If none fits cleanly → **kill branch**.
- **0.3** Separately answer: *does this driver require a model that exceeds 48 GB?* Default answer is **no**. A compliance/air-gap need calls for a **private** model, not necessarily a large one. Only a documented capability gap (the resident MoE provably cannot do the hard sub-task) justifies considering Phase 3. Record the justification or the absence of one.

### Acceptance criteria
- A recorded, verbatim purpose statement with a single classified driver, **or**
- A documented kill decision.
- An explicit yes/no on "model > 48 GB required," with justification if yes.

### Kill branch (honest exit — not a rubber stamp)
If 0.2 yields no clean answer: **stop. Do not build local inference.** Recommended fallback, written into the doc as the project's own recommendation:

> Put LangGraph in front of a **hosted frontier API** (OpenAI-compatible client pointed at the hosted endpoint). You get the agent loop, tool calling, and RAG — Phase 2's value — with none of the local-inference cost, thermal, or maturity risk. Revisit local only if a concrete driver later appears.

This is the expected outcome for most general "I want it local" impulses. Treat reaching it as a success.

### Go/No-Go
- **No-Go (kill):** no clean driver → ship the hosted-API + LangGraph fallback, end of project.
- **Go → Phase 1:** clean driver recorded. Proceed to resident baseline regardless of the model-size answer (Phase 1 is required even if Phase 3 is later unlocked).

---

## Phase 1 — Resident MoE Baseline (the 90% case)

Stand up a **30–35B-A3B MoE** at Q5/Q6 behind Ollama and prove the memory budget and throughput on real hardware.

**Model candidates [verify]:**
- `Qwen3-Coder-30B-A3B` — agentic-coding focus.
- `Qwen3.5 35B-A3B` — general.
- ~30–35B total, ~3B active. At **Q5/Q6** ≈ **20–26 GB** resident → leaves ~10–18 GB for KV cache + context.

**Engine:** **Ollama 0.19+** (MLX under the hood; best for interactive agents). Endpoint: `http://localhost:11434/v1`. (Alternatives for the ADR to weigh: LM Studio (GUI+server), mlx-lm (max throughput / fine-tune), vllm-mlx (concurrency / agent fleets, ~1,150 tok/s aggregate at 32 users). **[verify versions]**)

### Tasks
- **1.1** Install/confirm Ollama 0.19+ with the MLX backend. **[verify]** the version actually ships MLX serving.
- **1.2** Raise the Metal working-set ceiling: set `iogpu.wired_limit_mb` to a value that admits the chosen quant + target context. Document the exact value and the reasoning (total params × quant bits + KV headroom).
- **1.3** Pull one resident MoE candidate at Q5 **and** Q6. Record on-disk and resident sizes.
- **1.4** Measure, with the model loaded: (a) tok/s at short prompt, (b) tok/s and stability at a **long** prompt (target the agent's realistic prompt size, e.g. 8–16k tokens), (c) peak resident memory under load, (d) KV-cache headroom remaining. (Do not invent target tok/s — record measured values.)
- **1.5** Probe quant sensitivity: run a small tool-calling + code-gen smoke set at Q5 vs Q6; note any degradation in tool-call formatting / code correctness. Pick the quant that holds tool-calling.
- **1.6** Write a **lean ADR** (`docs/decisions/0001-resident-engine.md`): chosen model, quant, engine, `wired_limit` value, measured tok/s, memory budget, why. ~1 page. **Not** the full architecture doc.

### Acceptance criteria
- Endpoint answers OpenAI-compatible `/v1/chat/completions` and streams tokens.
- Peak resident memory + KV cache stays within ~32–38 GB usable with the OS healthy (no swap thrash).
- Long-prompt run completes without OOM at the target context size.
- Tool-calling output format is intact at the chosen quant (1.5).
- ADR committed.

### Go/No-Go
- **Go → Phase 2:** memory budget holds, tok/s usable for interactive agent steps, tool-calling survives quant.
- **No-Go:** if memory/throughput fail → drop to Q4 or a smaller-total MoE and re-measure before proceeding. Do **not** jump to Phase 3 to "solve" a resident-tuning problem.

**Effort estimate (flagged estimate):** ≈ **half a day** if downloads are warm; +half day if `wired_limit`/quant needs iteration.

---

## Phase 2 — Agent Layer (LangGraph against the endpoint)

Build the agent loop on top of the Phase 1 endpoint. This is where the system's value is realized.

### Tasks
- **2.1** Wire LangGraph's OpenAI-compatible client to `http://localhost:11434/v1`. Confirm chat + streaming + tool-call round-trips against the local model.
- **2.2** Define tool schemas (start with 2–3: a file/search tool, a calculator or code-exec tool, a retrieval tool). Validate the local model emits well-formed tool calls (depends on 1.5 result).
- **2.3** Stand up **RAG**: an embedding model + a local vector store, an ingest path, and a retrieval tool exposed to the agent. **[verify]** embedding-model choice; keep it small to protect the memory budget (embeddings share the same 48 GB).
- **2.4** Implement the multi-step loop: planner/executor graph, tool-result feedback, stop conditions, max-step guard.
- **2.5** **Context / prompt-budget management:** enforce a token budget per step that respects the KV headroom measured in 1.4 (truncate/summarize history, cap retrieved-chunk size). This is the integration point between Phase 1's memory reality and Phase 2's prompt growth.
- **2.6** Build one **representative end-to-end task** that exercises ≥2 tools + RAG + ≥3 reasoning steps. Use it as the phase's acceptance test.

### Acceptance criteria
- The representative multi-tool task completes end to end against the **local** endpoint, producing a correct, verifiable result.
- Tool calls are well-formed across the full loop (not just single-shot).
- Prompt budget stays inside KV headroom — no mid-loop OOM, no context overflow errors.
- RAG retrieval demonstrably influences answers (cite-back or retrieval-hit check).

### Go/No-Go
- **Go → Phase 3** *only if* Phase 0.3 justified a model > 48 GB **and** a concrete hard sub-task exceeds the resident model's capability. Otherwise **the project is done at Phase 2** — that is the intended outcome for the 90% case.
- **No-Go:** loop instability or tool-call breakage → fix at the resident/quant/prompt layer (revisit 1.5 / 2.5), not via Phase 3.

**Effort estimate (flagged estimate):** ≈ **1.5–3 days** (tooling + RAG ingest + loop + budget management).

---

## Phase 3 — Conditional: Large-MoE Streaming + Router

**Unlocks ONLY if Phase 0.3 justified a model > 48 GB and Phase 2 surfaced a real capability gap.** This is **not** a default. Maturity of the streaming path: **proof of concept, no releases.**

**Model [verify]:** a **large MoE, never dense** — e.g. `Qwen3-Coder-480B-A35B` (480B total, 35B active, 256K ctx). Streamed via **mlx-flash** (monkey-patch over `mlx_lm`/`mlx-engine`, surfaced as "Flash Weight Streaming" in LM Studio). Streams only top-K active experts per token; resident set ~7–15 GB; throughput tracks SSD bandwidth; **expect seconds per token.**

### The three streaming constraints — treat as ACCEPTANCE BLOCKERS
1. **Prefill OOM** on prompts ≳5k tokens — the host engine builds the full KV-cache graph before the layer-by-layer patch frees memory. **Agent prompts routinely exceed 5k.** Must be verified under *real* long agent prompts, not toy inputs, before anything relies on it.
2. **Single-stream** — requests serialize; no concurrency.
3. **Slow** — multi-step loops run minutes-to-hours.

### Tasks
- **3.1** Install LM Studio flash mode / mlx-flash. **[verify]** current state of the PoC; pin the exact commit/build used.
- **3.2** Stand up the large-MoE streaming endpoint (OpenAI-compatible). Confirm single-token generation works at all.
- **3.3** **Prefill stress test (blocker):** feed it the *actual* long agent prompts Phase 2 produces (8–16k tokens). Confirm whether prefill OOMs. **If it OOMs at real prompt sizes, Phase 3 fails its acceptance — do not build the router on top.**
- **3.4** Build the **router**: the resident MoE drives the many agent steps; the router escalates **only** the rare hard sub-task to either (a) a remote frontier API or (b) the streamed big-MoE endpoint. Make the escalation target configurable — prefer remote frontier API when air-gap allows it; use streamed big-MoE only when local is mandatory.
- **3.5** Write the **full A+B+hybrid architecture doc** (`docs/architecture/system.md`) — only reached here. Resident path, streaming path, router policy, failure modes.

### Acceptance criteria
- Prefill survives real (long) agent prompts without OOM (3.3) — **hard blocker**.
- Router escalates only the intended rare sub-task; resident path still serves the common case.
- Streaming endpoint is never placed in an interactive or multi-user role (single-stream, seconds/token are understood and bounded).
- Architecture doc committed.

### Go/No-Go
- **Go (operational):** prefill blocker passes under real prompts and router policy holds.
- **No-Go:** prefill OOMs at real sizes, or PoC too unstable → **route the hard sub-task to a remote frontier API instead** and ship the hybrid without the local big-MoE. Record the decision.

**Effort estimate (flagged estimate):** ≈ **3–6+ days, high variance** — PoC fragility dominates. Could be unbounded if the prefill blocker can't be worked around.

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| **Thermal throttling** — sustained agent loops can throttle M3 up to ~40% | Throughput silently drops mid-run | Monitor temps/clocks during long loops; budget step rate against throttled, not peak, tok/s; add cooldown/backoff; treat throttled tok/s as the planning number. |
| **mlx-flash PoC fragility** — no releases | Phase 3 may not work at all | Keep Phase 3 strictly conditional and gated; pin exact build (3.1); always keep remote-frontier-API as the router's fallback escalation target. |
| **Quant sensitivity** — tool-calling/code degrade under aggressive quant | Broken tool calls in the agent loop | Probe Q5 vs Q6 on a tool-calling smoke set (1.5); never go below the quant that holds tool-call formatting; re-test after any model swap. |
| **Prefill OOM** (Phase 3) on prompts ≳5k | Streaming path unusable for agents | Make 3.3 a hard acceptance blocker tested with *real* long prompts; if it fails, escalate hard sub-tasks to remote API instead. |
| **Model-version drift** — names/figures change | Plan built on stale facts | Every model/version/throughput fact tagged **[verify]**; re-confirm at the start of the phase that consumes it; ADR records the exact versions actually used. |
| **Memory mis-sizing** (active vs total params) | OOM after "it fit in theory" | Always size on total params × quant bits with KV headroom; verify empirically in 1.3/1.4, not on paper. |

---

## Recommended Repo / Directory Layout

```
local-llm/
├── docs/
│   ├── plan/local-agentic-llm-plan.md      # this plan
│   ├── decisions/
│   │   ├── 0000-why-local.md               # Phase 0 purpose statement / kill record
│   │   ├── 0001-resident-engine.md         # Phase 1 ADR
│   │   └── ...
│   └── architecture/system.md              # written ONLY if Phase 3 reached
├── infra/
│   ├── ollama/                             # engine config, wired_limit notes
│   └── streaming/                          # Phase 3 only: mlx-flash / LM Studio config
├── agent/
│   ├── graph.py                            # LangGraph loop
│   ├── tools/                              # tool schemas + impls
│   ├── rag/                                # ingest, vector store, retrieval tool
│   ├── router.py                           # Phase 3 only: escalation policy
│   └── budget.py                           # context/prompt-budget management
├── eval/
│   ├── smoke_toolcalling.py                # Phase 1.5 quant probe
│   └── e2e_task.py                         # Phase 2 representative task
├── requirements.txt                        # pinned versions, no `latest`
└── README.md
```

(Python + asyncio backend, pytest for `eval/`, ruff for lint — per repo conventions.)

---

## Effort Estimates (all flagged as estimates)

| Phase | Estimate |
|---|---|
| Phase 0 — Decision gate | ~1–2 hours (thinking + writing, no build) |
| Phase 1 — Resident baseline | ~0.5 day (warm), +0.5 day if tuning |
| Phase 2 — Agent layer | ~1.5–3 days |
| Phase 3 — Streaming + router (conditional) | ~3–6+ days, high variance |

---

## System-Level Definition of Done

The system is done when **one** of:

- **(Kill path)** Phase 0 recorded no clean local driver, and a LangGraph + hosted-frontier-API system runs the representative task — local inference correctly **not** built; **or**
- **(Resident path — the 90% case)** Phases 0–2 complete: a recorded purpose statement, a resident MoE behind `localhost:11434/v1` within the 48 GB budget with tool-calling intact, and a LangGraph agent completing a representative multi-tool + RAG + multi-step task end to end against the local endpoint; **or**
- **(Hybrid path)** all of the above **plus** Phase 3's router escalating only the rare hard sub-task, with the streaming endpoint's prefill verified under real long prompts (or the hard sub-task routed to a remote API), and the full architecture doc written.

In all cases: required ADRs/decision records committed; every model/version fact used was **[verify]**-confirmed at consumption time and recorded; memory sized on total params; no streaming of a dense model; streaming never used interactively or multi-user.

---

## Assumptions (explicitly flagged)

- **Repo convention:** no existing repo present (empty working dir); layout above assumes a greenfield `local-llm/` Python project per the user's stated stack (Python/asyncio, pytest, ruff). The plan's canonical target path mirrors the prompt's `docs/plan/...`.
- **All model names, quant→GB figures, tok/s, and version numbers** are from the prompt's authoritative mid-2026 context, carried verbatim and tagged **[verify]**. None invented; none web-re-verified in this pass.
- **Long-prompt target (8–16k tokens)** is an assumed realistic agent prompt size to drive 1.4/3.3 — adjust to the actual workload once Phase 0 names it.
