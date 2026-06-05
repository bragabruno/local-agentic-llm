# ADR 0000 — Why local (Phase 0 decision gate)

- **Status:** Accepted — gate result **GO**
- **Date:** 2026-06-05
- **Tickets:** LLM-0.1, LLM-0.2, LLM-0.3, LLM-0.GATE

## The question

> What specific workload requires this system to run locally rather than on a hosted
> frontier API?

## Purpose statement (recorded verbatim)

> **Everything is local.**

The operator runs this system under a standing policy that all inference happens
on-device; no request leaves the machine. This is a deliberate **hard-offline / by-policy**
posture, not a per-task optimization.

## Classification

- **Driver:** `hard-offline` (standing local-only policy).
- This is a clean, recorded driver → **not** the kill branch. The hosted-API fallback is
  **not** taken.

## Does this require a model larger than 48 GB? — **No**

A local-only policy calls for a **private** model, not necessarily a large one. No
documented capability gap has been shown where the resident 30–35B-A3B MoE provably cannot
do a required sub-task. Therefore:

- The build targets the **resident MoE path** (Phase 1 → Phase 2).
- **Phase 3 (large-MoE streaming + router) stays LOCKED.** It unlocks only if a concrete
  capability gap appears at the Phase 2 gate *and* this answer changes to "yes" with
  justification.

## Consequences

- Proceed to **Phase 1** (resident MoE baseline behind Ollama, `localhost:11434/v1`).
- Memory is sized on **total** params × quant bits (all experts resident), never active
  params.
- If, during Phase 2, a hard sub-task is found that the resident model cannot handle, this
  ADR is revisited before any Phase 3 work begins — escalation to a remote API is excluded
  by the hard-offline driver, so the only Phase-3 option would be a **local** streamed
  big-MoE, subject to its prefill-OOM acceptance blocker.
