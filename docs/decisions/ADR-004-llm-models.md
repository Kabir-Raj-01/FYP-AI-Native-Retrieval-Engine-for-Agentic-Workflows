# ADR-004: LLM providers and models (planner, verifier, generator) and prompt versions

> Status: Accepted (2026-10-08, Kabir for the team; Vishal/Samay review after the fact) · Date: 2026-09-25 · Author(s): Vishal · Reviewer: Samay

## Context
The planner and generator affect every controlled and fixed-iteration result; the model
must be fixed, affordable across hundreds of runs, and reliable at JSON output.

## Decision (2026-10-08)
Decided by Kabir for the team so that FYP-I could be completed; Vishal/Samay review after
the fact.
- Planner: provider `openai_compatible`, model **`qwen2.5:3b`** on a local Ollama v0.40.1
  server (`http://127.0.0.1:11434/v1`; tag checked with `ollama pull` on 2026-10-08),
  temperature 0. Why: no API key or credit exists for the project; it runs on the team's
  CPU-only laptops (i7-1185G7, 16 GB RAM, ~2 GB model) and follows the JSON schema well
  enough after the PLANNER_SPEC §7a validator fix (pilot below). A 7B model would not fit
  next to the engine's models in RAM on this machine.
- Generator: the same model and server, temperature 0, identical for all systems
  (`eval/generator.py`, post hoc over the bundles).
- Verifier: disabled (not built; `llm.verifier.enabled = false`).
- Cost: $0 (local); `llm.prices` stays empty, `llm.max_cost_usd_per_run` unchanged.
  Wall-clock instead: ~15 s per planner call (median TRANSFORM 14.5 s,
  `configs/latency_profile.yaml`), ~5 s cold start.
- Prompt templates: hashed in every controlled run's manifest (`prompt_hashes`) and in
  `answers.json` (`prompt_hash`).

### Pilot (dev subset, 2026-10-08)
- Before the validator fix: 0 of 6 INIT attempts accepted (5 × UNDECLARED_ENTITY from an
  extractor bug, 1 × SCHEMA). After it: 10 of 10 tasks got an accepted INIT plan, one of them
  on the second attempt (1 SCHEMA rejection in 11 attempts: the model sent two JSON
  objects). JSON validity on the calibration sample is reported in ADR-013.
- Reversal-condition watch: JSON validity must stay ≥ 95 % on dev; if it does not, switch
  to a hosted model (any OpenAI-compatible or Anthropic endpoint is a config change plus
  `LLM_API_KEY`; amend this ADR) — the planner's fallback keeps the engine working in the
  meantime.

### Notes from the LLM client build (2026-10-05, Kabir) — inputs for the decision
- With `provider: anthropic`, temperature is not sent (anthropic 1.11.0 has no such
  parameter; current models reject sampling params); repeatability then rests on the
  response cache (`llm.cache_enabled`, `llm.cache_namespace`).
- Current Claude models think adaptively by default and thinking tokens count towards
  `max_tokens`; `llm.planner.max_output_tokens = 600` may be too small — size it in the
  50-task pilot.
- Any OpenAI-compatible endpoint works with config + `.env` only (`base_url`, `model`,
  `LLM_API_KEY`), e.g. OpenAI, a router, or a local Ollama server.

## Alternatives considered
Local open-weight model via Ollama (no cost, slower, weaker JSON adherence) vs a small
hosted API model (fast, reliable JSON, costs credit). Record the pilot comparison
(JSON validity rate, planner rejection rate, latency, cost) here.

## Consequences
`llm.<role>.model`, prompt templates Frozen for reported experiments.

## Reversal condition
If the chosen model is deprecated or the JSON validity rate on dev < 95 %.
