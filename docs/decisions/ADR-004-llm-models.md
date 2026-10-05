# ADR-004: LLM providers and models (planner, verifier, generator) and prompt versions

> Status: Proposed · Date: 2026-09-25 · Author(s): Vishal · Reviewer: Samay

## Context
The planner and generator affect every controlled and fixed-iteration result; the model
must be fixed, affordable across hundreds of runs, and reliable at JSON output.

## Decision (fill in, then Accept)
- Planner: provider ___, model ___ (exact string from provider docs, checked on ___),
  temperature 0, why: ___
- Generator: provider ___, model ___ ; identical for all systems.
- Verifier: disabled / provider ___, model ___.
- Estimated cost per full dev run: ___ (from a 50-task pilot) ; per-run cap
  `llm.max_cost_usd_per_run` = ___.
- Prompt templates and their hashes: `planner/init.j2` ___, `rewrite.j2` ___, `split.j2` ___,
  `eval/prompts/answer.j2` ___.

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
