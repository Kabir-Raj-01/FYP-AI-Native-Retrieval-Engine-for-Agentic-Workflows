# ADR-016: Performance targets confirmed from profiled runs (SRS TBD-4, NFR-P1–P3, P7)

> Status: Accepted (2026-10-17, Kabir for the team; Samay/Vishal review after the fact) · Date: 2026-10-17 · Author(s): Kabir · Reviewer: Samay, Vishal
> Scope: the team laptop (Intel Core i7-1185G7, 8 logical cores, 16 GB RAM, no GPU),
> CPU-only models (ADR-004, ADR-005). Resolves SRS Appendix B **TBD-4**.

## Context
SRS §6: performance targets are provisional, "confirmed against the first profiled runs
(Week 14); any change is recorded in an ADR". ADR-001 confirmed NFR-P1; ADR-005 measured
reranking and left the single-worker figure to the controller-ON E1 run. The E1 runs now
exist (FYP1_REPORT_DRAFT §5.2), so all four targets can be checked.

## Measurements
| NFR | Target | Measured | Source | Status |
|---|---|---|---|---|
| P1 direct dense retrieval, k = 20 | median ≤ 200 ms | median 104 ms (p90 127 ms); 142 ms in the integration test | ADR-001; `tests/integration/test_direct_live.py` | **met** |
| P2 controller overhead per step (select incl. cache lookup, evidence, monitor; planning excluded) | median ≤ 100 ms | median 10 ms, p90 142 ms over 2,912 steps | per-stage `latency_ms` in the traces of `20261016-073716_controller_on_dev_62bb0be` | **met** |
| P3 cross-encoder, 20 candidates, CPU | median ≤ 1.5 s | **uncached:** median 32.3 s per 20 pairs (6 calls, 30.4–33.6 s; one process, on battery, 512-token pairs); 16.9 s with 4 workers (ADR-005); 10–24 s earlier (LIBRARY_NOTES). **In use:** 68 % of the 1,963 reranker calls of the E1 controller-ON run took ≤ 1 s (score cache), median 10 ms | procedure below, 2026-10-17 | **not met** |
| P7 300-task dev-subset run | ≤ 3 h, 4 workers | one worker (E1 configs: one local LLM serialises calls). One-shot: ~1.5 min. controller_on: ~1.0 h active for the cold B = 2 sub-run (2.89 h wall incl. a 111-min Modern Standby pause), 0.25 h / 0.18 h for B = 3 / 4. fixed_iteration: B = 3 0.70 h, **B = 4 3.70 h** (fresh planner REWRITEs, partly on battery) | E1 manifests and `run.subrun_done` events | **not met** for fixed_iteration B = 4 |

P3 procedure (repeatable; the production scorer with its cache off, so every pair is
scored): for the first 7 bundles of `20261016-073536_hybrid_oneshot_dev_62bb0be/b1`, time
`CrossEncoderScorer(cfg.rerank, cache_path=None)(bundle.goal, [e.text for e in
bundle.evidence][:20])`; the first call (model load, 100.8 s) is dropped and the median of
the other six is reported.

Why P3 cannot be met on this CPU: bge-reranker-base has ≈ 278 M parameters; 20 pairs ×
512 tokens ≈ 2 × 278 M × 10,240 ≈ 5.7 TFLOP, and a laptop CPU sustains roughly
100–200 GFLOP/s in PyTorch — 30–60 s. The 1.5 s target assumed a GPU or a much smaller
model; MiniLM-L6 was rejected for quality (ADR-005).

## Decision
1. **P1 and P2 are confirmed** unchanged.
2. **P3** keeps 1.5 s as the GPU/hosted reference (`rerank.device: cuda`, unverified — no
   GPU in the team). For the CPU team laptop the target becomes: **uncached** reranking of
   20 candidates median ≤ 35 s; **cached** pairs (the persistent score cache,
   `rerank.score_cache_enabled`) ≤ 10 ms. Evaluation runs are offline, so the cost is run
   time, not user latency; the quality gain of bge (ADR-005) is kept.
3. **P7** keeps "≤ 3 h with 4 workers" as the reference for a hosted or GPU LLM. For the
   CPU team laptop with one local LLM (ADR-004) the target becomes: **each 300-task E1
   sub-run completes within 4 h with one worker** (`eval.max_workers: 1`, as in the E1
   configs), with the LLM and reranker caches warm from earlier runs where available.
4. The FYP-I report lists P3 and P7 as measured limitations (§7). The SRS document
   (outside the repo) should mark TBD-4 as resolved by this ADR.

## Alternatives considered
- Keep the original targets and only report them as missed — leaves TBD-4 open, and gives
  no regression bound for the hardware the team actually has.
- Switch to MiniLM-L6 to approach 1.5 s — rejected by ADR-005 on quality.
- Four eval workers — the single local Ollama answers one call at a time, so extra
  workers queue past `llm.<role>.timeout_s` (comment in the E1 configs).

## Consequences
Re-measure P3 on AC power if a precise CPU figure is needed (battery figures vary by
about 2×, LIBRARY_NOTES). FYP-II performance work (TASKS, March 2027: batching, caching)
can reduce P7; a GPU or hosted model would make the reference targets testable.

## Reversal condition
A GPU or hosted model becomes available (re-test the reference targets), or a change to
the reranker or planner model (ADR-004/005) moves the CPU figures by more than 2×.

## Amendment A1 (2026-10-17) — P3 re-measured on AC power
The P3 figure above was measured on battery, which LIBRARY_NOTES says not to do (the CPU
throttles while `Win32_Processor.CurrentClockSpeed` still reports the full clock). Re-run
of the same procedure on AC power: **median 11.3 s per 20 uncached pairs** (6 calls,
10.0–12.3 s; model load 48 s) — about 3× faster than on battery. The FLOP estimate above
is also corrected: 192 M of the model's 278 M parameters are the token-embedding table (a
lookup, no multiply-adds); the 12 transformer layers (≈ 85 M) cost ≈ 0.17 GFLOP per token.
Twenty pairs padded to ~450–512 tokens are ≈ 9,000–10,000 tokens ≈ 1.5–1.9 TFLOP, so
11.3 s is ≈ 0.13–0.17 TFLOP/s — about a third of this CPU's AVX-512 peak (4 cores × 32 FP32
FLOP/cycle × ~3.5 GHz ≈ 0.45 TFLOP/s). Even at that peak the 20 pairs would take ≈ 4 s, so
1.5 s needs a GPU, as Decision 2 says.

Decision 2's CPU target is tightened accordingly: **uncached reranking of 20 candidates,
median ≤ 15 s on AC power** (was ≤ 35 s); cached pairs ≤ 10 ms unchanged. Battery timings
are not benchmarks. Status of P3 is unchanged: not met against the 1.5 s reference.
