# Action Policy, Utility, Monitor and Stopping

> Status: Stable (formulas) · Values: Draft until calibration ADR · Owner: Samay · Last updated: 2026-10-04
> Implementation: `engine/controller/` (policy + `signals.py`), `engine/monitor/`.
> All numbers below that are not in a worked example are config keys (CONFIG_SCHEMA.md).
> Amended by ADR-009: `plan_mode` selection (§2), the `MAX_STEPS` check timing (§6.4),
> and internal signal naming (§4) to avoid clashing with S_t's G/R_t.

## 1. Principles
- Deterministic: the same state + config always yields the same decision.
- Transparent: every decision records the eligible set, per-action signals and the rule
  that fired (`ActionDecision.rule`).
- Hard rules first (budget, preconditions), utility second, fixed tie-break last.
- The controller never calls an LLM. TRANSFORM *requests* a planner call; the `execute`
  node performs it.

## 2. Selection procedure (`Controller.select_action(state, cache_hit)`)
```
1. if budget.steps_used >= max_steps                → STOP(MAX_STEPS), rule="budget_guard"
2. E = {a : precondition(a) holds}                   (§3)
3. if retrieval_calls_used >= max_retrieval_calls:
       E -= {DENSE, SPARSE_HYBRID, RERANK_EXPAND, TRANSFORM}
       if E == ∅                                     → STOP(MAX_RETRIEVAL_CALLS), rule="budget_guard"
4. if E == ∅:
       reason = MAX_LLM_CALLS if TRANSFORM was excluded only for LLM budget else STAGNATION
                                                     → STOP(reason), rule="no_eligible_action"
5. if ablation.adaptive_actions is false            → fixed policy (§7), rule="fixed_policy"
6. for a in E: compute gain, cost, redundancy_signal, latency_signal (§4);
   U = α·gain − β·cost − γ·redundancy_signal − δ·latency_signal
7. a* = argmax U; ties broken by order REUSE > SPARSE_HYBRID > DENSE > RERANK_EXPAND > TRANSFORM
8. if U(a*) < controller.min_utility and retrieval_calls_used >= 1
                                                     → STOP(LOW_MARGINAL_UTILITY), rule="min_utility"
9. if a* == TRANSFORM: set plan_mode per PLANNER_SPEC §2's REWRITE/SPLIT conditions
                                                     (ActionDecision.plan_mode; ADR-009 §3)
10. return a*, rule="utility_argmax"
```
Step 8 never fires before the first retrieval call — a task always retrieves at least once.

Step 1 is a **defensive-only** guard (ADR-009 §12): in normal single-caller operation the
monitor's `MAX_STEPS` check (§6.4, evaluated near the end of the iteration that is about
to make `steps_used` reach `max_steps`) always stops the task before `select_action` could
be invoked again with `steps_used == max_steps`. Step 1 exists only to cover a caller that
invokes a step on a task the monitor should already have stopped (e.g. a race or a bug
elsewhere) — it must never fire in the normal flow, and `tests/unit/test_state_invariants.py`
plus the `max_steps` test (TESTING.md §4) assert this by construction. It stays written as
`steps_used >= max_steps`, unchanged — it is not the mechanism that makes the loop run
exactly `max_steps` real iterations; §6.4 is.

## 3. Preconditions
Let *preferred* = DENSE if `current_need_type == semantic`, SPARSE_HYBRID if `lexical`;
for `mixed` both are preferred. "Issued(a, q)" = query q (normalised) already appears in
`retrieval_history` with action a.
| Action | Eligible iff |
|---|---|
| REUSE | `ablation.cache` and `cache_hit is not None` for the pending query |
| DENSE | retrieval budget left, not Issued(DENSE, pending_query), and (DENSE is preferred **or** `stagnation_count ≥ 1` **or** Issued(preferred, pending_query)) |
| SPARSE_HYBRID | same rule as DENSE with the roles swapped |
| RERANK_EXPAND | retrieval budget left, `last_result` exists for the current gap's pending query, `last_result.depth < retrieval.max_k`, and at least one of `last_result.top_relevances` lies in `[controller.ambiguous_band.low, controller.ambiguous_band.high]` |
| TRANSFORM | `ablation.planner`, LLM budget left (`llm_calls_used < max_llm_calls`), `transforms_in_streak < monitor.max_transforms_per_streak`, and (`stagnation_count ≥ 1` **or** every retrieval action is blocked only by Issued(·)) |
| STOP | never chosen by utility; only by the rules in §2 and §6 |

The "or" clauses implement the proposal's *strategy switch*: when the preferred mode has
stagnated or already been tried, the other mode becomes available.

## 4. Utility signals (`engine/controller/signals.py`, one pure function each)
All signals are in [0, 1]. Weights `controller.weights.{alpha,beta,gamma,delta}` are
non-negative and sum to 1 (validated at config load).
Notation: q = embedding of `pending_query`; O = OPEN gaps; cos(·,·) clipped to [0,1];
ema(a) = `state.action_gain_ema.get(a, controller.prior_action_gain)`.
Code identifiers for these four signals are `gain`, `cost`, `redundancy_signal`,
`latency_signal` — not `G`/`C`/`R`/`L` — to avoid colliding with S_t's `G` (goal) and
`R_t` (retrieval history); they are assembled into `ActionScores(G=gain, C=cost,
R=redundancy_signal, L=latency_signal, U=...)` only at the point that model is
constructed (ADR-009 "Also"; `ActionScores`'s field names stay `G`/`C`/`R`/`L`/`U`, that
shape is frozen in STATE_SPEC).

**Evidence gain**
| Action | Formula |
|---|---|
| DENSE, SPARSE_HYBRID | `s_gap·gap_align + s_ema·ema(a) + s_type·type_match(a)` with `gap_align = max_{g∈O} cos(q, emb(g.description))` (1.0 if `ablation.gap_tracking` false and q = goal); `type_match` from `controller.type_match` table; sub-weights `controller.gain_subweights` sum to 1 |
| REUSE | `hit.similarity × (n_new / n)` where n = number of candidates in the hit and n_new = those whose chunk_id is not already in E_t — i.e. how much of the cached result is not yet evidence |
| RERANK_EXPAND | `0.5·ambiguity + 0.5·ema(RERANK_EXPAND)`, ambiguity = fraction of `last_result.top_relevances` inside the ambiguous band |
| TRANSFORM | `min(1, stagnation_count / monitor.stagnation_patience)`; 1.0 if eligible only because all retrieval actions are blocked |

**Cost** = `(controller.cost_units[a] / max(cost_units)) × (0.5 + 0.5·budget.pressure())`.
Costs rise as the budget is consumed. With `ablation.budget_awareness` false, pressure = 0.

**Redundancy signal**
| Action | Formula |
|---|---|
| DENSE, SPARSE_HYBRID | max cos(q, q′) over queries q′ previously issued **with the same action**; 0 if none |
| RERANK_EXPAND | fraction of `last_result.candidate_ids` already in E_t |
| REUSE, TRANSFORM | 0 |
With `ablation.redundancy_signal` false, redundancy_signal = 0.

**Latency signal** = `latency_ms[a] / max(latency_ms)`, from `configs/latency_profile.yaml`
if present (generated by `python scripts\latency_profile.py` on the dev subset), else
`controller.default_latency_ms`. The file's `latency_ms` map has the same five keys as
`controller.default_latency_ms` (REUSE, DENSE, SPARSE_HYBRID, RERANK_EXPAND, TRANSFORM);
`source.<action>` is `measured` or `default` (TRANSFORM stays `default` until the planner and
LLM client exist), and `p90_ms`, `generated_at`, `git_commit`, `collection`, `queries` are
informational. What each action times: REUSE = a `Cache.lookup` exact hit; DENSE = dense
search at `retrieval.default_k`; SPARSE_HYBRID = hybrid search at `default_k`; RERANK_EXPAND =
dense search at `min(default_k + expand_step_k, max_k)` — medians of wall-clock ms with a cold
query embedding each time; reranking is excluded (it follows every candidate-producing
action alike).

## 5. Action → execution mapping
See ARCHITECTURE §5. SPARSE_HYBRID executes mode `sparse` when `current_need_type` is
`lexical`, `hybrid` otherwise. RERANK_EXPAND re-issues `last_result.query` in
`last_result.mode` at depth `min(last_result.depth + retrieval.expand_step_k, retrieval.max_k)`
and passes on only candidates not in `seen_candidate_ids`.

## 6. Trajectory monitor (`TrajectoryMonitor.observe`, after every step)
Runs in this order:
1. **Progress bookkeeping.** If the step produced candidates (DENSE, SPARSE_HYBRID,
   RERANK_EXPAND, REUSE): gain = `last_update.evidence_gain`;
   `ema(a) ← d·ema(a) + (1−d)·gain` with d = `controller.ema_decay`;
   if gain < `monitor.stagnation_threshold` → `stagnation_count += 1` (and the query is a
   "failed query" for the planner), else `stagnation_count = 0`, `transforms_in_streak = 0`.
   If the step was TRANSFORM → `transforms_in_streak += 1` (stagnation_count unchanged).
2. **Parent propagation.** A SUPERSEDED gap becomes RESOLVED when all children are
   RESOLVED; it is treated as closed-unresolved when all children are closed and any is
   EXHAUSTED.
3. **Gap advancement.** If the current gap is RESOLVED → advance. Else if
   `stagnation_count ≥ monitor.stagnation_patience` and
   `transforms_in_streak ≥ monitor.max_transforms_per_streak` → mark current gap
   EXHAUSTED, then advance.
   *Advance* = call the shared `advance(state)` function (STATE_SPEC §3 `AdvanceResult`;
   ADR-009 §4): `current_gap_id` := lowest-index OPEN gap's id (`pending_query`,
   `current_subgoal`, `current_need_type` are computed properties of that gap and so
   update automatically, without a separate write — ADR-009 §4 correction); reset
   `stagnation_count`, `transforms_in_streak`, `last_result` (and so, implicitly,
   `last_result.depth`, which replaces the old `current_depth` field — STATE_SPEC §4).
4. **Stop checks** (first match wins):
   - no OPEN gaps and every root gap (no parent) is RESOLVED → `ALL_GAPS_RESOLVED`
   - no OPEN gaps otherwise → `STAGNATION`
   - `steps_used + 1 ≥ max_steps` → `MAX_STEPS`
   - `tokens_used ≥ max_tokens` → `MAX_TOKENS`
   - the step's decision was STOP → its `stop_reason`
5. On stop: `status = STOPPED`, `stop_reason` set, verdict `should_continue = False`.

**Why `steps_used + 1`, not `steps_used` (ADR-009 §12).** `persist` increments
`budget.steps_used` *after* this node runs, so at the time this check runs,
`steps_used` still holds the count from *before* the current iteration; `+ 1` accounts
for the iteration in progress, which is about to be persisted as a completed step. This
makes the current iteration — a real action, not a wasted STOP — the one that ends the
task: with `max_steps = N`, exactly `N` real iterations run and `steps_used` ends at
`N`, never `N − 1` real iterations plus a STOP-only `N`-th (verified by
`tests/unit/test_state_invariants.py`'s `max_steps` case, TESTING.md §4). The
controller's own `steps_used >= max_steps` guard (§2 step 1) is unrelated to this fix
and stays unreachable in normal operation — see the note there.

With `ablation.gap_tracking` false, INIT creates exactly one gap (the goal), gap
resolution uses the evidence-sufficiency rule in EVIDENCE_MANAGER_SPEC §5, and
advancement never happens.

## 7. Fixed policy (ablation `adaptive_actions = false`)
For the current gap: first step DENSE if need is semantic, otherwise SPARSE_HYBRID; if
that query was already issued, the other mode; if both were issued, STOP(STAGNATION) for
this gap (monitor marks it EXHAUSTED and advances). No REUSE, RERANK_EXPAND or TRANSFORM.
Planning, reranking and evidence management stay on, isolating the value of adaptive
action selection.

## 8. Worked example (illustrative numbers)
State: need `mixed`; budget steps 2/8, retrieval 2/4, LLM 1/8 → pressure = 0.5.
Weights all 0.25; gain sub-weights gap 0.4 / ema 0.3 / type 0.3; cost units DENSE 1,
SPARSE_HYBRID 1, TRANSFORM 2 (max 2); latency DENSE 60, SPARSE_HYBRID 90, TRANSFORM 1500 ms.
gap_align = 0.8; ema(DENSE) = 0.2 (last dense step stagnated), ema(SPARSE_HYBRID) = prior
0.5; type_match mixed: DENSE 0.7, SPARSE_HYBRID 0.8; a similar earlier dense query gives
R(DENSE) = 0.9; no sparse queries yet → R(SPARSE_HYBRID) = 0; stagnation_count = 1 →
TRANSFORM eligible with G = 0.5 (patience 2). No cache hit; top relevances outside the band.

| Action | G | C | R | L | U |
|---|---|---|---|---|---|
| DENSE | 0.32+0.06+0.21 = 0.59 | 0.5×0.75 = 0.375 | 0.9 | 0.04 | 0.25×(0.59−0.375−0.9−0.04) = **−0.181** |
| SPARSE_HYBRID | 0.32+0.15+0.24 = 0.71 | 0.375 | 0.0 | 0.06 | 0.25×(0.71−0.375−0−0.06) = **0.069** |
| TRANSFORM | 0.5 | 1.0×0.75 = 0.75 | 0.0 | 1.0 | 0.25×(0.5−0.75−0−1.0) = **−0.3125** |

SPARSE_HYBRID wins (0.069 ≥ min_utility 0.05): a strategy switch rather than repeating
dense search or paying for a rewrite. This example is also a unit test
(`tests/unit/controller/test_worked_example.py`).

## 9. Calibration (dev split only)
Order matters, because later values depend on earlier ones:
1. Evidence thresholds (`evidence.relevance_threshold`, `evidence.novelty_threshold`,
   `evidence.gap_coverage_threshold`) — from dev traces, choose values maximising gold
   precision/recall of acceptance and gap resolution.
2. `cache.similarity_threshold` — from dev pairs of (queries, same-information?) labels.
3. `monitor.stagnation_threshold`, `controller.min_utility`, and weights α, β, γ, δ —
   grid search: stage 1 on the dev subset with step 0.2 over the weight simplex (56
   combinations) × 3 values of each of the two thresholds; stage 2 on the full dev split
   for the top 5 configurations.
   Objective: `J = CER − λ · mean_retrieval_calls / max_retrieval_calls`,
   λ = `controller.calibration_lambda`. Report the sensitivity table, not only the winner.
4. Write `docs/decisions/ADR-00X-calibration.md` with the chosen values and the objective
   table; update `configs/base.yaml`; mark keys Frozen; tag `v0.8-frozen-config`.
The LLM response cache makes repeated planning prompts free during the grid search.
