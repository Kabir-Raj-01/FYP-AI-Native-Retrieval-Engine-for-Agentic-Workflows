"""FYP-I demo: one question, controller-OFF vs controller-ON, through the HTTP API.

    python -m examples.demo_fyp1 [--url http://127.0.0.1:8000] [--question "..."]
                                 [--max-retrieval-calls 4]

Needs the API server (``python -m uvicorn engine.api.app:app --host 127.0.0.1 --port 8000``),
Qdrant and Ollama running (SETUP §2-§3). Imports only ``client`` — exactly what a host
application would do (CLAUDE.md: examples/ never import the engine). The default question
is from the dev subset (multihop:0082), never from the test split.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from client import RetrievalClient, RetrievalClientError
from client.models import BudgetRequest, EvidenceItem, RetrievalMode

QUESTION = (
    "Does the Polygon article suggest that Martin Scorsese has less autonomy in his "
    "filmmaking choices compared to earlier in his career, while The Independent - Life and "
    "Style article indicates that he has previously engaged with newer media platforms like "
    "TikTok through his daughter's videos?"
)


def _title(e: EvidenceItem) -> str:
    source = e.metadata.get("source") or "?"
    return f"{source}: {str(e.metadata.get('title') or e.chunk_id)[:70]}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.demo_fyp1", description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--max-retrieval-calls", type=int, default=4)
    args = parser.parse_args(argv)
    client = RetrievalClient(args.url)
    try:
        health = client.health()
        print(f"server {health.version} · status {health.status} · qdrant {health.qdrant}\n")
        print(f"Question: {args.question}\n")

        print("== Controller-OFF: one dense retrieval + relevance rerank ==")
        start = time.perf_counter()
        direct = client.retrieve_direct(args.question, RetrievalMode.DENSE, rerank=True)
        took = time.perf_counter() - start
        for c, rel in list(zip(direct.candidates, direct.relevances or [], strict=False))[:5]:
            print(f"  {rel:.3f}  {c.metadata.get('source')}: {str(c.metadata.get('title'))[:70]}")
        print(f"  ({len(direct.candidates)} candidates, {took:.1f} s)\n")

        print("== Controller-ON: stateful loop, one step at a time ==")
        budget = BudgetRequest(max_retrieval_calls=args.max_retrieval_calls)
        task = client.create_task(args.question, budget=budget)
        tid = task.task_id
        done = False
        while not done:
            start = time.perf_counter()
            r = client.step(tid)
            took = time.perf_counter() - start
            gaps = "; ".join(g.description[:50] for g in r.open_gaps) or "-"
            print(
                f"  step {r.step}: {r.action.value:<13} +{len(r.new_evidence)} evidence  "
                f"retrievals {r.budget.retrieval_calls_used}/{r.budget.max_retrieval_calls}  "
                f"({took:.1f} s)  open gaps: {gaps}"
            )
            done = r.done
        bundle = client.result(tid)
        print(f"\n  stop reason: {bundle.stop_reason.value}")
        print(f"  gaps: {[(g.description[:60], g.status.value) for g in bundle.gaps]}")
        print("  evidence bundle (the host app generates the answer from this):")
        for e in bundle.evidence[:8]:
            print(f"    final {e.final_score:.3f}  rel {e.relevance:.3f}  {_title(e)}")

        print("\n== Trace (why each action was chosen) ==")
        for s in client.trace(tid):
            scores = {a.value: round(v.U, 3) for a, v in s.utility_scores.items()}
            print(f"  {s.step}: {s.action.value:<13} rule={s.rule:<15} U={scores}")
    except RetrievalClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
