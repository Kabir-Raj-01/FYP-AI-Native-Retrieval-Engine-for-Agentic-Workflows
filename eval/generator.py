"""Answer generation and answer accuracy (EVAL_PROTOCOL §3-§4; ADR-003, ADR-004).

    python -m eval.generator --run <run_id> [--max-tasks N] [--config configs\\base.yaml]

Post hoc over a finished run: for every ``b<B>/bundles/<task>.json`` the role ``generator``
(same model, temperature 0 and prompt for every system, EVAL_PROTOCOL §1.5) answers from the
bundle's evidence in bundle order, each passage as ``[i] (title, date) text``, truncated to
``eval.generator_max_context_tokens`` (estimated as chars / 4, as the LLM client does).
Writes ``b<B>/answers.csv`` and ``b<B>/answers.json``. LLM settings come from ``--config``;
tasks and gold answers from the run's ``config.resolved.yaml``. The response cache makes a
re-run free, and identical bundles across systems cost one call.

Scoring: ``accuracy`` is the official MultiHop-RAG rule (ADR-003: the quoted text after
``The answer to the question is`` shares ≥ 1 lower-cased whitespace token with the gold
answer); ``exact_match`` is the stricter normalised exact match, reported alongside.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import string
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jinja2
import yaml

from engine import factory
from engine.config import EngineConfig, ExperimentConfig, load_config
from engine.errors import EngineError
from engine.llm.interfaces import LLMClient, Message
from engine.state.models import EvidenceBundle, RunContext
from engine.text import escape_untrusted
from eval.datasets.base import Task
from eval.datasets.multihop import load_tasks

__all__ = [
    "answer_messages",
    "exact_match",
    "extract_answer",
    "generate",
    "official_correct",
    "run_tasks",
]

PROMPT = Path(__file__).parent / "prompts" / "answer.j2"
_ANSWER = re.compile(r'The answer to the question is:?\s*"([^"]*)"', re.IGNORECASE)
_UNQUOTED = re.compile(r"The answer to the question is:?\s*([^\n\"]+)", re.IGNORECASE)
_PLACEHOLDER = "<answer>"  # qwen2.5:3b copied the old prompt's placeholder (2026-10-12)
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPT.parent),
    autoescape=False,
    undefined=jinja2.StrictUndefined,
)
_ENV.filters["untrusted"] = escape_untrusted


def prompt_hash() -> str:
    return hashlib.sha256(PROMPT.read_bytes()).hexdigest()


def answer_messages(question: str, bundle: EvidenceBundle, max_tokens: int) -> list[Message]:
    budget = max_tokens * 4
    passages: list[dict[str, str]] = []
    for e in bundle.evidence:
        if budget <= 0:
            break
        text = e.text[:budget]
        budget -= len(text)
        date = str(e.metadata.get("published_at") or "")[:10]
        passages.append({"title": str(e.metadata.get("title") or ""), "date": date, "text": text})
    module = _ENV.get_template(PROMPT.name).make_module(
        {"question": question, "passages": passages}
    )
    return [
        Message(role="system", content=str(getattr(module, "system")).strip()),  # noqa: B009
        Message(role="user", content=str(module).strip()),
    ]


def extract_answer(output: str) -> str:
    """The quoted text of the last answer line; else an unquoted ``…is: X`` line; else the
    whole output. A copied placeholder is no answer (scored wrong)."""
    found = _ANSWER.findall(output)
    if not found:
        found = [f.rstrip(".") for f in _UNQUOTED.findall(output)]
    pred = found[-1].strip() if found else output.strip()
    return "" if pred.lower() == _PLACEHOLDER else pred


def official_correct(prediction: str, gold: str) -> bool:
    return bool(set(prediction.lower().split()) & set(gold.lower().split()))


def _normalise(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def exact_match(prediction: str, gold: str) -> bool:
    return _normalise(prediction) == _normalise(gold)


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def run_tasks(run_dir: Path, cfg: EngineConfig) -> list[Task]:
    """The tasks of a finished run: its ``experiment`` section (``config.resolved.yaml``),
    the data under the current ``cfg``."""
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    if not resolved.get("experiment"):
        raise ValueError(f"{run_dir}: not an experiment run")
    exp = ExperimentConfig.model_validate(resolved["experiment"])
    return load_tasks(cfg, exp.split, exp.subset)


def generate(
    run_dir: Path, tasks: Sequence[Task], cfg: EngineConfig, llm: LLMClient
) -> dict[str, Any]:
    """Answer every bundle of every ``b<B>`` sub-run; returns ``{"b<B>": summary}``."""
    out: dict[str, Any] = {}
    for sub in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("b")):
        rows: list[dict[str, Any]] = []
        traces = factory.build_trace_writer(cfg, RunContext(run_id=run_dir.name, system="-"), sub)
        for task in tasks:
            bundle = traces.read_bundle(task.task_id)
            if bundle is None or task.answer is None:
                continue
            messages = answer_messages(task.question, bundle, cfg.eval.generator_max_context_tokens)
            reply = llm.text("generator", messages)
            pred = extract_answer(reply.value)
            rows.append(
                {
                    "task_id": task.task_id,
                    "question_type": task.question_type,
                    "gold": task.answer,
                    "prediction": pred,
                    "correct": int(official_correct(pred, task.answer)),
                    "exact_match": int(exact_match(pred, task.answer)),
                }
            )
        with (sub / "answers.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task_id",
                    "question_type",
                    "gold",
                    "prediction",
                    "correct",
                    "exact_match",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        types = sorted({str(r["question_type"]) for r in rows})
        summary = {
            "model": cfg.llm.generator.model,
            "prompt_hash": prompt_hash(),
            "tasks": len(rows),
            "accuracy": _mean([r["correct"] for r in rows]),
            "exact_match": _mean([r["exact_match"] for r in rows]),
            "accuracy_by_type": {
                t: _mean([r["correct"] for r in rows if str(r["question_type"]) == t])
                for t in types
            },
        }
        (sub / "answers.json").write_bytes((json.dumps(summary, indent=1) + "\n").encode())
        out[sub.name] = summary
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.generator", description=__doc__)
    parser.add_argument("--run", required=True, help="run id (folder under paths.runs_dir)")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        cfg.require_llm_model("generator")
        run_dir = cfg.paths.runs_dir / args.run
        tasks = run_tasks(run_dir, cfg)[: args.max_tasks]
        result = generate(run_dir, tasks, cfg, factory.build_llm_client(cfg))
    except (EngineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
