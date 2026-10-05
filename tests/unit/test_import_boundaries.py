"""Import-boundary rules checked with `ast` (TESTING §4.9; CLAUDE.md rules 3, 5; ARCHITECTURE §2).

Real now:
- provider SDKs (`anthropic`, `openai`) are imported only inside engine/llm (and its own unit
  tests, which build SDK exceptions to check the error mapping);
- nothing in examples/ imports `engine`;
- concrete component modules (`engine.<component>.<anything but interfaces>`) are imported
  only by engine/factory.py or from inside their own package (tests are exempt: unit tests
  exercise implementations directly);
- client/ imports from the engine only `engine.state.models` and `engine.api.schemas`.
Most target folders are still empty, so each checker is also run on synthetic violating
code below to prove it detects what it claims to.
"""

from __future__ import annotations

import ast
import functools
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {
    ".venv",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "tools",
    "var",
    "raw",
    "processed",
    "htmlcov",
    "runs",
}

PROVIDER_SDKS = {"anthropic", "openai"}
COMPONENT_PACKAGES = {
    "planner",
    "cache",
    "controller",
    "retrieval",
    "rerank",
    "evidence",
    "monitor",
    "embeddings",
    "llm",
    "state",
    "graph",
    "api",
}
# Shared, non-component modules every module may import (ARCHITECTURE §2 dependency rule).
SHARED_MODULES = {
    "engine.config",
    "engine.errors",
    "engine.state.models",
    "engine.state.serialization",
    "engine.state.invariants",
    "engine.monitor.logging",  # the shared event logger (OBSERVABILITY §3), not a component
    "engine.state.sqlite",  # shared SQLite schema + cache-table rules (TaskStore and Cache)
    "engine.api.schemas",  # the HTTP request/response models (CLAUDE.md rule 2); client uses them
}
CLIENT_ALLOWED = {"engine.state.models", "engine.api.schemas"}
# The Engine facade runs the controlled path itself (ARCHITECTURE §1: core sits directly over
# engine/graph, which has no interface and is not chosen from config) — the only exemption.
FACADE_PACKAGES = {("engine/core.py", "graph")}


@dataclass(frozen=True)
class Imported:
    path: str  # repo-relative, forward slashes
    module: str  # fully qualified module (for `from a import b`: "a.b" and "a")
    line: int


def python_files(root: Path = REPO_ROOT) -> Iterator[Path]:
    # os.walk with in-place pruning: never descends into .venv, data stores, caches.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def imports_in(source: str, rel_path: str) -> Iterator[Imported]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Imported(rel_path, alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield Imported(rel_path, node.module, node.lineno)
            for alias in node.names:
                yield Imported(rel_path, f"{node.module}.{alias.name}", node.lineno)


@functools.cache
def _scan_repository() -> tuple[Imported, ...]:
    out: list[Imported] = []
    for path in python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        out.extend(imports_in(path.read_text(encoding="utf-8"), rel))
    return tuple(out)


def all_imports() -> list[Imported]:
    return list(_scan_repository())  # one scan per test session


def _top(module: str) -> str:
    return module.split(".", 1)[0]


# ------------------------------------------------------------------------------ checkers
def sdk_violations(imports: list[Imported]) -> list[Imported]:
    return [
        i
        for i in imports
        if _top(i.module) in PROVIDER_SDKS
        # engine/llm itself, and its own unit tests (they build SDK exceptions to check the
        # providers' error mapping)
        and not i.path.startswith(("engine/llm/", "tests/unit/llm/"))
    ]


def examples_violations(imports: list[Imported]) -> list[Imported]:
    return [i for i in imports if i.path.startswith("examples/") and _top(i.module) == "engine"]


def _concrete_component_module(module: str) -> str | None:
    """'engine.cache.task_cache[.X]' -> 'cache'; None for interfaces/shared/non-engine."""
    parts = module.split(".")
    if len(parts) < 3 or parts[0] != "engine" or parts[1] not in COMPONENT_PACKAGES:
        return None
    if parts[2] == "interfaces" or ".".join(parts[:3]) in SHARED_MODULES:
        return None
    return parts[1]


def factory_violations(imports: list[Imported]) -> list[Imported]:
    out = []
    for i in imports:
        package = _concrete_component_module(i.module)
        if package is None or i.path.startswith("tests/") or i.path == "engine/factory.py":
            continue
        if i.path.startswith(f"engine/{package}/"):
            continue  # a package may use its own implementation modules
        if (i.path, package) in FACADE_PACKAGES:
            continue
        out.append(i)
    return out


def _client_may_import(module: str) -> bool:
    inside_allowed = any(module == a or module.startswith(f"{a}.") for a in CLIENT_ALLOWED)
    # `from engine.state.models import X` also yields its parent "engine.state.models";
    # pure parents of an allowed module (e.g. "engine.state") are not violations either.
    parent_of_allowed = any(a.startswith(f"{module}.") for a in CLIENT_ALLOWED)
    return inside_allowed or parent_of_allowed


def client_violations(imports: list[Imported]) -> list[Imported]:
    return [
        i
        for i in imports
        if i.path.startswith("client/")
        and _top(i.module) == "engine"
        and not _client_may_import(i.module)
    ]


# ------------------------------------------------------------------------ repository checks
def test_the_scan_sees_the_engine_sources() -> None:
    paths = {i.path for i in all_imports()}
    assert "engine/factory.py" in paths
    assert "engine/state/models.py" in paths


def test_provider_sdks_only_in_engine_llm() -> None:
    assert sdk_violations(all_imports()) == []


def test_examples_never_import_engine() -> None:
    assert examples_violations(all_imports()) == []


def test_concrete_components_only_imported_by_factory() -> None:
    assert factory_violations(all_imports()) == []


def test_client_imports_only_shared_models_from_engine() -> None:
    assert client_violations(all_imports()) == []


# ------------------------------------------------------------- checker self-tests (synthetic)
def _scan(source: str, path: str) -> list[Imported]:
    return list(imports_in(source, path))


@pytest.mark.parametrize("source", ["import anthropic", "from openai import OpenAI"])
def test_sdk_checker_flags_imports_outside_engine_llm(source: str) -> None:
    assert sdk_violations(_scan(source, "engine/planner/planner.py"))
    assert sdk_violations(_scan(source, "eval/generator.py"))
    assert not sdk_violations(_scan(source, "engine/llm/providers/anthropic.py"))
    assert not sdk_violations(_scan(source, "tests/unit/llm/test_providers.py"))
    assert sdk_violations(_scan(source, "tests/unit/planner/test_planner.py"))


@pytest.mark.parametrize(
    "source", ["import engine", "from engine.state.models import Action", "import engine.core"]
)
def test_examples_checker_flags_engine_imports(source: str) -> None:
    assert examples_violations(_scan(source, "examples/langgraph_agent/agent.py"))
    assert not examples_violations(_scan("from client import RetrievalClient", "examples/a.py"))


@pytest.mark.parametrize(
    "source",
    [
        "from engine.cache.task_cache import TaskCache",
        "from engine.cache import task_cache",
        "import engine.state.store",
        "from engine.retrieval.qdrant_adapter import QdrantAdapter",
    ],
)
def test_factory_checker_flags_concrete_imports(source: str) -> None:
    assert factory_violations(_scan(source, "engine/core.py"))
    assert factory_violations(_scan(source, "baselines/dense_oneshot.py"))
    assert not factory_violations(_scan(source, "engine/factory.py"))
    assert not factory_violations(_scan(source, "tests/unit/cache/test_task_cache.py"))


@pytest.mark.parametrize(
    "source",
    [
        "from engine.cache.interfaces import Cache",
        "from engine.state.models import RetrievalState",
        "from engine.state.serialization import dump_with_vectors",
        "from engine.config import EngineConfig",
        "from engine.errors import BackendError",
    ],
)
def test_factory_checker_allows_interfaces_and_shared_modules(source: str) -> None:
    assert not factory_violations(_scan(source, "engine/graph/nodes.py"))


def test_factory_checker_allows_a_package_to_use_its_own_modules() -> None:
    source = "from engine.cache.normalize import normalize"
    assert not factory_violations(_scan(source, "engine/cache/task_cache.py"))
    assert factory_violations(_scan(source, "engine/planner/validator.py"))


def test_only_the_facade_may_import_the_graph() -> None:
    source = "from engine.graph.graph import run_iteration"
    assert not factory_violations(_scan(source, "engine/core.py"))
    assert factory_violations(_scan(source, "baselines/dense_oneshot.py"))
    assert factory_violations(_scan(source, "engine/api/routers/direct.py"))


def test_client_checker() -> None:
    assert not client_violations(_scan("from engine.state.models import Action", "client/m.py"))
    assert not client_violations(_scan("from engine.api.schemas import X", "client/m.py"))
    assert client_violations(_scan("from engine.core import Engine", "client/m.py"))
    assert client_violations(_scan("from engine.config import load_config", "client/m.py"))


# ---------------------------------------------------- ML / backend libraries (CLAUDE.md 6)
# Library -> path prefixes allowed to import it (ARCHITECTURE §2). Tests are exempt.
LIBRARY_HOMES = {
    "sentence_transformers": ("engine/embeddings/", "engine/rerank/"),
    "fastembed": ("engine/embeddings/",),
    "huggingface_hub": ("engine/embeddings/",),
    "transformers": ("engine/embeddings/", "engine/rerank/", "data/"),  # data/: chunk tokenizer
    "qdrant_client": ("engine/retrieval/",),
}


def library_violations(imports: list[Imported]) -> list[Imported]:
    return [
        i
        for i in imports
        if _top(i.module) in LIBRARY_HOMES
        and not i.path.startswith("tests/")
        and not i.path.startswith(LIBRARY_HOMES[_top(i.module)])
    ]


def test_ml_and_backend_libraries_only_in_their_modules() -> None:
    assert library_violations(all_imports()) == []


@pytest.mark.parametrize(
    ("source", "path", "violates"),
    [
        ("from qdrant_client import QdrantClient", "data/index.py", True),
        ("from qdrant_client import QdrantClient", "engine/retrieval/qdrant_adapter.py", False),
        ("import sentence_transformers", "engine/core.py", True),
        ("from sentence_transformers import CrossEncoder", "engine/rerank/cross.py", False),
        ("from fastembed import SparseTextEmbedding", "data/ingest.py", True),
        ("from transformers import AutoTokenizer", "data/ingest.py", False),
        ("from qdrant_client import models", "tests/integration/test_x.py", False),
    ],
)
def test_library_checker(source: str, path: str, violates: bool) -> None:
    assert bool(library_violations(_scan(source, path))) is violates


# ----------------------------------------------- STATE_SPEC §5 invariant 9 (static part)
STATEFUL_PACKAGES = ("planner", "cache", "controller", "rerank", "evidence", "monitor", "graph")


def state_reference_violations(source: str, rel_path: str) -> list[str]:
    """``self.<attr> = <anything mentioning a parameter named state/gs>`` inside a method:
    a component keeping (part of) the task state between calls."""
    out: list[str] = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} & {"state", "gs"}
        if not params:
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            on_self = any(
                isinstance(tg, ast.Attribute)
                and isinstance(tg.value, ast.Name)
                and tg.value.id == "self"
                for tg in targets
            )
            value = node.value
            uses = value is not None and any(
                isinstance(n, ast.Name) and n.id in params for n in ast.walk(value)
            )
            if on_self and uses:
                out.append(f"{rel_path}:{node.lineno}")
    return out


def test_components_do_not_store_state_references() -> None:
    found: list[str] = []
    for package in STATEFUL_PACKAGES:
        for path in (REPO_ROOT / "engine" / package).glob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            found += state_reference_violations(path.read_text(encoding="utf-8"), rel)
    assert found == []


@pytest.mark.parametrize(
    ("source", "violates"),
    [
        ("def f(self, state):\n    self.last = state\n", True),
        ("def f(self, state):\n    self.n = len(state.gaps)\n", True),
        ("def f(self, gs, ctx):\n    self.s = gs.state\n", True),
        ("def f(self, state):\n    x = state.gaps\n    return x\n", False),
        ("def f(self, cfg):\n    self.cfg = cfg\n", False),
    ],
)
def test_state_reference_checker(source: str, violates: bool) -> None:
    assert bool(state_reference_violations(source, "engine/x.py")) is violates
