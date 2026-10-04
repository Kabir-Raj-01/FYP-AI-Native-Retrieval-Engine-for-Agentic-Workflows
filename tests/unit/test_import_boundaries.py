"""Import-boundary rules checked with `ast` (TESTING §4.9; CLAUDE.md rules 3, 5; ARCHITECTURE §2).

Real now:
- provider SDKs (`anthropic`, `openai`) are imported only inside engine/llm;
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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".venv", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache", "tools", "var"}

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
}
CLIENT_ALLOWED = {"engine.state.models", "engine.api.schemas"}


@dataclass(frozen=True)
class Imported:
    path: str  # repo-relative, forward slashes
    module: str  # fully qualified module (for `from a import b`: "a.b" and "a")
    line: int


def python_files(root: Path = REPO_ROOT) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if not SKIP_DIRS.intersection(path.relative_to(root).parts):
            yield path


def imports_in(source: str, rel_path: str) -> Iterator[Imported]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Imported(rel_path, alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield Imported(rel_path, node.module, node.lineno)
            for alias in node.names:
                yield Imported(rel_path, f"{node.module}.{alias.name}", node.lineno)


def all_imports() -> list[Imported]:
    out: list[Imported] = []
    for path in python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        out.extend(imports_in(path.read_text(encoding="utf-8"), rel))
    return out


def _top(module: str) -> str:
    return module.split(".", 1)[0]


# ------------------------------------------------------------------------------ checkers
def sdk_violations(imports: list[Imported]) -> list[Imported]:
    return [
        i
        for i in imports
        if _top(i.module) in PROVIDER_SDKS and not i.path.startswith("engine/llm/")
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


def test_client_checker() -> None:
    assert not client_violations(_scan("from engine.state.models import Action", "client/m.py"))
    assert not client_violations(_scan("from engine.api.schemas import X", "client/m.py"))
    assert client_violations(_scan("from engine.core import Engine", "client/m.py"))
    assert client_violations(_scan("from engine.config import load_config", "client/m.py"))


# --------------------------------------------------------------------------------- TODO
@pytest.mark.skip(
    reason="TODO(Week 8): restrict sentence_transformers / fastembed / qdrant_client to their "
    "modules (CLAUDE.md rule 6) once the Embedder and QdrantAdapter land and the allowed "
    "modules are written into ARCHITECTURE §2"
)
def test_ml_and_backend_sdks_only_in_their_modules() -> None:  # pragma: no cover
    raise AssertionError


@pytest.mark.skip(
    reason="TODO(Weeks 8-9): STATE_SPEC §5 invariant 9 (no component keeps a reference to the "
    "state between calls) needs the graph nodes; define the static check with them"
)
def test_components_do_not_store_state_references() -> None:  # pragma: no cover
    raise AssertionError
