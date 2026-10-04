"""The only place concrete component classes are chosen (CLAUDE.md rule 3; ARCHITECTURE §2).

Week 7 stub: it fixes the selection structure; every builder raises NotImplementedError
until its module is implemented (docs/TASKS.md Weeks 8-13). Each builder takes the config
and the components it depends on, and returns the module's interface type, so callers
(``engine/core.py``, ``eval/run.py``, the baselines) never import a concrete class.

Selection (config key -> implementation):
- TaskStore        SQLite store at ``paths.state_db`` (``engine/state/store.py``)
- Embedder         sentence-transformers ``embedding.model_name`` on ``embedding.device``
- RetrieverAdapter Qdrant at ``qdrant.url`` / ``qdrant.collection`` (dense, sparse, hybrid)
- LLMClient        ``llm.<role>.provider``: ``anthropic`` | ``openai_compatible`` | ``fake``
- Planner          bounded LLM planner (``ablation.planner`` false -> no planner calls)
- Cache            task-local SQLite cache in ``paths.state_db`` (``ablation.cache``)
- Controller       rule + utility policy (``ablation.adaptive_actions`` false -> fixed policy)
- Reranker         cross-encoder ``rerank.cross_encoder_model`` + state-aware scoring
- EvidenceManager  thresholds ``evidence.*``; verifier when ``llm.verifier.enabled``
- TrajectoryMonitor stagnation/stop rules ``monitor.*``
"""

from __future__ import annotations

from engine.cache.interfaces import Cache
from engine.config import EngineConfig
from engine.controller.interfaces import Controller
from engine.embeddings.interfaces import Embedder
from engine.evidence.interfaces import EvidenceManager
from engine.llm.interfaces import LLMClient
from engine.monitor.interfaces import TrajectoryMonitor
from engine.planner.interfaces import Planner
from engine.rerank.interfaces import Reranker
from engine.retrieval.interfaces import RetrieverAdapter
from engine.state.interfaces import TaskStore

__all__ = [
    "build_cache",
    "build_controller",
    "build_embedder",
    "build_evidence_manager",
    "build_llm_client",
    "build_monitor",
    "build_planner",
    "build_reranker",
    "build_retriever",
    "build_task_store",
]


def _not_built(component: str) -> NotImplementedError:
    return NotImplementedError(f"{component} is not implemented yet (see docs/TASKS.md)")


def build_task_store(cfg: EngineConfig) -> TaskStore:
    raise _not_built("TaskStore")


def build_embedder(cfg: EngineConfig) -> Embedder:
    raise _not_built("Embedder")


def build_retriever(cfg: EngineConfig, embedder: Embedder) -> RetrieverAdapter:
    raise _not_built("RetrieverAdapter")


def build_llm_client(cfg: EngineConfig) -> LLMClient:
    raise _not_built("LLMClient")


def build_planner(cfg: EngineConfig, llm: LLMClient) -> Planner:
    raise _not_built("Planner")


def build_cache(cfg: EngineConfig, embedder: Embedder) -> Cache:
    raise _not_built("Cache")


def build_controller(cfg: EngineConfig, embedder: Embedder) -> Controller:
    raise _not_built("Controller")


def build_reranker(cfg: EngineConfig, embedder: Embedder) -> Reranker:
    raise _not_built("Reranker")


def build_evidence_manager(
    cfg: EngineConfig, embedder: Embedder, llm: LLMClient
) -> EvidenceManager:
    raise _not_built("EvidenceManager")


def build_monitor(cfg: EngineConfig) -> TrajectoryMonitor:
    raise _not_built("TrajectoryMonitor")
