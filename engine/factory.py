"""The only place concrete component classes are chosen (CLAUDE.md rule 3; ARCHITECTURE §2).

Each builder takes the config and the components it depends on, and returns the module's
interface type, so callers (``engine/core.py``, ``eval/run.py``, the baselines, ``data/``)
never import a concrete class.

Selection (config key -> implementation):
- TaskStore         SQLite store at ``paths.state_db`` (``engine/state/store.py``)
- Embedder          sentence-transformers ``embedding.model_name`` @ ``embedding.revision``
- SparseEncoder     fastembed BM25 ``sparse.model_name`` @ ``sparse.revision``
- RetrieverAdapter  Qdrant at ``qdrant.url`` / ``qdrant.collection`` (dense, sparse, hybrid)
- IndexWriter       Qdrant collection creation + upsert (ingestion only)
- LLMClient         ``llm.<role>.provider``: ``anthropic`` | ``openai_compatible`` (``fake``: tests)
- Planner           bounded LLM planner (``ablation.planner`` false -> no planner calls)
- Cache             task-local SQLite cache in ``paths.state_db`` (``ablation.cache``)
- Controller        rule + utility policy (``ablation.adaptive_actions`` false -> fixed)
- Reranker          cross-encoder ``rerank.cross_encoder_model`` + state-aware scoring
- EvidenceManager   thresholds ``evidence.*`` (verifier not built: enabled -> ConfigError)
- TrajectoryMonitor stagnation/stop rules ``monitor.*``
- TraceWriter       JSONL traces + bundles under a run folder (OBSERVABILITY §4)
"""

from __future__ import annotations

from pathlib import Path

from engine.cache.interfaces import Cache
from engine.config import EngineConfig
from engine.controller.interfaces import Controller
from engine.embeddings.interfaces import Embedder, SparseEncoder
from engine.evidence.interfaces import EvidenceManager
from engine.llm.interfaces import LLMClient
from engine.monitor.interfaces import TraceWriter, TrajectoryMonitor
from engine.planner.interfaces import Planner
from engine.rerank.interfaces import Reranker
from engine.retrieval.interfaces import IndexWriter, RetrieverAdapter
from engine.state.interfaces import TaskStore
from engine.state.models import RunContext

__all__ = [
    "build_cache",
    "build_controller",
    "build_embedder",
    "build_evidence_manager",
    "build_index_writer",
    "build_llm_client",
    "build_monitor",
    "build_planner",
    "build_reranker",
    "build_retriever",
    "build_sparse_encoder",
    "build_task_store",
    "build_trace_writer",
    "prompt_hashes",
]


def build_task_store(cfg: EngineConfig) -> TaskStore:
    from engine.state.store import SqliteTaskStore

    return SqliteTaskStore(cfg)


def build_embedder(cfg: EngineConfig) -> Embedder:
    from engine.embeddings.sentence_transformer import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(cfg.embedding)


def build_sparse_encoder(cfg: EngineConfig) -> SparseEncoder:
    from engine.embeddings.bm25 import FastEmbedBM25Encoder

    return FastEmbedBM25Encoder(cfg.sparse, cfg.paths.model_cache_dir)


def build_retriever(
    cfg: EngineConfig, embedder: Embedder, sparse_encoder: SparseEncoder
) -> RetrieverAdapter:
    from engine.retrieval.qdrant_adapter import QdrantAdapter

    return QdrantAdapter(cfg, embedder, sparse_encoder)


def build_index_writer(cfg: EngineConfig) -> IndexWriter:
    from engine.retrieval.qdrant_index import QdrantIndexWriter

    return QdrantIndexWriter(cfg)


def build_llm_client(cfg: EngineConfig) -> LLMClient:
    from engine.llm.client import ProviderLLMClient

    return ProviderLLMClient(cfg)  # providers and API keys are resolved on first use


def build_planner(cfg: EngineConfig, llm: LLMClient) -> Planner:
    from engine.planner.llm_planner import LLMPlanner

    return LLMPlanner(cfg, llm)


def build_cache(cfg: EngineConfig, embedder: Embedder) -> Cache:
    from engine.cache.task_cache import TaskCache

    return TaskCache(cfg, embedder)


def build_controller(cfg: EngineConfig, embedder: Embedder) -> Controller:
    from engine.controller.policy import RuleUtilityController

    return RuleUtilityController(cfg, embedder)


def build_reranker(cfg: EngineConfig, embedder: Embedder) -> Reranker:
    from engine.rerank.cross_encoder import CrossEncoderScorer
    from engine.rerank.state_aware import StateAwareReranker

    return StateAwareReranker(cfg, embedder, CrossEncoderScorer(cfg.rerank))  # loads lazily


def build_evidence_manager(
    cfg: EngineConfig, embedder: Embedder, llm: LLMClient
) -> EvidenceManager:
    from engine.evidence.manager import ThresholdEvidenceManager

    return ThresholdEvidenceManager(cfg, embedder)  # llm: for the verifier, not built yet


def build_monitor(cfg: EngineConfig) -> TrajectoryMonitor:
    from engine.monitor.trajectory import RuleTrajectoryMonitor

    return RuleTrajectoryMonitor(cfg)


def prompt_hashes() -> dict[str, str]:
    """``{template: sha256}`` of every planner prompt template (run manifest, OBSERVABILITY §5)."""
    from engine.planner.prompting import prompt_hashes as planner_hashes

    return planner_hashes()


def build_trace_writer(cfg: EngineConfig, ctx: RunContext, root: Path) -> TraceWriter:
    """Trace writer for one run (``ctx``) writing under ``root`` (an eval sub-run folder, or
    the server's trace folder — the latter is decided with the ``/tasks`` routes)."""
    from engine.monitor.trace import JsonlTraceWriter

    return JsonlTraceWriter(root, ctx)
