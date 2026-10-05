"""Test fakes implementing the engine interfaces (docs/TESTING.md §2)."""

from tests.fakes.clock import FakeClock
from tests.fakes.controlled import FakeController, FakeEvidenceManager, FakeMonitor, FakePlanner
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient, LLMRequest
from tests.fakes.reranker import FakeReranker
from tests.fakes.retriever import CorpusChunk, FakeRetriever, SearchRequest
from tests.fakes.sparse import FakeSparseEncoder

__all__ = [
    "CorpusChunk",
    "FakeClock",
    "FakeController",
    "FakeEmbedder",
    "FakeEvidenceManager",
    "FakeLLMClient",
    "FakeMonitor",
    "FakePlanner",
    "FakeReranker",
    "FakeRetriever",
    "FakeSparseEncoder",
    "LLMRequest",
    "SearchRequest",
]
