"""Test fakes implementing the engine interfaces (docs/TESTING.md §2)."""

from tests.fakes.clock import FakeClock
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient, LLMRequest
from tests.fakes.reranker import FakeReranker
from tests.fakes.retriever import CorpusChunk, FakeRetriever, SearchRequest

__all__ = [
    "CorpusChunk",
    "FakeClock",
    "FakeEmbedder",
    "FakeLLMClient",
    "FakeReranker",
    "FakeRetriever",
    "LLMRequest",
    "SearchRequest",
]
