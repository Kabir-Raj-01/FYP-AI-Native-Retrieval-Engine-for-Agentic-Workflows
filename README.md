# AI-Native Retrieval Engine for Agentic Workflows

Final Year Project — BS(CS), FAST School of Computing, FAST-NUCES Karachi (2026–27).
Kabir Raj (23K-0702) · Samay Kumar (23K-0620) · Vishal Kothari (23K-0040).

A standalone, pluggable retrieval-control middleware. It augments an existing RAG or
agent application — without replacing its retriever or generator — by maintaining
task-scoped retrieval state and deciding, step by step, whether to reuse, retrieve
(dense or sparse/hybrid), rewrite, rerank/expand, or stop. A controller-OFF bypass path
keeps normal one-shot retrieval available for direct before/after comparison.

- **Setup:** [`docs/SETUP.md`](docs/SETUP.md)
- **All documentation:** [`docs/README.md`](docs/README.md)
- **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Current work:** [`docs/TASKS.md`](docs/TASKS.md)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

Quick start on Windows (PowerShell, after setup — see `docs/SETUP.md` §3 for starting Qdrant):
```powershell
python -m data.ingest --dataset multihop
python -m uvicorn engine.api.app:app --host 127.0.0.1 --port 8000   # API at http://127.0.0.1:8000/docs
python -m eval.run --config configs\experiments\dense_oneshot_dev.yaml   # first baseline run (second terminal)
```
