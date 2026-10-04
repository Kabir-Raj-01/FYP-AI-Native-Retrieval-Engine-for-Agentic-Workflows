## What & why
<!-- one paragraph -->

**TASKS.md item / issue:**
**Governing doc(s):**

## Checklist
- [ ] Tests added/updated; `ruff check .`, `mypy --strict engine` and `pytest -m "not integration and not llm and not e2e"` pass locally (output checked)
- [ ] Integration tests run if retrieval/Qdrant code changed (`pytest -m integration` with local `qdrant.exe` running)
- [ ] No hard-coded thresholds/weights/models/paths — config keys added to base.yaml + CONFIG_SCHEMA.md + EngineConfig
- [ ] Field-ownership matrix and import-boundary rules respected
- [ ] Docs updated (spec, TASKS.md status, LIBRARY_NOTES.md, ARCHITECTURE change log)
- [ ] ADR included if a contract or Frozen key changed
- [ ] I can explain every line of this diff

## How I verified it
<!-- commands run, outputs, trace excerpts -->
