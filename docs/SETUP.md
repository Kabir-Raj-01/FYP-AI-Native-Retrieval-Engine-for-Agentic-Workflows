# Development Setup (Windows)

> Status: Stable · Owner: Kabir · Last updated: 2026-10-05

The project is developed and run on **Windows only**. Every command in this repository is
written for **PowerShell** and is run from the repository root with the virtual
environment activated. §5 lists the command for every task.

## 1. Prerequisites
| Tool | Version | Notes |
|---|---|---|
| Windows | 10 or 11, 64-bit | |
| PowerShell | 7.x recommended (Windows PowerShell 5.1 also works) | all commands below |
| Git for Windows | recent | |
| Python | 3.11.x, 64-bit (python.org installer, with the `py` launcher) | `py -3.11 --version` |
| Qdrant | native Windows release (`qdrant-x86_64-pc-windows-msvc.zip`) | version pinned in LIBRARY_NOTES (§3) |
| Node.js | per Claude Code docs | only if using Claude Code |
| Disk | ~10 GB free | models + index + runs |
| RAM | 8 GB minimum, 16 GB comfortable | |
| GPU | optional (CUDA) | CPU is enough at MultiHop-RAG scale; slower ingestion |

Keep the clone in a short path such as `C:\fyp\retrieval-engine` (see the long-path
note in §7).

## 2. First-time setup
```powershell
git clone <repo-url> retrieval-engine
cd retrieval-engine
py -3.11 -m venv .venv               # python.org 3.11 (signed), see §7 if `py -3.11` fails
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock -e ".[dev]"   # exact versions (§6)
pre-commit install                   # hooks: .pre-commit-config.yaml
Copy-Item .env.example .env          # then fill in values (below)
```
Start Qdrant (§3), then run the unit tests — they must pass on a fresh clone:
```powershell
pytest -m "not integration and not llm and not e2e"
```

### `.env` values
`engine/config.py` loads `.env` with python-dotenv at start-up; a variable already set
in the environment wins over `.env` (CONFIG_SCHEMA §1).

| Variable | Needed for | Example |
|---|---|---|
| `LLM_API_KEY` | API-hosted LLM roles | your key (leave empty for local Ollama) |
| `QDRANT_URL` | non-default Qdrant | `http://127.0.0.1:6333` |
| `ENGINE_CONFIG` | server config file | `configs/base.yaml` |
| `QDRANT_VERSION` | the pinned Qdrant release tag | e.g. `v1.x.y`, see LIBRARY_NOTES |
| `ENGINE_API_TOKEN` | optional API auth | empty = disabled |
| `HF_HOME` | model download cache | `C:\Users\<you>\.cache\huggingface` |

Use `127.0.0.1` rather than `localhost` in URLs: on Windows `localhost` is tried as IPv6
(`::1`) first, which adds a connection delay to every call and distorts latency numbers.

### Local LLM (Ollama, the ADR-004 default — no API cost, no key)
Install once — the standalone zip into `tools\ollama\` (gitignored, ~1.5 GB), models in
`var\ollama` (~1.9 GB for `qwen2.5:3b`):
```powershell
New-Item -ItemType Directory -Force tools\ollama | Out-Null
curl.exe -L --fail -o tools\ollama\ollama.zip https://github.com/ollama/ollama/releases/download/v0.40.1/ollama-windows-amd64.zip
Expand-Archive tools\ollama\ollama.zip -DestinationPath tools\ollama -Force
Remove-Item tools\ollama\ollama.zip
```
Start (every session), then pull the model once:
```powershell
$env:OLLAMA_MODELS = "$PWD\var\ollama"; $env:OLLAMA_HOST = "127.0.0.1:11434"
Start-Process -FilePath .\tools\ollama\ollama.exe -ArgumentList serve -WindowStyle Minimized
.\tools\ollama\ollama.exe pull qwen2.5:3b              # once
python -m engine.llm.smoke --role planner              # must end with "ok: role=planner ..."
python -m engine.llm.smoke --role generator
```
Stop: `Stop-Process -Name ollama`. `configs/base.yaml` already points the planner and
generator at `http://127.0.0.1:11434/v1` with `provider: openai_compatible`. To use a hosted
model instead, change `llm.<role>.provider/model/base_url` and set `LLM_API_KEY` in `.env`
(a new model is an ADR-004 amendment).

## 3. Qdrant (native Windows binary)
Install once — download the pinned release into `tools\qdrant\` (gitignored):
```powershell
$QdrantVersion = "<pinned tag from LIBRARY_NOTES, e.g. v1.x.y>"
New-Item -ItemType Directory -Force tools\qdrant | Out-Null
Invoke-WebRequest -Uri "https://github.com/qdrant/qdrant/releases/download/$QdrantVersion/qdrant-x86_64-pc-windows-msvc.zip" -OutFile tools\qdrant\qdrant.zip
Expand-Archive tools\qdrant\qdrant.zip -DestinationPath tools\qdrant -Force
```
Start (every session) — REST on 127.0.0.1:6333, gRPC on 6334, data in `var\qdrant_storage`:
```powershell
$env:QDRANT__SERVICE__HOST = "127.0.0.1"
$env:QDRANT__STORAGE__STORAGE_PATH = "var\qdrant_storage"
$env:QDRANT__STORAGE__SNAPSHOTS_PATH = "var\qdrant_snapshots"
Start-Process -FilePath .\tools\qdrant\qdrant.exe -WindowStyle Minimized
Invoke-RestMethod http://127.0.0.1:6333       # prints the Qdrant version
```
Stop: `Stop-Process -Name qdrant`.

## 4. First data run
```powershell
python -m data.download --dataset multihop          # source + checksums from ADR-003
python -m data.ingest --dataset multihop            # chunk, embed, index, gold map, split
python scripts\check_gold_mapping.py multihop       # must report ≥ 95 % mapped
python -m eval.run --config configs\experiments\dense_oneshot_dev.yaml   # first baseline numbers
```

## 5. Command reference
| Task | Command |
|---|---|
| Install (editable, dev extras) | `pip install -r requirements.lock -e ".[dev]"` then `pre-commit install` |
| Lint / format | `ruff check .` / `ruff format .` |
| Type check | `mypy --strict engine` |
| Unit tests + coverage | `pytest -m "not integration and not llm and not e2e" --cov=engine --cov-report=term-missing` |
| Integration tests (Qdrant running) | `pytest -m integration` |
| Golden trajectories (fixture corpus, no Qdrant) | `pytest -m e2e` (rewrite: `$env:UPDATE_GOLDEN = "1"`, explain why in the PR) |
| Show a trajectory | `python scripts\show_trace.py <run_id> <task_id> [--budget B]` |
| All tests | `pytest` |
| Start / stop Qdrant | §3 / `Stop-Process -Name qdrant` |
| Ingest a dataset | `python -m data.ingest --dataset multihop` |
| Serve the API | `python -m uvicorn engine.api.app:app --host 127.0.0.1 --port 8000` |
| Run an experiment | `python -m eval.run --config configs\experiments\<name>.yaml` |
| Latency profile | `python scripts\latency_profile.py` (writes `configs\latency_profile.yaml`) |
| Answers for a finished run (generator, ADR-004) | `python -m eval.generator --run <run_id> [--max-tasks N]` |
| Report for a run / compare runs | `python -m eval.report --run <run_id> [--run <other_run_id> ...]` |
| Check an LLM role | `python -m engine.llm.smoke --role planner` (or `generator`) |
| Start / stop Ollama | §2 "Local LLM" / `Stop-Process -Name ollama` |
| Dependency audit | `pip-audit` |
| Clean caches | `Remove-Item -Recurse -Force .pytest_cache, .mypy_cache, .ruff_cache -ErrorAction SilentlyContinue` |
| Delete local state DBs | `Remove-Item -Confirm var\*.sqlite3` |

## 6. Pinning versions
`requirements.lock` was first generated on 2026-10-05 (python.org 3.11.9, Windows 11).
Everyone — and CI — installs with `pip install -r requirements.lock -e ".[dev]"`.
To regenerate it after a deliberate upgrade (keeps the file's `#` header; update its
date line by hand):
```powershell
python -c "import pathlib, subprocess, sys; p = pathlib.Path('requirements.lock'); head = [l for l in p.read_text(encoding='utf-8').splitlines() if l.startswith('#')] if p.exists() else []; body = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--exclude-editable'], capture_output=True, text=True, check=True).stdout; p.write_text('\n'.join(head + [body]).rstrip('\n') + '\n', encoding='utf-8', newline='\n')"
```
Why not `pip freeze | Out-File …`: `--exclude-editable` keeps out the `-e d:\…` line (an
absolute path that breaks on every other machine), and Windows PowerShell 5.1 adds a
byte-order mark both in `Out-File -Encoding utf8` **and** when piping text into a native
program, so Python runs pip itself (verified 2026-10-05: the command reproduces the
committed file byte-for-byte). Copy exact versions of the core
libraries into `LIBRARY_NOTES.md`. Upgrades happen in a dedicated PR that re-runs all
tests (`pytest`); keep the Smart App Control holds listed in the lock's header (§7).

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded … running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then reopen PowerShell |
| `connection refused` on 127.0.0.1:6333 | Qdrant not running (§3); check `Get-Process qdrant`; port in use? `Get-NetTCPConnection -LocalPort 6333` |
| Slow first request to the API, Qdrant or Ollama | a URL uses `localhost`; use `127.0.0.1` |
| `qdrant.exe` or torch fails with a missing `VCRUNTIME140.dll` | install the Microsoft Visual C++ Redistributable (x64) |
| "path too long" errors during install or model download | enable Win32 long paths (`LongPathsEnabled = 1`), run `git config --global core.longpaths true`, keep the repo in a short path |
| Hugging Face warns about symlinks | harmless (models are copied instead); enabling Windows Developer Mode removes the warning |
| Qdrant upsert error on point id | point ids must be UUIDs — use UUIDv5(chunk_id) (RETRIEVAL_SPEC §2) |
| Embedding dimension mismatch | collection built with another model — new collection name |
| Very slow ingestion on CPU | lower `embedding.batch_size`; ingest the dev subset first |
| LLM JSON parse failures | check the model supports JSON output; lower temperature to 0; see LIBRARY_NOTES |
| `database is locked` (SQLite) | WAL mode enabled? one TaskStore connection per worker thread |
| mypy errors from third-party libs | add module to `[[tool.mypy.overrides]] ignore_missing_imports` in pyproject |
| `DLL load failed … An Application Control policy has blocked this file` | Windows **Smart App Control** blocks some unsigned native modules. Do not turn Smart App Control off (it cannot be re-enabled without reinstalling Windows); use the fixes in the next rows. Details: LIBRARY_NOTES "Windows environment" |
| …while importing `_sqlite3` (pre-commit, `pytest --cov` or any SQLite store fails) | the venv was built from an unsigned Python (e.g. uv-managed). Install the signed python.org build — `winget install --id Python.Python.3.11 -e --scope user` — delete `.venv` and recreate it with `py -3.11` |
| …while importing `_uuid_utils` (`import langgraph` fails) | `uuid-utils` 0.17.1 is blocked; 0.12.0–0.17.0 and 1.0.0 load. `requirements.lock` holds 0.17.0; reinstall from the lock |
| …in `sklearn\…` or `pandas\_libs\…` (also breaks `transformers` / `sentence-transformers`) | scikit-learn 1.9.1 and pandas 3.0.5/3.0.6 are blocked; the lock holds scikit-learn 1.9.0 and pandas 3.0.3. Reinstall from the lock. For a future upgrade, import every extension module of the new release in a scratch venv first (LIBRARY_NOTES) |
| `py -3.11` → "No suitable Python runtime found" | only a non-python.org 3.11 is registered (`py -0p` shows e.g. `Astral/CPython3.11.16`); install python.org 3.11 as above |
| `py.exe` → "The file cannot be accessed by the system" | `py` resolved to the Python install manager's app alias (`…\WindowsApps\py.exe`). Call the interpreter directly, e.g. `& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv`, or disable the `py.exe` alias under Settings → Apps → Advanced app settings → App execution aliases |
