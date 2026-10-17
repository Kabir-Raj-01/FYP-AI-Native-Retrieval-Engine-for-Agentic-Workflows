# ADR-015: Optional bearer token for the HTTP API (SRS NFR-S3)

> Status: Accepted (2026-10-17, Kabir for the team; Vishal review after the fact) · Date: 2026-10-17 · Author(s): Kabir · Reviewer: Vishal

## Context
SRS NFR-S3: "The service shall bind to localhost by default; remote exposure requires
explicit configuration and an optional bearer token." SECURITY §2 and `.env.example`
already described `ENGINE_API_TOKEN` as enabling such a check, but no code implemented it
(found in the FYP-I requirements audit, 2026-10-17). API_CONTRACT is a frozen contract, so
adding a status code needs this ADR.

## Decision
- When `ENGINE_API_TOKEN` is set and non-empty (environment, or `.env` loaded by
  `load_config`), every `/v1/*` request must carry `Authorization: Bearer <token>`.
  Otherwise the server answers **401 `UNAUTHORIZED`** in the standard error body, with
  `WWW-Authenticate: Bearer`, before the request reaches the engine.
- Unset or empty (the default): no check, exactly as before. The default bind stays
  `127.0.0.1`.
- `/docs`, `/redoc` and `/openapi.json` stay readable (no secrets; the browser cannot send
  the header).
- Implementation: one HTTP middleware in `engine/api/app.py`; constant-time comparison
  (`hmac.compare_digest`). The server reads the variable after `load_config`; an Engine
  passed into `create_app` (tests) uses only the explicit `api_token` argument, so unit
  tests never depend on a developer's shell.
- The token is **not** an `EngineConfig` key, so it never reaches `config.resolved.yaml`,
  the config hash, logs or traces (NFR-S2). A rejected request logs `api.unauthorized`
  with the path only.
- Client: `RetrievalClient(base_url, token=...)` sends the header; the 401 maps to the new
  `client.Unauthorized` exception.

## Alternatives considered
- FastAPI `HTTPBearer` dependency on each router — adds a security scheme to the OpenAPI
  document (snapshot change) and must be repeated per router; the middleware covers every
  current and future `/v1` route in one place.
- Put the token in `configs/base.yaml` — would commit a secret or leak it into run folders.
- Leave it unimplemented and document "SSH tunnel only" — contradicts NFR-S3.

## Consequences
API_CONTRACT gains the 401 row and a Behaviour bullet; SECURITY §2, CONFIG_SCHEMA §1,
OBSERVABILITY §3 (`api.unauthorized`), ARCHITECTURE change log and `.env.example` are
updated in the same change. Tests: `tests/unit/api/test_auth.py`.

## Reversal condition
Real multi-user deployment (out of scope for the FYP) would need proper authentication
(per-user keys, TLS); this check is only for a remote demo of a single-user service.
