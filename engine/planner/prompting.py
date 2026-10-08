"""Planner prompt rendering (PLANNER_SPEC §3, §7; SECURITY §3).

Templates live in ``engine/planner/prompts/`` (``init``/``rewrite``/``split``.j2 plus the
shared ``_rules``/``_evidence`` parts). Each template sets ``system`` (role, rules, the
untrusted-data warning; the LLM client appends the JSON schema) and renders the user
message. Every inserted value goes through ``untrusted`` (``&``, ``<``, ``>`` escaped), and
evidence summaries appear only inside the ``<evidence untrusted="true">`` block of the user
message — never in the system message. ``prompt_hashes()`` feeds the run manifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import jinja2

from engine.llm.interfaces import Message
from engine.state.models import PlanMode
from engine.text import escape_untrusted

__all__ = ["PROMPTS_DIR", "escape_untrusted", "prompt_hashes", "render"]

PROMPTS_DIR = Path(__file__).parent / "prompts"
_TEMPLATES = {PlanMode.INIT: "init.j2", PlanMode.REWRITE: "rewrite.j2", PlanMode.SPLIT: "split.j2"}


_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPTS_DIR),
    autoescape=False,  # escaping is explicit, via the `untrusted` filter
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=False,
)
_ENV.filters["untrusted"] = escape_untrusted


def render(mode: PlanMode, **values: Any) -> list[Message]:
    """System + user messages for ``mode``; ``values`` are the template variables."""
    module = _ENV.get_template(_TEMPLATES[mode]).make_module(values)
    system = str(getattr(module, "system")).strip()  # noqa: B009 — a template-exported name
    return [
        Message(role="system", content=system),
        Message(role="user", content=str(module).strip()),
    ]


def prompt_hashes() -> dict[str, str]:
    """``{"planner/<template>": sha256}`` of every template file (OBSERVABILITY §5)."""
    return {
        f"planner/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()  # LF: .gitattributes
        for p in sorted(PROMPTS_DIR.glob("*.j2"))
    }
