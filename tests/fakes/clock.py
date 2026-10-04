"""FakeClock (TESTING §2): controllable wall-clock and monotonic time.

No Clock interface exists in the docs yet (Week 7 plan, S10); this fake defines the shape
one is expected to have — ``now()`` (UTC-aware) for timestamps and ``monotonic_ms()`` for
latency fields — and will implement that interface once it is added to ARCHITECTURE §2.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

DEFAULT_START = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, start: datetime = DEFAULT_START, monotonic_ms: float = 0.0) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs a timezone-aware start time")
        self._now = start.astimezone(UTC)
        self._monotonic_ms = monotonic_ms

    def now(self) -> datetime:
        return self._now

    def monotonic_ms(self) -> float:
        return self._monotonic_ms

    def advance(self, ms: float) -> None:
        if ms < 0:
            raise ValueError("time cannot go backwards")
        self._now += timedelta(milliseconds=ms)
        self._monotonic_ms += ms
