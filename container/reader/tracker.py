"""Readings -> speaker timeline in stream seconds.

Rules (after Neleceni's state machine):
- a speaker counts as changed after `confirm` readings of the same new name (MP or not);
  the new segment is back-dated to the first of those readings;
- readings with no usable name change nothing, because the speech usually goes on after
  the caption fades;
- a non-MP speaker (minister, guest) closes the current MP segment without opening one;
- an open segment is closed after `max_speech` seconds as a safety net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .match import Observation, similar_lines


SILENCE_CUT = 5.0  # a pause this long stops counting as talking


@dataclass
class Segment:
    start: float
    end: float | None
    mp: int
    talk: float = 0.0  # seconds with voice, not counting pauses of SILENCE_CUT or more
    quiet: float = 0.0  # current run of silence
    heard: bool = False  # any audio at all for this segment (else talk time is unknown)

    def json(self) -> dict:
        return {"from": round(self.start, 1), "to": None if self.end is None else round(self.end, 1), "mp": self.mp,
                "talk": round(self.talk, 1) if self.heard else None}


@dataclass
class Tracker:
    confirm: int = 3
    max_speech: float = 1800.0
    keep: int = 600
    video: str | None = None
    segments: list[Segment] = field(default_factory=list)
    current: tuple | None = None  # ("mp", id) | ("other", line) | None
    _pending: tuple | None = None
    _pending_first: float = 0.0
    _pending_hits: int = 0
    _breaks: int = 0
    first_break: float | None = None  # first confirmed break before anyone spoke (joins stretches of history)
    _break_t: float = 0.0

    @property
    def open(self) -> Segment | None:
        return self.segments[-1] if self.segments and self.segments[-1].end is None else None

    def reset(self, video: str | None) -> None:
        self.video, self.segments, self.current = video, [], None
        self._pending, self._pending_hits = None, 0

    @staticmethod
    def _same(a: tuple | None, b: tuple | None) -> bool:
        if a is None or b is None or a[0] != b[0]:
            return False
        return a[1] == b[1] if a[0] == "mp" else similar_lines(a[1], b[1])

    def audio(self, t: float, dt: float, voiced: bool) -> None:
        """Count talk time for whoever is speaking: pauses shorter than SILENCE_CUT count, longer ones don't."""
        cur = self.open
        if cur is None or t < cur.start:
            return
        cur.heard = True
        if voiced:
            cur.talk += dt + (cur.quiet if cur.quiet < SILENCE_CUT else 0.0)
            cur.quiet = 0.0
        else:
            cur.quiet += dt

    def observe(self, t: float, obs: Observation) -> bool:
        """Feed one reading at stream time t. Returns True when the timeline changed."""
        changed = self.tick(t)
        if obs.kind == "break":  # the break screen, seen twice in a row: nobody is speaking
            self._pending, self._pending_hits = None, 0
            self._breaks += 1
            if self._breaks == 1:
                self._break_t = t
                return changed
            self.current = None
            if not self.segments and self.first_break is None:
                self.first_break = self._break_t
            if self.open is not None:
                self.open.end = max(self.open.start, self._break_t)
                return True
            return changed
        self._breaks = 0
        if obs.kind == "none":
            return changed
        ident = ("mp", obs.mp) if obs.kind == "mp" else ("other", obs.line)
        if self._same(ident, self.current):
            self._pending, self._pending_hits = None, 0
            return changed
        if self._same(ident, self._pending):
            self._pending_hits += 1
        else:
            self._pending, self._pending_first, self._pending_hits = ident, t, 1
        if self._pending_hits >= self.confirm:
            self._switch(self._pending_first, ident)
            self._pending, self._pending_hits = None, 0
            return True
        return changed

    def _switch(self, t: float, ident: tuple) -> None:
        cur = self.open
        if cur is not None:
            cur.end = max(cur.start, t)
        self.current = ident
        if ident[0] == "mp":
            self.segments.append(Segment(t, None, ident[1]))
            del self.segments[: -self.keep]

    def tick(self, t: float) -> bool:
        cur = self.open
        if cur is not None and t - cur.start > self.max_speech:
            cur.end = cur.start + self.max_speech
            self.current = None
            return True
        return False

    def end(self, t: float) -> bool:
        cur = self.open
        self.current, self._pending, self._pending_hits = None, None, 0
        if cur is not None:
            cur.end = max(cur.start, t)
            return True
        return False

    def json(self) -> list[dict]:
        return [s.json() for s in self.segments]

    def restore(self, video: str | None, segments: list[dict]) -> bool:
        """Reload a timeline the Worker kept, after a container restart on the same stream."""
        if self.segments or not video or self.video not in (None, video):
            return False
        self.video = video
        self.segments = [Segment(float(s["from"]), None if s.get("to") is None else float(s["to"]), int(s["mp"]),
                                 float(s.get("talk") or 0), heard=s.get("talk") is not None) for s in segments]
        if self.open is not None:
            self.current = ("mp", self.open.mp)
        return True
