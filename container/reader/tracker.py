"""Readings -> speaker timeline in stream seconds.

Rules (after Neleceni's state machine):
- a speaker counts as changed after `confirm` readings of the same new name (MP or not) within
  `window` seconds; a stray misread of some other name in between no longer throws those readings
  away (it used to, so a short caption with one bad read was missed and the previous speaker kept
  the floor); the new segment is back-dated to the first of those readings;
- readings with no usable name change nothing, because the speech usually goes on after
  the caption fades;
- a non-MP speaker (minister, guest) closes the current MP segment without opening one;
- the speech timer (top right, read with the name) says how long they have been speaking: two readings
  that agree move the segment's start back to when they really began, even before the camera found them;
- sound ends a speech: the microphone cut (digital silence) for `end_silent` seconds, or no voice (only
  room noise) for `end_quiet` seconds. A cut speaker read again with a timer that still counts from the
  same start (the chair let them go on) gets their segment back instead of a new one;
- an open segment is closed after `max_speech` seconds as a safety net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .match import Observation, similar_lines


SILENCE_CUT = 5.0  # a pause this long stops counting as talking
TIMER_AGREE = 2.0  # two timer readings agree when the start times they give are this close (seconds)


@dataclass
class Segment:
    start: float
    end: float | None
    mp: int
    talk: float = 0.0  # seconds with voice, not counting pauses of SILENCE_CUT or more
    quiet: float = 0.0  # current run of silence
    heard: bool = False  # any audio at all for this segment (else talk time is unknown)
    starts: list = field(default_factory=list)  # start times given by timer readings
    anchored: bool = False  # start set from the timer
    muted: float = 0.0  # current run of digital silence
    closed: str = ""  # "sound" or "break" when something other than the next speaker ended it

    def json(self) -> dict:
        return {"from": round(self.start, 1), "to": None if self.end is None else round(self.end, 1), "mp": self.mp,
                "talk": round(self.talk, 1) if self.heard else None}


@dataclass
class Tracker:
    confirm: int = 3
    window: float = 12.0  # readings of a new name count together only this long after the last one
    end_silent: float = 0.5  # microphone cut this long: the speech is over
    end_quiet: float = 5.0  # no voice (room noise only) this long: the speech is over
    max_speech: float = 1800.0
    keep: int = 600
    video: str | None = None
    segments: list[Segment] = field(default_factory=list)
    current: tuple | None = None  # ("mp", id) | ("other", line) | None
    _cands: list = field(default_factory=list)  # [ident, first t, hits, last t, timer starts] for names not yet confirmed
    _cut: Segment | None = None  # the last segment ended by sound
    _voice_since_cut: bool = False
    _breaks: int = 0
    first_break: float | None = None  # first confirmed break before anyone spoke (joins stretches of history)
    _break_t: float = 0.0
    _last_break: float = -1.0  # last time the break screen was seen

    @property
    def open(self) -> Segment | None:
        return self.segments[-1] if self.segments and self.segments[-1].end is None else None

    def reset(self, video: str | None) -> None:
        self.video, self.segments, self.current = video, [], None
        self._cands = []

    @staticmethod
    def _same(a: tuple | None, b: tuple | None) -> bool:
        if a is None or b is None or a[0] != b[0]:
            return False
        return a[1] == b[1] if a[0] == "mp" else similar_lines(a[1], b[1])

    def audio(self, t: float, dt: float, voiced: bool, silent: bool = False) -> bool:
        """Count talk time for whoever is speaking (pauses shorter than SILENCE_CUT count, longer ones don't),
        and end the speech when the microphone is cut or nobody has spoken for a while. True if a segment ended."""
        if voiced:
            self._voice_since_cut = True
        cur = self.open
        if cur is None or t < cur.start:
            return False
        cur.heard = True
        if voiced:
            cur.talk += dt + (cur.quiet if cur.quiet < SILENCE_CUT else 0.0)
            cur.quiet = cur.muted = 0.0
            return False
        cur.quiet += dt
        cur.muted = cur.muted + dt if silent else 0.0
        if cur.muted >= self.end_silent - 1e-6:
            return self._cut_at(t + dt - cur.muted)
        if cur.quiet >= self.end_quiet - 1e-6:
            return self._cut_at(t + dt - cur.quiet)
        return False

    def _cut_at(self, t: float) -> bool:
        cur = self.open
        cur.end = max(cur.start, t)
        cur.closed = "sound"
        cur.quiet = cur.muted = 0.0
        self.current, self._cands = None, []
        self._cut, self._voice_since_cut = cur, False
        return True

    def wants_timer(self, mp: int | None) -> bool:
        """Is a timer reading for this MP still useful? (Saves OCR once a start is pinned down.)"""
        if mp is None:
            return False
        cur = self.open
        if cur is not None and cur.mp == mp:
            return not cur.anchored
        return True

    @staticmethod
    def _agreed(starts: list) -> float | None:
        """The start time two timer readings agree on (the later two that agree), else None."""
        for i in range(len(starts) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                if abs(starts[i] - starts[j]) <= TIMER_AGREE:
                    return min(starts[i], starts[j])
        return None

    def _earliest(self, seg: Segment) -> float:
        """A timer may move a start back, but not before the speech before it began (or ended, if sound or a
        break ended it), nor into a break."""
        i = self.segments.index(seg)
        lo = self._last_break if self._last_break >= 0 else 0.0
        if i > 0:
            prev = self.segments[i - 1]
            lo = max(lo, prev.start + 1.0)
            if prev.closed and prev.end is not None:
                lo = max(lo, prev.end)
        return lo

    def _anchor(self, seg: Segment, start: float) -> bool:
        start = max(start, self._earliest(seg))
        seg.anchored = True
        if start >= seg.start - 0.5:
            return False
        i = self.segments.index(seg)
        if i > 0 and (self.segments[i - 1].end is None or self.segments[i - 1].end > start):
            self.segments[i - 1].end = start
        seg.start = start
        return True

    def observe(self, t: float, obs: Observation) -> bool:
        """Feed one reading at stream time t. Returns True when the timeline changed."""
        changed = self.tick(t)
        if obs.kind == "break":  # the break screen, seen twice in a row: nobody is speaking
            self._cands = []
            self._last_break = t
            self._breaks += 1
            if self._breaks == 1:
                self._break_t = t
                return changed
            self.current = None
            if not self.segments and self.first_break is None:
                self.first_break = self._break_t
            if self.open is not None:
                self.open.closed = "break"
                self.open.end = max(self.open.start, self._break_t)
                return True
            return changed
        self._breaks = 0
        if obs.kind == "none":
            return changed
        ident = ("mp", obs.mp) if obs.kind == "mp" else ("other", obs.line)
        began = t - obs.elapsed if obs.kind == "mp" and obs.elapsed is not None and obs.elapsed <= self.max_speech else None
        if self._same(ident, self.current):
            self._cands = []  # the current speaker's caption is still up
            cur = self.open
            if began is not None and cur is not None and not cur.anchored:
                cur.starts.append(began)
                at = self._agreed(cur.starts)
                if at is not None and self._anchor(cur, at):
                    return True
            return changed
        cut = self._cut
        if cut is not None and ident == ("mp", cut.mp) and self.open is None:
            if began is not None and abs(began - cut.start) <= TIMER_AGREE * 2 and cut is self.segments[-1]:
                cut.end, cut.closed = None, ""  # the timer still counts from their start: the same speech goes on
                self.current, self._cut, self._cands = ident, None, []
                return True
            if not self._voice_since_cut:
                return changed  # their caption is still up after the cut: not a new speech
        self._cands = [c for c in self._cands if t - c[3] <= self.window]
        c = next((c for c in self._cands if self._same(ident, c[0])), None)
        if c is None:
            c = [ident, t, 0, t, []]
            self._cands.append(c)
        c[2] += 1
        c[3] = t
        if began is not None:
            c[4].append(began)
        if c[2] >= self.confirm:
            self._switch(c[1], c[0])
            if ident[0] == "mp":
                seg = self.segments[-1]
                seg.starts = list(c[4])
                at = self._agreed(seg.starts)
                if at is not None:
                    self._anchor(seg, at)
            self._cands = []
            self._cut = None
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
        self.current, self._cands = None, []
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
