"""skupstina-reader: watches the @ParlamentSrbija live stream and reports who is speaking.

Every second of stream time it OCRs the caption strip, picks the speaker-name line, matches it
against the MPs from /api/mps and keeps a timeline [{from, to, mp}] in stream seconds. Each
change (and a heartbeat) is POSTed to the Worker at PUSH_URL with the shared secret.

Local endpoints (port 8080): /health, /segments, /debug.
Replay mode (SOURCE_URL set): reads a recording instead, prints the timeline and exits.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import caption, frames, hls, match
from .config import Config
from .tracker import Tracker

log = logging.getLogger("reader")

BOOT = uuid.uuid4().hex[:12]
RESOLVE_EVERY = 3 * 3600  # signed HLS URLs last ~6 h
MPS_EVERY = 6 * 3600
OFFLINE_POLL = 60
HEARTBEAT = 15
BACKLOG = 3  # segments read when joining a stream
MIN_INK = 0.002  # less white than this: no caption on screen
SAME_IMAGE = 0.01
MAX_REUSE = 5.0  # seconds a reading may be reused for an unchanged crop


VOICE_HISTORY = 1800  # loudness windows kept per pass (0.5 s each: 15 minutes)


def voice_threshold(history: list[float], fallback: float, margin: float) -> float:
    """dBFS above which a 0.5 s window counts as someone talking. Adapts to the stream: the room's background
    level (quietest 5 % of the last 15 minutes) plus a margin, never stricter than -35 dBFS or looser than -60."""
    if len(history) < 60:  # under 30 s of audio: not enough to know the room yet
        return fallback
    q = sorted(history)
    floor = q[int(0.05 * (len(q) - 1))]
    return min(-35.0, max(-60.0, floor + margin))


class Pass:
    """State of one pass over the video: live, or the backfill of what aired before we joined."""

    def __init__(self, tracker: Tracker, interval: float):
        self.tracker = tracker
        self.sampler = frames.Sampler(interval)
        self.sig = None
        self.sig_t = -1e9
        self.sig_obs = match.Observation("none")
        self.n_frames = self.n_ocr = 0
        self.db: list[float] = []  # recent loudness windows (dBFS), to find the room's background level
        self.n_audio = 0
        self.threshold = 0.0
        self.kinds: dict = {}
        self.last_none: dict | None = None
        self.snap_at = 0.0


class Reader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.tracker = Tracker(confirm=cfg.confirm, max_speech=cfg.max_speech, end_silent=cfg.end_silent, end_quiet=cfg.end_quiet, pre_gap=cfg.pre_gap)
        self.live_pass = Pass(self.tracker, cfg.interval)
        self.sampler = self.live_pass.sampler
        self.backfill: "Backfill | None" = None
        self.restored: list[dict] = []
        self.audio_template: str | None = None
        self.pending_bf: tuple | None = None
        self.bf_resume_t = 0.0
        self.index: list[match.MP] = []
        self.mps_at = 0.0
        self.stream: hls.Stream | None = None
        self.stream_at = 0.0
        self.last_seq: int | None = None
        self.last_t: float | None = None
        self.live = False
        self.offline_since: float | None = None
        self.dirty = True
        self.pushed_at = 0.0
        self.restore_tried = False
        self.last_frame = self.last_cap = None
        self.bf_frame = self.bf_cap = None
        self.bf_t = 0.0
        self.last_t_frame = 0.0
        self.snap_at = 0.0
        self.debug: dict = {"boot": BOOT, "readings": [], "backfill_readings": []}

    # ---- inputs -----------------------------------------------------------
    def refresh_mps(self) -> None:
        if self.index and time.time() - self.mps_at < MPS_EVERY:
            return
        try:
            idx = match.build_index(json.loads(hls.get(self.cfg.mps_url, timeout=30)))
            if idx:
                self.index, self.mps_at = idx, time.time()
                log.info("loaded %d MPs", len(idx))
        except Exception as e:
            self.note("mps_error", repr(e)[:300])

    def read(self, t: float, frame, ps: "Pass | None" = None) -> match.Observation:
        ps = ps or self.live_pass
        if frames.is_break(frame):
            ps.n_frames += 1
            ps.kinds["break"] = ps.kinds.get("break", 0) + 1
            return match.Observation("break")
        cap = caption.capture(frame, self.cfg.region, self.cfg.white)
        ps.n_frames += 1
        if ps is self.live_pass:
            self.last_frame, self.last_cap, self.last_t_frame = frame, cap, t
        elif cap.ink >= MIN_INK and time.time() - ps.snap_at > 60:
            ps.snap_at = time.time()  # a backfill frame with something white in the caption area
            self.bf_frame, self.bf_cap, self.bf_t = frame, cap, t
        if cap.ink < MIN_INK:
            obs, lines = match.Observation("none"), []
        elif caption.signature_distance(ps.sig, cap.signature) < SAME_IMAGE and t - ps.sig_t < MAX_REUSE:
            return self.with_timer(frame, ps, ps.sig_obs)
        else:
            ps.n_ocr += 1
            lines = caption.ocr_lines(cap.image, self.cfg.langs)
            cur = ps.tracker.current
            obs = match.classify(lines, self.index, self.cfg.min_confidence, cur[1] if cur and cur[0] == "mp" else None)
            ps.sig, ps.sig_t, ps.sig_obs = cap.signature, t, obs
            ps.kinds[obs.kind] = ps.kinds.get(obs.kind, 0) + 1
            if obs.kind == "none" and lines and ps is not self.live_pass:
                ps.last_none = {"t": round(t, 1), "ocr": [f"{l.confidence:.0f}% {l.text}" for l in lines][:4]}
        obs = self.with_timer(frame, ps, obs)
        if obs.kind != "none" or ps is self.live_pass:
            r = self.debug["readings" if ps is self.live_pass else "backfill_readings"]
            r.append({"t": round(t, 1), "kind": obs.kind, "mp": obs.mp, "line": obs.line, "timer": obs.elapsed,
                      "ocr": [f"{l.confidence:.0f}% {l.text}" for l in lines][:4]})
            del r[:-20]
        return obs

    def with_timer(self, frame, ps: "Pass", obs: match.Observation) -> match.Observation:
        """An MP's name is on screen: also read the speech timer, while it can still move their start."""
        if obs.kind != "mp" or not ps.tracker.wants_timer(obs.mp):
            return obs
        try:
            elapsed = caption.read_timer(frame, self.cfg.white)
        except Exception:
            elapsed = None
        return dataclasses.replace(obs, elapsed=elapsed) if elapsed is not None else obs

    def run_segment(self, data: bytes, t0: float, ps: "Pass", chunk: "Chunk | None" = None, audio: bytes | None = None) -> None:
        """Frames and audio of one media segment, in time order, into a pass's tracker. `audio` is the matching
        segment of a separate audio-only playlist, when the video has no sound of its own."""
        windows = frames.segment_audio(audio if audio is not None else data, t0)
        ps.n_audio += len(windows)
        for _, _, db in windows:
            ps.db.append(db)
        del ps.db[:-VOICE_HISTORY]
        threshold = voice_threshold(ps.db, self.cfg.voice_db, self.cfg.voice_margin)
        ps.threshold = threshold
        i = 0

        def audio_until(limit: float) -> None:
            nonlocal i
            with self.lock:
                while i < len(windows) and windows[i][0] < limit:
                    w0, dt, db = windows[i]
                    if ps.tracker.audio(w0, dt, db > threshold, db < self.cfg.silent_db):
                        self.dirty = True
                    i += 1

        for t, frame in frames.segment_frames(data, t0, ps.sampler):
            audio_until(t)
            obs = self.read(t, frame, ps)
            with self.lock:
                if ps.tracker.observe(t, obs):
                    self.dirty = True
                    if ps is self.live_pass:
                        log.info("t=%.1f now: %s", t, ps.tracker.current)
                if ps is self.live_pass:
                    self.last_t = t
                elif chunk is not None:
                    chunk.last_t = t
        audio_until(float("inf"))

    def feed(self, t: float, frame) -> None:
        obs = self.read(t, frame)
        with self.lock:
            if self.tracker.observe(t, obs):
                self.dirty = True
                log.info("t=%.1f now: %s", t, self.tracker.current)
            self.last_t = t

    def note(self, k: str, v) -> None:
        with self.lock:
            self.debug[k] = v
            self.debug[k + "_at"] = int(time.time())

    # ---- live loop --------------------------------------------------------
    def go_offline(self) -> None:
        with self.lock:
            if self.last_t is not None and self.tracker.end(self.last_t):
                self.dirty = True
            if self.live:
                self.dirty = True
            self.live = False
        self.stream = None
        self.last_seq = None
        if self.offline_since is None:
            self.offline_since = time.time()

    def step(self) -> float:
        self.refresh_mps()
        if self.stream is None or time.time() - self.stream_at > RESOLVE_EVERY:
            s = hls.resolve_live(self.cfg.channel, self.cfg.max_height, self.cfg.cookies, self.cfg.proxy)
            if s is None:
                self.go_offline()
                self.note("status", "offline")
                return OFFLINE_POLL
            with self.lock:
                if s.video != self.tracker.video:
                    if self.backfill:
                        self.backfill.stop = True
                        self.backfill = None
                    self.tracker.reset(s.video)
                    self.restored = []
                    self.last_seq = self.last_t = None
                    self.restore_tried = False
                    self.dirty = True
                if not self.live:
                    self.dirty = True
                self.live = True
            self.offline_since = None
            self.stream, self.stream_at = s, time.time()
            self.note("status", f"live {s.video} {s.info}")

        try:
            text = hls.get(self.stream.url, self.stream.headers).decode("utf-8", "replace")
        except hls.HTTPError as e:
            if e.code in (403, 404, 410):  # URL expired or stream gone: resolve again
                self.stream = None
                return 2
            raise
        segs, ended = hls.parse_playlist(text, self.stream.url)
        if not segs:
            return 2
        if self.last_seq is None or segs[-1].seq < self.last_seq:
            self.last_seq = segs[-1].seq - BACKLOG
            self.start_backfill(segs[-1], self.last_seq + 1)
        new = [s for s in segs if s.seq > self.last_seq]
        if len(new) > 12:  # fell behind: skip ahead instead of lagging forever
            new = new[-BACKLOG:]
        started = time.time()
        audio_by_seq: dict[int, str] = {}
        if self.stream.audio_url:
            try:
                atext = hls.get(self.stream.audio_url, self.stream.headers).decode("utf-8", "replace")
                asegs, _ = hls.parse_playlist(atext, self.stream.audio_url)
                audio_by_seq = {a.seq: a.uri for a in asegs}
                if asegs:
                    self.audio_template = asegs[-1].uri
            except Exception as e:
                self.note("audio_error", repr(e)[:200])
        for s in new:
            data = hls.get(s.uri, self.stream.headers, timeout=30)
            audio = None
            if s.seq in audio_by_seq:
                try:
                    audio = hls.get(audio_by_seq[s.seq], self.stream.headers, timeout=30)
                except Exception as e:
                    self.note("audio_error", repr(e)[:200])
            # YouTube numbers live segments from the start of the broadcast, so seq * duration
            # is the position the embedded player reports with getCurrentTime().
            t0 = s.seq * s.duration + self.cfg.time_offset
            self.run_segment(data, t0, self.live_pass, audio=audio)
            self.last_seq = s.seq
            if self.live_pass.db:
                q = sorted(self.live_pass.db)
                self.debug["audio_db"] = {**{f"p{p}": round(q[int(p / 100 * (len(q) - 1))], 1) for p in (5, 50, 90)},
                                          "threshold": round(self.live_pass.threshold, 1)}
            self.debug["audio_windows"] = self.live_pass.n_audio
            self.debug.update(seq=s.seq, seg_duration=s.duration, stream_t=round(t0, 1),
                              behind_s=round((segs[-1].seq - s.seq) * s.duration, 1))
        if new:
            # processing speed: 1.0 = exactly real time, below 1 = falling behind
            self.debug["speed"] = round(sum(s.duration for s in new) / max(0.01, time.time() - started), 2)
        if ended:
            self.go_offline()
            return 10
        return 0.5 if new else max(1.0, (segs[-1].duration or 2) / 2)

    def start_backfill(self, latest: hls.Seg, end_seq: int) -> None:
        if self.backfill is not None or not self.cfg.backfill or end_seq <= 0:
            return
        if self.cfg.push_url and not self.restore_tried:
            self.pending_bf = (latest, end_seq)  # started after the first push tells us what's already done
            return
        if not BF_SQ.search(latest.uri):
            self.note("backfill_error", "segment URLs have no /sq/N/: can't rewind")
            return
        start = int(max(0.0, self.bf_resume_t - self.cfg.time_offset) // (latest.duration or 5.0))
        self.backfill = Backfill(self, self.tracker.video, latest, end_seq, start, self.cfg.backfill_workers)
        if start:
            log.info("backfill resumes at %.0f s", start * (latest.duration or 5.0))
        threading.Thread(target=self.backfill.run, daemon=True).start()

    def merged(self) -> list[dict]:
        """History, then the live pass. History is what the backfill has (re-)read, plus whatever was restored
        from before a restart on the stretches the backfill hasn't covered yet."""
        live = self.tracker.json()
        bf = self.backfill
        if bf is not None and bf.video != self.tracker.video:
            bf = None
        live_cut = live[0]["from"] if live else float("inf")
        covered = bf.covered() if bf else []
        hist = bf.json() if bf else []
        old = []
        for sg in self.restored:
            if sg["from"] >= live_cut or any(a <= sg["from"] < b for a, b in covered):
                continue
            sg = dict(sg)
            # stop it at the next thing the backfill knows about
            nxt = [h["from"] for h in hist if h["from"] > sg["from"]] + ([live_cut] if live else [])
            if nxt and (sg["to"] is None or sg["to"] > min(nxt)):
                sg["to"] = min(nxt)
            old.append(sg)
        if bf is not None and hist:
            later = [x["from"] for x in old if x["from"] >= bf.last_t]
            cut = min([live_cut] + later)
            if cut == float("inf"):
                cut = bf.end_t
            out = []
            for sg in hist:
                if sg["from"] >= cut:
                    break
                if sg["to"] is None or sg["to"] > cut:
                    if not bf.done:
                        sg = dict(sg, to=round(min(bf.last_t, cut), 1)) if sg["from"] < bf.last_t else dict(sg, to=sg["to"] if sg["to"] is not None and sg["to"] <= cut else round(cut, 1))
                    elif live:
                        sg = dict(sg, to=cut)
                    else:
                        sg = dict(sg, to=None if self.tracker.current is None else cut)
                out.append(sg)
            hist = out
        return sorted(hist + old, key=lambda x: x["from"]) + live

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "boot": BOOT,
                "video": self.tracker.video,
                "live": self.live,
                "updated": int(time.time() * 1000),
                "segments": self.merged(),
            }

    def snapshot_jpeg(self, which: str = "live") -> bytes | None:
        """Frame (scaled), caption strip and the mask OCR saw, stacked: for debugging only."""
        import io
        from PIL import Image

        f, cap = (self.last_frame, self.last_cap) if which == "live" else (self.bf_frame, self.bf_cap)
        if f is None or cap is None:
            return None
        W = 960
        full = f.convert("RGB").resize((W, round(f.height * W / f.width)))
        x, y, w, h = self.cfg.region
        strip = f.convert("RGB").crop((round(x / 100 * f.width), round(y / 100 * f.height),
                                       round((x + w) / 100 * f.width), round((y + h) / 100 * f.height)))
        strip = strip.resize((W, round(strip.height * W / strip.width)))
        mask = cap.image.convert("RGB").resize((W, round(cap.image.height * W / cap.image.width)))
        out = Image.new("RGB", (W, full.height + strip.height + mask.height + 8), (255, 0, 0))
        out.paste(full, (0, 0))
        out.paste(strip, (0, full.height + 4))
        out.paste(mask, (0, full.height + strip.height + 8))
        b = io.BytesIO()
        out.save(b, "JPEG", quality=70)
        return b.getvalue()

    def push(self, force: bool = False) -> None:
        if not self.cfg.push_url or not (force or self.dirty or time.time() - self.pushed_at > HEARTBEAT):
            return
        self.dirty = False
        body = self.snapshot()
        reach = self.bf_reach()
        body["bfReach"] = max(reach, self.bf_resume_t) if reach is not None else self.bf_resume_t
        body["bfVersion"] = BF_VERSION
        with self.lock:
            body["debug"] = {k: v for k, v in self.debug.items() if k not in ("readings", "backfill_readings")} | {
                "readings": self.debug["readings"][-6:], "backfill_readings": self.debug["backfill_readings"][-8:]}
        if time.time() - self.snap_at > 30:
            try:
                import base64
                self.snap_n = getattr(self, "snap_n", 0) + 1
                which = "backfill" if self.snap_n % 2 == 0 and self.bf_frame is not None else "live"
                jpg = self.snapshot_jpeg(which)
                if jpg:
                    body["snapshot"] = {"t": round(self.last_t_frame if which == "live" else self.bf_t, 1), "which": which,
                                        "jpeg": base64.b64encode(jpg).decode()}
                    self.snap_at = time.time()
            except Exception as e:
                self.note("snapshot_error", repr(e)[:200])
        try:
            r = hls.get(self.cfg.push_url, {"authorization": f"Bearer {self.cfg.secret}", "content-type": "application/json"},
                        timeout=15, data=json.dumps(body).encode(), method="POST")
            self.pushed_at = time.time()
            self.maybe_restore(json.loads(r or b"{}"))
        except Exception as e:
            self.dirty = True
            self.note("push_error", repr(e)[:300])

    def bf_reach(self) -> float | None:
        bf = self.backfill
        if bf is None or bf.video != self.tracker.video:
            return None
        return bf.end_t if bf.done else bf.last_t  # only the gap-free part: a restart resumes from there

    def maybe_restore(self, resp: dict) -> None:
        """After a restart the Worker sends back what it had for this stream; it fills history until the
        backfill has re-read that stretch."""
        prev = resp.get("previous")
        if self.restore_tried:
            return
        self.restore_tried = True
        with self.lock:
            if prev and prev.get("video") and prev.get("video") == self.tracker.video and not self.restored:
                self.restored = [dict(sg) for sg in prev.get("segments") or []]
                same_rules = int(prev.get("bfVersion") or 1) == BF_VERSION
                self.bf_resume_t = float(prev.get("bfReach") or 0) if same_rules else 0.0  # new rules: rescan
                log.info("restored %d segments, backfill had reached %.0f s", len(self.restored), self.bf_resume_t)
                self.dirty = True
        if self.pending_bf:
            latest, end_seq = self.pending_bf
            self.pending_bf = None
            self.start_backfill(latest, end_seq)

    def run(self) -> None:
        while True:
            try:
                wait = self.step()
            except Exception as e:
                log.exception("step failed")
                self.note("error", repr(e)[:500])
                self.stream = None
                wait = 20
            self.push()
            if self.offline_since and time.time() - self.offline_since > self.cfg.offline_exit:
                log.info("offline for %ds, exiting", self.cfg.offline_exit)
                self.push(force=True)
                return
            end = time.time() + wait
            while time.time() < end:  # keep heartbeats going during long waits
                time.sleep(min(5, max(0.0, end - time.time())))
                self.push()

    # ---- replay (testing) -------------------------------------------------
    def replay(self) -> dict:
        self.refresh_mps()
        s = hls.resolve_recording(self.cfg.source_url, self.cfg.max_height, self.cfg.cookies, self.cfg.proxy, self.cfg.channel)
        log.info("replay %s at %sp from %ss for %ss", s.video, s.height, self.cfg.start, self.cfg.duration)
        self.tracker.reset(s.video)
        stop = self.cfg.start + self.cfg.duration
        started = time.time()
        n = 0
        for t, frame in frames.url_frames(s.url, s.headers, self.cfg.start, self.sampler):
            if t < self.cfg.start:
                continue
            if t > stop:
                break
            self.feed(t, frame)
            n += 1
            r = self.debug["readings"][-1] if self.debug["readings"] else {}
            if r.get("t") == round(t, 1) and r.get("kind") != "none":
                print(f"{t:8.1f}  {r['kind']:5}  {r.get('mp') or '':>5}  {r['line']}   | {' / '.join(r['ocr'])}", flush=True)
        self.tracker.end(stop)
        return {"video": s.video, "height": s.height, "frames": n, "seconds": round(time.time() - started),
                "segments": self.tracker.json()}


BF_SQ = re.compile(r"/sq/(\d+)/")


BF_VERSION = 4  # bump to rescan history with new detection rules (2: break screen + talk time, 3: audio-only playlist, 4: stricter name matching)


class Chunk:
    """One stretch of the history read by one worker thread."""

    def __init__(self, bf: "Backfill", start_seq: int, end_seq: int):
        r = bf.r
        self.bf, self.seq, self.end_seq = bf, start_seq, end_seq
        self.start_t = start_seq * bf.dur + r.cfg.time_offset
        self.end_t = end_seq * bf.dur + r.cfg.time_offset
        self.tracker = Tracker(confirm=r.cfg.confirm, max_speech=r.cfg.max_speech, end_silent=r.cfg.end_silent, end_quiet=r.cfg.end_quiet, pre_gap=r.cfg.pre_gap)
        self.ps = Pass(self.tracker, r.cfg.interval)
        self.last_t = self.start_t
        self.done = False

    def run(self) -> None:
        bf, r = self.bf, self.bf.r
        fails = 0
        while self.seq < self.end_seq and not bf.stop:
            try:
                data = hls.get(bf.url(self.seq), r.stream.headers if r.stream else None, timeout=30)
            except hls.HTTPError as e:
                if e.code in (403, 404, 410):  # outside the DVR window or a gap: skip ahead
                    fails += 1
                    self.seq += 60 if fails < 50 else 600
                    continue
                time.sleep(5)
                continue
            except Exception:
                time.sleep(5)
                continue
            fails = 0
            t0 = self.seq * bf.dur + r.cfg.time_offset
            audio = None
            if r.audio_template and BF_SQ.search(r.audio_template):
                try:
                    audio = hls.get(BF_SQ.sub(f"/sq/{self.seq}/", r.audio_template, count=1), r.stream.headers if r.stream else None, timeout=30)
                except Exception:
                    audio = None
            try:
                r.run_segment(data, t0, self.ps, self, audio=audio)
            except Exception as e:
                r.note("backfill_error", repr(e)[:200])
            self.seq += 1
            self.last_t = max(self.last_t, t0 + bf.dur)
            bf.report()
        with r.lock:
            self.done = not bf.stop
            r.dirty = True


class Backfill:
    """Reads what aired before the reader joined, from YouTube's DVR window, faster than real time,
    split into a few stretches read in parallel."""

    def __init__(self, reader: Reader, video: str | None, latest: hls.Seg, end_seq: int, start_seq: int = 0,
                 workers: int = 2):
        self.r, self.video, self.template, self.dur = reader, video, latest.uri, latest.duration or 5.0
        start_seq = max(0, start_seq)
        self.start_t = start_seq * self.dur + reader.cfg.time_offset
        self.end_t = end_seq * self.dur + reader.cfg.time_offset
        n = max(1, workers if end_seq - start_seq > 240 else 1)  # short stretches: one worker
        cuts = [start_seq + (end_seq - start_seq) * k // n for k in range(n + 1)]
        self.chunks = [Chunk(self, a, b) for a, b in zip(cuts, cuts[1:])]
        self.stop = False
        self.started = time.time()

    @property
    def done(self) -> bool:
        return all(c.done for c in self.chunks)

    @property
    def last_t(self) -> float:
        """How far the history is read without gaps from the start."""
        for c in self.chunks:
            if not c.done:
                return c.last_t
        return self.end_t

    def covered(self) -> list[tuple[float, float]]:
        return [(c.start_t, c.end_t if c.done else c.last_t) for c in self.chunks]

    def url(self, seq: int) -> str:
        return BF_SQ.sub(f"/sq/{seq}/", self.template, count=1)

    def json(self) -> list[dict]:
        """All stretches as one timeline: a speech still open at the end of a stretch runs on until the next
        stretch's first speaker or break, or as far as reading has got."""
        out: list[dict] = []
        for c in self.chunks:
            segs = c.tracker.json()
            if out and out[-1]["to"] is None:
                firsts = [x for x in (segs[0]["from"] if segs else None, c.tracker.first_break) if x is not None]
                if firsts:
                    out[-1]["to"] = round(min(firsts), 1)
                elif not c.done:
                    out[-1]["to"] = round(c.last_t, 1) if c.last_t > c.start_t else round(c.start_t, 1)
            out.extend(segs)
        return out

    def report(self) -> None:
        r = self.r
        read = sum(c.last_t - c.start_t for c in self.chunks)
        ps = [c.ps for c in self.chunks]
        kinds: dict = {}
        for p in ps:
            for k, v in p.kinds.items():
                kinds[k] = kinds.get(k, 0) + v
        r.debug.update(backfill_t=round(self.last_t), backfill_end=round(self.end_t),
                       backfill_read=round(read), backfill_speed=round(read / max(1, time.time() - self.started), 2),
                       backfill_stats={"frames": sum(p.n_frames for p in ps), "ocr": sum(p.n_ocr for p in ps), "kinds": kinds,
                                       "audio_windows": sum(p.n_audio for p in ps)})

    def run(self) -> None:
        threads = [threading.Thread(target=c.run, daemon=True) for c in self.chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.r.debug["backfill_done"] = self.done
        log.info("backfill done: %d segments", len(self.json()))


def serve(reader: Reader, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p == "/health":
                obj, code = "ok", 200
            elif p == "/segments":
                obj, code = reader.snapshot(), 200
            elif p == "/debug":
                with reader.lock:
                    obj, code = dict(reader.debug, mps=len(reader.index)), 200
            else:
                obj, code = {"error": "not found"}, 404
            b = (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)).encode()
            self.send_response(code)
            self.send_header("content-type", "text/plain" if isinstance(obj, str) else "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = Config.from_env()
    reader = Reader(cfg)
    if cfg.source_url:
        out = reader.replay()
        names = {m.id: m.names[0] for m in reader.index}
        for s in out["segments"]:
            print(f"{s['from']:8.1f} – {s['to'] if s['to'] is not None else '…':>8}  {s['mp']:>5}  {names.get(s['mp'], '?')}")
        print(json.dumps(out))
        return
    serve(reader, cfg.port)
    log.info("listening on :%d, boot %s", cfg.port, BOOT)
    reader.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
