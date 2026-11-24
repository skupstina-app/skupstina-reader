"""skupstina-reader: watches the @ParlamentSrbija live stream and reports who is speaking.

Every 1.5 s of stream time it OCRs the caption strip, picks the speaker-name line, matches it
against the MPs from /api/mps and keeps a timeline [{from, to, mp}] in stream seconds. Each
change (and a heartbeat) is POSTed to the Worker at PUSH_URL with the shared secret.

Local endpoints (port 8080): /health, /segments, /debug.
Replay mode (SOURCE_URL set): reads a recording instead, prints the timeline and exits.
"""

from __future__ import annotations

import json
import logging
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


class Reader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.tracker = Tracker(confirm=cfg.confirm, max_speech=cfg.max_speech)
        self.sampler = frames.Sampler(cfg.interval)
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
        self._sig = None
        self._sig_t = -1e9
        self._sig_obs = match.Observation("none")
        self.debug: dict = {"boot": BOOT, "readings": []}

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

    def read(self, t: float, frame) -> match.Observation:
        cap = caption.capture(frame, self.cfg.region, self.cfg.white)
        if cap.ink < MIN_INK:
            obs, lines = match.Observation("none"), []
        elif caption.signature_distance(self._sig, cap.signature) < SAME_IMAGE and t - self._sig_t < MAX_REUSE:
            return self._sig_obs
        else:
            lines = caption.ocr_lines(cap.image, self.cfg.langs)
            cur = self.tracker.current
            obs = match.classify(lines, self.index, self.cfg.min_confidence, cur[1] if cur and cur[0] == "mp" else None)
            self._sig, self._sig_t, self._sig_obs = cap.signature, t, obs
        r = self.debug["readings"]
        r.append({"t": round(t, 1), "kind": obs.kind, "mp": obs.mp, "line": obs.line,
                  "ocr": [f"{l.confidence:.0f}% {l.text}" for l in lines][:4]})
        del r[:-20]
        return obs

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
                    self.tracker.reset(s.video)
                    self.last_seq = self.last_t = None
                    self.restore_tried = False
                    self.dirty = True
                if not self.live:
                    self.dirty = True
                self.live = True
            self.offline_since = None
            self.stream, self.stream_at = s, time.time()
            self.note("status", f"live {s.video} {s.height}p")

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
        new = [s for s in segs if s.seq > self.last_seq]
        if len(new) > 12:  # fell behind: skip ahead instead of lagging forever
            new = new[-BACKLOG:]
        started = time.time()
        for s in new:
            data = hls.get(s.uri, self.stream.headers, timeout=30)
            # YouTube numbers live segments from the start of the broadcast, so seq * duration
            # is the position the embedded player reports with getCurrentTime().
            t0 = s.seq * s.duration + self.cfg.time_offset
            for t, frame in frames.segment_frames(data, t0, self.sampler):
                self.feed(t, frame)
            self.last_seq = s.seq
            self.debug.update(seq=s.seq, seg_duration=s.duration, stream_t=round(t0, 1),
                              behind_s=round((segs[-1].seq - s.seq) * s.duration, 1))
        if new:
            # processing speed: 1.0 = exactly real time, below 1 = falling behind
            self.debug["speed"] = round(sum(s.duration for s in new) / max(0.01, time.time() - started), 2)
        if ended:
            self.go_offline()
            return 10
        return 0.5 if new else max(1.0, (segs[-1].duration or 2) / 2)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "boot": BOOT,
                "video": self.tracker.video,
                "live": self.live,
                "updated": int(time.time() * 1000),
                "segments": self.tracker.json(),
            }

    def push(self, force: bool = False) -> None:
        if not self.cfg.push_url or not (force or self.dirty or time.time() - self.pushed_at > HEARTBEAT):
            return
        self.dirty = False
        body = self.snapshot()
        with self.lock:
            body["debug"] = {k: v for k, v in self.debug.items() if k != "readings"} | {"readings": self.debug["readings"][-6:]}
        try:
            r = hls.get(self.cfg.push_url, {"authorization": f"Bearer {self.cfg.secret}", "content-type": "application/json"},
                        timeout=15, data=json.dumps(body).encode(), method="POST")
            self.pushed_at = time.time()
            self.maybe_restore(json.loads(r or b"{}"))
        except Exception as e:
            self.dirty = True
            self.note("push_error", repr(e)[:300])

    def maybe_restore(self, resp: dict) -> None:
        """After a restart the Worker sends back what it had for this stream."""
        prev = resp.get("previous")
        if self.restore_tried or not prev:
            return
        self.restore_tried = True
        with self.lock:
            if self.tracker.restore(prev.get("video"), prev.get("segments") or []):
                log.info("restored %d segments", len(self.tracker.segments))
                self.dirty = True

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
