"""Decoding video and sampling one frame every `interval` seconds of stream time."""

from __future__ import annotations

import io
from typing import Iterator

from PIL import Image


class Sampler:
    """Keeps a fixed grid (0, 1.5, 3.0, ...) across segments so sampling never drifts."""

    def __init__(self, interval: float):
        self.interval = interval
        self.next_t: float | None = None

    def due(self, t: float) -> bool:
        if self.next_t is None or t + self.interval < self.next_t:  # first frame, or stream restarted
            self.next_t = t
        if t >= self.next_t:
            self.next_t = (int(t / self.interval) + 1) * self.interval
            return True
        return False


def _decode(container, sampler: Sampler, t_of) -> Iterator[tuple[float, Image.Image]]:
    st = container.streams.video[0]
    st.thread_type = "AUTO"
    for frame in container.decode(st):
        if frame.time is None:
            continue
        t = t_of(frame.time)
        if sampler.due(t):
            yield t, frame.to_image()


def segment_frames(data: bytes, t0: float, sampler: Sampler) -> Iterator[tuple[float, Image.Image]]:
    """Frames of one HLS media segment; t0 is the segment's start in stream time."""
    import av

    with av.open(io.BytesIO(data)) as c:
        if not c.streams.video:
            return
        first: list[float] = []

        def t_of(ft: float) -> float:
            if not first:
                first.append(ft)
            return t0 + ft - first[0]

        yield from _decode(c, sampler, t_of)


def url_frames(url: str, headers: dict, start: float, sampler: Sampler) -> Iterator[tuple[float, Image.Image]]:
    """Frames of a recording (replay testing); times are seconds into the video."""
    import av

    hdr = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    with av.open(url, options={"headers": hdr} if hdr else {}) as c:
        st = c.streams.video[0]
        if start > 0:
            c.seek(int(start / st.time_base), stream=st, backward=True)
        yield from _decode(c, sampler, lambda ft: ft)


def is_break(img: Image.Image) -> bool:
    """The "WWW.PARLAMENT.RS" break screen: a big blue ellipse in the middle and gold text along the bottom.
    In normal shots both are ~0 (the small corner logo stays outside these areas)."""
    import numpy as np

    a = np.asarray(img.convert("RGB").resize((192, 108))).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b > 110) & (b - r > 50) & (b - g > 30)
    gold = (r > 150) & (g > 110) & (b < 110) & (r - b > 70) & (g - b > 40)
    mid = blue[16:81, 38:154].mean()
    bottom = gold[86:].mean()
    return bool(mid > 0.15 and bottom > 0.03)


def segment_audio(data: bytes, t0: float, window: float = 0.5) -> list[tuple[float, float, float]]:
    """Loudness of a media segment's audio: [(start, duration, dBFS)] per `window` seconds."""
    import math

    import av
    import numpy as np

    out: list[tuple[float, float, float]] = []
    try:
        with av.open(io.BytesIO(data)) as c:
            if not c.streams.audio:
                return out
            st = c.streams.audio[0]
            rate = st.codec_context.sample_rate or 48000
            buf: list = []
            first = None
            for fr in c.decode(st):
                if first is None:
                    first = fr.time or 0.0
                x = fr.to_ndarray()
                if x.dtype.kind in "iu":
                    x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
                buf.append(x.reshape(-1, x.shape[-1]).mean(axis=0) if x.ndim > 1 and x.shape[0] <= 8 else x.ravel())
            if not buf:
                return out
            mono = np.concatenate(buf).astype(np.float32)
            n = max(1, int(rate * window))
            for i in range(0, len(mono) - n // 2, n):
                w = mono[i : i + n]
                rms = float(np.sqrt(np.mean(w * w))) if len(w) else 0.0
                out.append((t0 + i / rate, len(w) / rate, 20 * math.log10(max(rms, 1e-6))))
    except Exception:
        return []
    return out
