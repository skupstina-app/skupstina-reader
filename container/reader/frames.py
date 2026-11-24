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
