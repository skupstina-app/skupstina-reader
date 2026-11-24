from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _region(v: str) -> tuple[float, float, float, float]:
    try:
        p = tuple(float(x) for x in v.split(","))
        if len(p) == 4:
            return p  # type: ignore[return-value]
    except ValueError:
        pass
    # Neleceni's default: left 0 %, top 80 %, width 90 %, height 20 % covers both caption sizes
    return (0.0, 80.0, 90.0, 20.0)


@dataclass
class Config:
    channel: str
    mps_url: str
    push_url: str | None
    secret: str | None
    max_height: int
    region: tuple[float, float, float, float]
    white: int
    interval: float
    confirm: int
    min_confidence: float
    langs: str
    time_offset: float
    max_speech: float
    offline_exit: float
    cookies: str | None
    proxy: str | None
    port: int
    # replay mode (testing): read a recording instead of the live stream
    source_url: str | None
    start: float
    duration: float

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ.get
        cookies = None
        if e("YT_COOKIES"):
            cookies = "/tmp/yt-cookies.txt"
            with open(cookies, "w") as f:
                f.write(e("YT_COOKIES", "").replace("\\n", "\n"))
        return cls(
            channel=e("YT_CHANNEL") or "UCghngrfQwzJ4aaPZ-UgNIZQ",
            mps_url=e("MPS_URL") or "https://skupstina.app/api/mps",
            push_url=e("PUSH_URL") or None,
            secret=e("READER_SECRET") or None,
            max_height=int(_f("MAX_HEIGHT", 720)),
            region=_region(e("REGION", "")),
            white=int(_f("WHITE_THRESHOLD", 225)),
            interval=_f("INTERVAL", 1.5),
            confirm=int(_f("CONFIRM", 3)),
            min_confidence=_f("MIN_CONFIDENCE", 35),
            langs=e("TESS_LANGS") or "srp+srp_latn",
            time_offset=_f("TIME_OFFSET", 0.0),
            max_speech=_f("MAX_SPEECH", 1800),
            offline_exit=_f("OFFLINE_EXIT", 900),
            cookies=cookies,
            proxy=e("YTDLP_PROXY") or None,
            port=int(_f("PORT", 8080)),
            source_url=e("SOURCE_URL") or None,
            start=_f("START", 0),
            duration=_f("DURATION", 600),
        )
