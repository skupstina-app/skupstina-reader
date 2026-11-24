"""Finding the stream with yt-dlp and walking its HLS media playlist."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urljoin

log = logging.getLogger("reader.hls")

UA = "Mozilla/5.0 (compatible; skupstina-reader; +https://skupstina.app)"
HTTPError = urllib.error.HTTPError
NOT_LIVE = ("not currently live", "will begin", "Premieres", "This live event", "is offline", "does not have a live")


@dataclass
class Stream:
    video: str
    url: str
    headers: dict
    height: int
    live: bool


@dataclass
class Seg:
    seq: int
    duration: float
    uri: str


def _ydl_opts(cookiefile: str | None, proxy: str | None) -> dict:
    o = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    if cookiefile:
        o["cookiefile"] = cookiefile
    if proxy:
        o["proxy"] = proxy
    return o


def _pick(formats: list[dict], max_height: int, want) -> dict | None:
    fs = [f for f in formats if want(f) and f.get("vcodec") not in (None, "none") and f.get("height")]
    if not fs:
        return None
    under = [f for f in fs if f["height"] <= max_height]
    # prefer H.264: cheapest to decode
    key = lambda f: (f["height"], str(f.get("vcodec", "")).startswith("avc"))
    return max(under, key=key) if under else min(fs, key=lambda f: f["height"])


def resolve_live(channel: str, max_height: int, cookiefile: str | None = None, proxy: str | None = None) -> Stream | None:
    """The channel's live HLS rendition, or None when nothing is live."""
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(_ydl_opts(cookiefile, proxy)) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/channel/{channel}/live", download=False)
    except yt_dlp.utils.DownloadError as e:
        if any(s in str(e) for s in NOT_LIVE):
            return None
        raise
    if not info or not info.get("is_live"):
        return None
    f = _pick(info.get("formats") or [], max_height, lambda f: str(f.get("protocol", "")).startswith("m3u8"))
    if f is None:
        raise RuntimeError("live, but no HLS video format offered")
    log.info("live %s at %sp", info["id"], f["height"])
    return Stream(info["id"], f["url"], dict(f.get("http_headers") or {}), f["height"], True)


def latest_recording(channel: str, cookiefile: str | None = None, proxy: str | None = None) -> str:
    """URL of the channel's most recent finished live stream (for replay tests)."""
    import yt_dlp

    opts = _ydl_opts(cookiefile, proxy) | {"extract_flat": True, "playlistend": 10, "noplaylist": False}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/channel/{channel}/streams", download=False)
    for e in info.get("entries") or []:
        if e.get("live_status") not in ("is_live", "is_upcoming") and e.get("id"):
            return f"https://www.youtube.com/watch?v={e['id']}"
    raise RuntimeError("no finished stream found on the channel")


def resolve_recording(url: str, max_height: int, cookiefile: str | None = None, proxy: str | None = None, channel: str = "") -> Stream:
    import yt_dlp

    if url == "latest":
        url = latest_recording(channel, cookiefile, proxy)
        log.info("latest recording: %s", url)
    with yt_dlp.YoutubeDL(_ydl_opts(cookiefile, proxy)) as ydl:
        info = ydl.extract_info(url, download=False)
    f = _pick(info.get("formats") or [], max_height, lambda f: f.get("protocol") in ("https", "http"))
    if f is None:
        raise RuntimeError("no progressive/https video format for " + url)
    return Stream(info["id"], f["url"], dict(f.get("http_headers") or {}), f["height"], False)


def get(url: str, headers: dict | None = None, timeout: float = 20, data: bytes | None = None, method: str | None = None) -> bytes:
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h, data=data, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_playlist(text: str, base: str) -> tuple[list[Seg], bool]:
    """Media playlist -> (segments, ended)."""
    seq, dur, ended = 0, None, False
    segs: list[Seg] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            seq = int(line.split(":", 1)[1])
        elif line.startswith("#EXTINF:"):
            dur = float(line.split(":", 1)[1].split(",", 1)[0])
        elif line.startswith("#EXT-X-ENDLIST"):
            ended = True
        elif line and not line.startswith("#"):
            segs.append(Seg(seq, dur or 0.0, urljoin(base, line)))
            seq, dur = seq + 1, None
    return segs, ended
