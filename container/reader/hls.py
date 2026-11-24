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
    audio_url: str | None = None  # separate audio-only playlist when the video rendition has no sound
    info: str = ""


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
    # highest height, then <= 30 fps and H.264: cheapest to decode in real time
    key = lambda f: (f["height"], (f.get("fps") or 30) <= 31, str(f.get("vcodec", "")).startswith("avc"))
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
    fmts = info.get("formats") or []
    hls_ = lambda f: str(f.get("protocol", "")).startswith("m3u8")
    has_audio = lambda f: f.get("acodec") not in (None, "none")
    # prefer a rendition with sound in it; otherwise take the video and a separate audio-only playlist
    f = _pick(fmts, max_height, lambda f: hls_(f) and has_audio(f)) or _pick(fmts, max_height, hls_)
    if f is None:
        raise RuntimeError("live, but no HLS video format offered")
    audio_url = None
    if not has_audio(f):
        auds = [a for a in fmts if hls_(a) and has_audio(a) and a.get("vcodec") in (None, "none")]
        if auds:
            a = max(auds, key=lambda a: a.get("abr") or a.get("tbr") or 0)
            audio_url = a["url"]
    desc = f"{f.get('format_id')} {f['height']}p{f.get('fps') or ''} v={f.get('vcodec')} a={f.get('acodec')}" + (" +audio-only" if audio_url else "")
    log.info("live %s: %s", info["id"], desc)
    return Stream(info["id"], f["url"], dict(f.get("http_headers") or {}), f["height"], True, audio_url, desc)


def latest_recording(channel: str, cookiefile: str | None = None, proxy: str | None = None) -> str:
    """URL of the channel's most recent finished live stream (for replay tests)."""
    import yt_dlp

    opts = _ydl_opts(cookiefile, proxy) | {"extract_flat": "in_playlist", "playlistend": 15, "noplaylist": False}

    def flat(entries):
        for e in entries or []:
            if e.get("_type") == "playlist" or e.get("entries"):
                yield from flat(e.get("entries"))
            else:
                yield e

    seen = []
    for tab in ("streams", "videos"):
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/channel/{channel}/{tab}", download=False)
        for e in flat(info.get("entries")):
            seen.append((tab, e.get("id"), e.get("live_status"), (e.get("title") or "")[:60]))
            dur = e.get("duration") or 0
            if e.get("id") and e.get("live_status") not in ("is_live", "is_upcoming") and (dur == 0 or dur > 1800):
                log.info("latest recording: %s %s", e["id"], e.get("title"))
                return f"https://www.youtube.com/watch?v={e['id']}"
    raise RuntimeError(f"no finished stream found on the channel; saw {seen[:10]}")


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
