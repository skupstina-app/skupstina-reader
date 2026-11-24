"""Save full frames of a recording as JPEGs (for looking at captions/timer by eye).

python -m reader.dump URL START DURATION EVERY OUTDIR
"""
import os
import sys

from . import frames, hls
from .config import Config


def main() -> None:
    url, start, duration, every, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    cfg = Config.from_env()
    s = hls.resolve_recording(url, cfg.max_height, cfg.cookies, cfg.proxy, cfg.channel)
    os.makedirs(out, exist_ok=True)
    for t, img in frames.url_frames(s.url, s.headers, start, frames.Sampler(every)):
        if t < start:
            continue
        if t > start + duration:
            break
        img.convert("RGB").save(f"{out}/{int(t):06d}.jpg", quality=85)
        print(t, flush=True)


if __name__ == "__main__":
    main()
