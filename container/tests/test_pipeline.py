"""End to end on synthetic frames: caption strip -> binarize -> Tesseract -> classify."""

import os
import random
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from reader import caption
from reader.frames import Sampler
from reader.hls import parse_playlist
from reader.match import build_index, classify

FONT = next((p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"] if os.path.exists(p)), None)
needs_tesseract = pytest.mark.skipif(not shutil.which("tesseract") or not FONT, reason="tesseract or font missing")

MPS = build_index([
    {"id": 30, "l": "Aleksandar Vučić", "c": "Александар Вучић"},
    {"id": 1029, "l": "Milenko Jovanov", "c": "Миленко Јованов"},
    {"id": 1190, "l": "Ana Brnabić", "c": "Ана Брнабић"},
])


def frame(name: str, group: str, size=(1280, 720), large=False, seed=1) -> Image.Image:
    """A chamber-ish frame with a white, dark-outlined two-line caption, like the broadcast."""
    rnd = random.Random(seed)
    W, H = size
    img = Image.new("RGB", size, (88, 70, 60))
    d = ImageDraw.Draw(img)
    for _ in range(400):  # busy background: wood, suits, lit faces, monitors
        x, y = rnd.randrange(W), rnd.randrange(H)
        c = rnd.choice([(120, 90, 70), (40, 40, 50), (200, 190, 170), (218, 214, 205)])
        d.rectangle([x, y, x + rnd.randrange(10, 120), y + rnd.randrange(10, 80)], fill=c)
    for _ in range(6):  # sheets of white paper: too big to be letters
        x, y = rnd.randrange(W), rnd.randrange(H)
        d.rectangle([x, y, x + int(0.12 * W), y + int(0.09 * H)], fill=(250, 250, 248))
    s = H / 720
    f1 = ImageFont.truetype(FONT, int((34 if large else 26) * s))
    f2 = ImageFont.truetype(FONT, int((24 if large else 18) * s))
    x0 = int((0.08 if large else 0.03) * W)
    y1, y2 = (int(0.83 * H), int(0.895 * H)) if large else (int(0.905 * H), int(0.955 * H))
    for txt, y, f in ((name, y1, f1), (group, y2, f2)):
        d.text((x0, y), txt, font=f, fill=(255, 255, 255), stroke_width=max(2, int(2 * s)), stroke_fill=(20, 20, 20))
    return img


def read(img):
    cap = caption.capture(img, (0, 80, 90, 20), 225)
    return classify(caption.ocr_lines(cap.image, "srp+srp_latn"), MPS, 35), cap


@needs_tesseract
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080)])
def test_reads_name_line_not_group(large, size):
    obs, cap = read(frame("Ана Брнабић", "Александар Вучић - Србија побеђује", size, large))
    assert cap.ink > 0.002
    assert (obs.kind, obs.mp) == ("mp", 1190), obs
    obs, _ = read(frame("Александар Вучић", "Александар Вучић - Србија побеђује", size, large, seed=2))
    assert (obs.kind, obs.mp) == ("mp", 30), obs


@needs_tesseract
def test_latin_caption_and_non_mp():
    obs, _ = read(frame("Milenko Jovanov", "Aleksandar Vučić - Srbija pobeđuje", seed=3))
    assert (obs.kind, obs.mp) == ("mp", 1029), obs
    obs, _ = read(frame("Милош Вучевић", "Председник Владе", seed=4))
    assert obs.kind in ("other", "none")  # never an MP


def test_no_caption_no_ink():
    img = Image.new("RGB", (1280, 720), (60, 50, 40))
    assert caption.capture(img, (0, 80, 90, 20), 225).ink < 0.002


def test_big_white_shapes_are_dropped():
    img = Image.new("RGB", (1280, 720), (60, 50, 40))
    ImageDraw.Draw(img).rectangle([100, 600, 700, 700], fill=(255, 255, 255))  # a sheet of paper
    assert caption.capture(img, (0, 80, 90, 20), 225).ink < 0.002


def test_playlist_and_sampler():
    text = "#EXTM3U\n#EXT-X-TARGETDURATION:5\n#EXT-X-MEDIA-SEQUENCE:1200\n#EXTINF:5.0,\nseg/1200.ts\n#EXTINF:5.0,\nseg/1201.ts\n"
    segs, ended = parse_playlist(text, "https://x.test/a/index.m3u8")
    assert [(s.seq, s.duration, s.uri) for s in segs] == [(1200, 5.0, "https://x.test/a/seg/1200.ts"), (1201, 5.0, "https://x.test/a/seg/1201.ts")]
    assert not ended
    sm = Sampler(1.5)
    picked = [t for t in [i / 30 for i in range(0, 300)] if sm.due(t)]
    assert len(picked) == 7 and picked[0] == 0 and all(abs(b - a - 1.5) < 0.05 for a, b in zip(picked, picked[1:]))
