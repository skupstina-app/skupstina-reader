"""Caption crop -> black-on-white text mask -> Tesseract lines.

Binarisation ported from Neleceni (src/shared/binarize.ts, glyphs.ts, src/content/capture.ts),
MIT License, Copyright (c) 2026 Goran Grbic, https://github.com/Tpojka/neleceni.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .match import OcrLine

MAX_CHROMA = 40  # white text is grey: max(RGB) - min(RGB) at most this
REF_HEIGHT = 216  # the 80-100 % strip of a 1080p frame
THIN_RUN = 3
RING_SHARE = 0.3
OPEN_RADIUS = 1
TARGET_HEIGHT = 160  # Tesseract likes ~30-60 px letters
PAD = 12
SIG_W, SIG_H = 48, 8


@dataclass
class Capture:
    image: Image.Image  # black text on white, padded
    ink: float  # share of pixels kept as text
    signature: np.ndarray


def _keep_glyphs(mask: np.ndarray) -> int:
    """Drop blobs that are too big, too small, or cut by the top/bottom edge (desks, papers...)."""
    h = mask.shape[0]
    max_h, max_w = round(h * 0.5), round(h * 0.75)
    min_area = max(4, round(h * h * 0.0004))
    lab, n = ndimage.label(mask, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = sl
        bh, bw = ys.stop - ys.start, xs.stop - xs.start
        cut = ys.start == 0 or ys.stop == h
        keep[i] = not (bh > max_h or bw > max_w or cut or sizes[i] < min_area)
    mask[:] = keep[lab]
    return int(mask.sum())


def _thin_run_share(mask: np.ndarray) -> float:
    thin = max(1, round(THIN_RUN * mask.shape[0] / REF_HEIGHT))
    m = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if not len(starts):
        return 0.0
    lens = ends[:, 1] - starts[:, 1]
    return float((lens <= thin).sum() / len(lens))


def binarize(rgb: np.ndarray, white: int) -> tuple[np.ndarray, int, bool]:
    lo = rgb.min(axis=2)
    hi = rgb.max(axis=2)
    mask = (lo >= white) & (hi.astype(np.int16) - lo <= MAX_CHROMA)
    ink = _keep_glyphs(mask)
    ringed = _thin_run_share(mask) >= RING_SHARE
    if ringed:
        r = max(1, round(OPEN_RADIUS * mask.shape[0] / REF_HEIGHT))
        st = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        mask = ndimage.binary_erosion(mask, st, border_value=1)
        mask = ndimage.binary_dilation(mask, st, border_value=0)
        ink = _keep_glyphs(mask)
    return mask, ink, ringed


def capture(frame: Image.Image, region: tuple[float, float, float, float], white: int) -> Capture:
    """region = (x, y, w, h) in percent of the frame."""
    W, H = frame.size
    x, y, w, h = region
    sx, sy = round(x / 100 * W), round(y / 100 * H)
    sw, sh = max(1, round(w / 100 * W)), max(1, round(h / 100 * H))
    crop = frame.convert("RGB").crop((sx, sy, sx + sw, sy + sh))
    scale = min(3.0, max(1.0, TARGET_HEIGHT / sh))
    if scale != 1:
        crop = crop.resize((round(sw * scale), round(sh * scale)), Image.BICUBIC)
    rgb = np.asarray(crop)
    mask, ink, _ = binarize(rgb, white)
    dh, dw = mask.shape
    out = np.full((dh + 2 * PAD, dw + 2 * PAD), 255, dtype=np.uint8)
    out[PAD : PAD + dh, PAD : PAD + dw][mask] = 0
    # coarse signature to skip OCR when nothing changed
    ys = np.minimum(SIG_H - 1, (np.arange(dh) * SIG_H) // dh)
    xs = np.minimum(SIG_W - 1, (np.arange(dw) * SIG_W) // dw)
    sig = np.zeros((SIG_H, SIG_W), dtype=np.float64)
    np.add.at(sig, (ys[:, None].repeat(dw, 1)[mask], xs[None, :].repeat(dh, 0)[mask]), 1)
    cell = dh * dw / (SIG_W * SIG_H)
    sig = np.minimum(255, np.round(sig / cell * 255)).astype(np.uint8).ravel()
    return Capture(Image.fromarray(out, "L"), ink / mask.size, sig)


def signature_distance(a: np.ndarray | None, b: np.ndarray) -> float:
    if a is None or a.shape != b.shape:
        return 1.0
    return float(np.abs(a.astype(np.int16) - b).sum() / (a.size * 255))


def ocr_lines(img: Image.Image, langs: str) -> list[OcrLine]:
    """Tesseract (LSTM, automatic page segmentation) -> text lines with confidence and box."""
    import pytesseract

    d = pytesseract.image_to_data(img, lang=langs, config="--oem 1 --psm 3", output_type=pytesseract.Output.DICT)
    groups: dict[tuple, list[int]] = {}
    for i, word in enumerate(d["text"]):
        if word.strip() and float(d["conf"][i]) >= 0:
            groups.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
    lines = []
    for idx in groups.values():
        words = [d["text"][i] for i in idx]
        w = [max(1, len(t)) for t in words]
        conf = sum(float(d["conf"][i]) * k for i, k in zip(idx, w)) / sum(w)
        lines.append(OcrLine(
            " ".join(words), conf,
            min(d["left"][i] for i in idx), min(d["top"][i] for i in idx),
            max(d["left"][i] + d["width"][i] for i in idx), max(d["top"][i] + d["height"][i] for i in idx),
        ))
    return lines
