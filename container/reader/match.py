"""Name matching and name-line picking.

Ported from Neleceni (src/shared/match.ts), MIT License, Copyright (c) 2026 Goran Grbic,
https://github.com/Tpojka/neleceni. Changed: instead of a short list of names to mute, a
reading is matched against every MP and reported as one MP id.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import product

_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "dj", "е": "e", "ж": "z", "з": "z", "и": "i",
    "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "ћ": "c", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "c", "џ": "dz", "ш": "s",
}


def normalize(text: str) -> str:
    """'Александар Вучић' and 'Aleksandar Vučić' both become 'aleksandar vucic'."""
    s = "".join(_CYR.get(ch, ch) for ch in (text or "").lower()).replace("đ", "dj")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return re.sub(r"[^a-z]+", " ", s).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i]
        for j in range(1, len(b) + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1])))
        prev = cur
    return prev[-1]


def substring_distance(hay: str, needle: str) -> int:
    """Smallest edit distance between `needle` and any substring of `hay` (Sellers)."""
    if not needle:
        return 0
    prev = [0] * (len(hay) + 1)
    for i in range(1, len(needle) + 1):
        cur = [i]
        for j in range(1, len(hay) + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (needle[i - 1] != hay[j - 1])))
        prev = cur
    return min(prev)


def allowed_errors(n: int) -> int:
    return 2 if n >= 8 else 1 if n >= 4 else 0


def line_matches_name(line: str, name: str) -> bool:
    """True when `line` shows `name`.

    Each part of the name must be its own word of the line (a different word for each part), read whole with a
    few OCR errors allowed, or as a clear piece of it ("А екса · р Вучић"); a long first name may be badly
    garbled ("АпењСандар Вучић"), but the surname must really be read. Two parts glued into one word
    ("АлександарВучић") count for both. At least one part must be read whole.
    """
    tokens = [t for t in normalize(name).split(" ") if t]
    words = [w for w in normalize(line).split(" ") if w]
    if not tokens or not words:
        return False
    last = len(tokens) - 1
    pairs = list(zip(tokens, tokens[1:]))
    options: list[list] = []
    wholes: list[set] = []
    for i, t in enumerate(tokens):
        whole: set = set()
        other: set = set()
        for j, w in enumerate(words):
            if levenshtein(w, t) <= allowed_errors(len(t)):
                whole.add(j)
            elif (len(w) >= max(3, -(-len(t) * 4 // 10)) and len(w) < len(t)
                  and substring_distance(t, w) <= (2 if len(w) >= 6 else allowed_errors(len(w)))):
                other.add(j)  # a clear piece of the word
            elif i != last and len(t) >= 6 and levenshtein(w, t) <= (len(t) * 4) // 10:
                other.add(j)  # a long first name read badly
            for a, b in pairs:
                if t in (a, b) and levenshtein(w, a + b) <= allowed_errors(len(a + b)):
                    whole.add(("glued", j))
        if not whole and not other:
            return False
        options.append(sorted(whole | other, key=str))
        wholes.append(whole)
    for combo in product(*options):
        plain = [c for c in combo if not isinstance(c, tuple)]
        if len(plain) == len(set(plain)) and any(c in wholes[i] for i, c in enumerate(combo)):
            return True
    return False


@dataclass
class OcrLine:
    text: str
    confidence: float  # 0-100
    x0: int
    y0: int
    x1: int
    y1: int


def merge_rows(lines: list[OcrLine]) -> list[OcrLine]:
    """Join pieces of one text row that Tesseract split into blocks."""
    def centre(l: OcrLine) -> float:
        return (l.y0 + l.y1) / 2

    def same_row(a: OcrLine, b: OcrLine) -> bool:
        return abs(centre(a) - centre(b)) <= 0.4 * min(a.y1 - a.y0, b.y1 - b.y0)

    rows: list[list[OcrLine]] = []
    for l in sorted((l for l in lines if re.search(r"[^\W\d_]", l.text)), key=centre):
        row = next((r for r in rows if same_row(r[0], l)), None)
        if row is not None:
            row.append(l)
        else:
            rows.append([l])
    out = []
    for r in rows:
        if len(r) == 1:
            out.append(r[0])
            continue
        r.sort(key=lambda l: l.x0)
        w = [max(1, len(l.text)) for l in r]
        out.append(OcrLine(
            " ".join(l.text for l in r),
            sum(l.confidence * k for l, k in zip(r, w)) / sum(w),
            min(l.x0 for l in r), min(l.y0 for l in r), max(l.x1 for l in r), max(l.y1 for l in r),
        ))
    return out


JUNK_CONFIDENCE = 10
_GROUP_DASH = re.compile(r"\s[-–—]\s")
_EDGE = re.compile(r"^[\W\d_]+|[\W\d_]+$")


def trim_noise(s: str) -> str:
    return _EDGE.sub("", s)


def clean_line(text: str) -> str:
    return " ".join(w for w in (trim_noise(w) for w in text.split()) if len(re.findall(r"[^\W\d_]", w)) >= 2)


def name_words(text: str) -> list[str]:
    return [w for w in normalize(text).split(" ") if len(w) >= 3]


def capitalised_words(text: str) -> list[str]:
    return [w for w in re.split(r"[\W\d_]+", text) if len(w) >= 3 and w[0].isupper()]


def is_group_line(text: str) -> bool:
    return bool(_GROUP_DASH.search(text)) or len(name_words(text)) >= 4


def pick_speaker_line(lines: list[OcrLine]) -> OcrLine | None:
    """The name line is the line right above the parliamentary-group line (or the top line)."""
    rows = merge_rows([l for l in lines if l.confidence >= JUNK_CONFIDENCE])
    rows = [OcrLine(trim_noise(l.text.strip()), l.confidence, l.x0, l.y0, l.x1, l.y1) for l in rows]
    s = sorted((l for l in rows if len(name_words(l.text)) >= 1), key=lambda l: l.y0)

    def above(i: int) -> bool:
        return i > 0 and s[i].y0 - s[i - 1].y1 <= s[i].y1 - s[i].y0

    g = next((i for i, l in enumerate(s) if is_group_line(l.text)), -1)
    if g == -1:
        g = next((i for i, l in enumerate(s) if len(name_words(l.text)) >= 3 and above(i)), -1)
    if g == -1:
        return s[0] if s else None
    group = s[g]
    h = group.y1 - group.y0
    for i in range(g - 1, -1, -1):
        l = s[i]
        if is_group_line(l.text):
            continue
        return l if group.y0 - l.y1 <= h else None
    return None


def has_words_after_name(line: str, name: str) -> bool:
    words = [w for w in normalize(line).split(" ") if w]
    parts = [t for t in normalize(name).split(" ") if t]
    if not parts:
        return False
    last = parts[-1]
    end = -1
    for i, w in enumerate(words):
        if substring_distance(w, last) <= allowed_errors(len(last)):
            end = i
    return end >= 0 and any(len(w) >= 4 for w in words[end + 1 :])


def mentions_surname(line: str, name: str) -> bool:
    parts = [t for t in normalize(name).split(" ") if t]
    compact = normalize(line).replace(" ", "")
    return bool(parts) and len(parts[-1]) >= 4 and substring_distance(compact, parts[-1]) <= allowed_errors(len(parts[-1]))


def similar_lines(a: str, b: str) -> bool:
    ca, cb = normalize(a).replace(" ", ""), normalize(b).replace(" ", "")
    longest = max(len(ca), len(cb))
    return longest == 0 or levenshtein(ca, cb) <= (longest * 35) // 100


# ---- MPs -------------------------------------------------------------------


@dataclass(frozen=True)
class MP:
    id: int
    names: tuple[str, ...]  # display spellings (Latin and Cyrillic)
    keys: tuple[str, ...]  # normalized


def build_index(mps: list[dict]) -> list[MP]:
    """`mps` is the /api/mps payload: [{id, l: Latin name, c: Cyrillic name, ...}]."""
    out = []
    for m in mps:
        names = tuple(n for n in {m.get("l") or "", m.get("c") or ""} if len(normalize(n).split()) >= 2)
        if names and m.get("id") is not None:
            out.append(MP(int(m["id"]), names, tuple({normalize(n) for n in names})))
    return out


def _distance(line: str, mp: MP) -> int:
    compact = normalize(line).replace(" ", "")
    return min(sum(substring_distance(compact, t) for t in k.split()) for k in mp.keys)


@dataclass
class Observation:
    kind: str  # "none" | "mp" | "other"
    mp: int | None = None
    line: str = ""
    elapsed: int | None = None  # the speech timer in the same frame (seconds), when it was read


# Surnames that are also in a parliamentary group's name ("Александар Вучић – Србија ...").
# For these MPs the line read must have another row under it: the name line sits above the group line,
# so a name read on the bottom row is the group line with the real speaker's name missed.
IN_GROUP_NAMES = {"vucic"}


def _has_row_below(line: OcrLine, lines: list[OcrLine]) -> bool:
    h = max(1, line.y1 - line.y0)
    rows = merge_rows([l for l in lines if l.confidence >= JUNK_CONFIDENCE])
    return any(r.y0 >= line.y1 - h * 0.3 and len(name_words(r.text)) >= 1 for r in rows)


def classify(lines: list[OcrLine], index: list[MP], min_confidence: float = 35, current: int | None = None) -> Observation:
    """One reading of the caption: an MP, another (non-MP) speaker, or nothing usable.

    `current` is the MP speaking now: a garbled reading of their surname is never "someone else".
    """
    line = pick_speaker_line(lines)
    if line is None:
        return Observation("none")
    hits = [mp for mp in index if any(line_matches_name(line.text, n) for n in mp.keys)]
    if hits:
        scored = sorted(((_distance(line.text, mp), mp.id, mp) for mp in hits), key=lambda x: (x[0], x[1]))
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return Observation("none")  # two MPs fit equally well: don't guess
        best = scored[0][2]
        if any(k.split()[-1] in IN_GROUP_NAMES for k in best.keys) and not _has_row_below(line, lines):
            return Observation("none")  # the group line, read without the name line above it
        # The best name followed by more words is a group line whose dash OCR lost ("Александар Вучић
        # Србија"): it tells us nothing about who is speaking.
        if any(has_words_after_name(line.text, n) for n in best.keys):
            return Observation("none")
        return Observation("mp", best.id, clean_line(line.text))
    # A garbled reading of the current speaker's name ("Хиоса ћр ву Вучић") must not count as
    # someone else, or it could end their segment while they are still speaking.
    if current is not None and any(mentions_surname(line.text, k) or similar_lines(line.text, k)
                                   for mp in index if mp.id == current for k in mp.keys):
        return Observation("none")
    # A clean, name-shaped line that is no MP: a minister, a guest, the President...
    if line.confidence >= min_confidence and len(capitalised_words(line.text)) >= 2 and not is_group_line(line.text):
        return Observation("other", None, clean_line(line.text))
    return Observation("none")
