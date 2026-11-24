import io
import math
import os

import numpy as np
import pytest
from PIL import Image

from reader import frames
from reader.main import voice_threshold
from reader.match import Observation
from reader.tracker import Tracker

MP = Observation("mp", 7, "x")


def opened(t=0.0, **kw):
    tr = Tracker(**kw)
    for i in range(3):
        tr.observe(t + i * 1.5, MP)
    return tr


def audio(tr, t, secs, voiced):
    for k in range(int(secs / 0.5)):
        tr.audio(t + k * 0.5, 0.5, voiced)
    return t + secs


def test_short_pauses_count_long_ones_dont():
    tr = opened(end_quiet=1e9)  # quiet-ends-the-speech switched off for this one
    t = audio(tr, 3, 20, True)      # 20 s talking
    t = audio(tr, t, 3, False)      # 3 s pause: still talking
    t = audio(tr, t, 10, True)      # 10 s talking
    t = audio(tr, t, 60, False)     # a minute of silence: not talking
    t = audio(tr, t, 5, True)       # 5 s more
    assert tr.json()[0]["talk"] == 38.0


def test_long_silence_ends_the_speech():
    tr = opened()
    t = audio(tr, 3, 20, True)
    audio(tr, t, 60, False)
    assert tr.json()[0]["to"] == 23.0 and tr.json()[0]["talk"] == 20.0


def test_no_audio_means_unknown_not_zero():
    assert opened().json()[0]["talk"] is None


def test_break_needs_two_frames_and_ends_at_the_first():
    tr = opened()
    tr.observe(100, Observation("break"))
    assert tr.open is not None                      # one odd frame: nothing happens
    tr.observe(101.5, Observation("none"))
    tr.observe(103, Observation("break"))
    assert tr.open is not None
    tr.observe(104.5, Observation("break"))
    assert tr.json()[0]["to"] == 103.0 and tr.current is None


def test_threshold_adapts_and_is_clamped():
    quiet_room = [-58.0] * 300 + [-25.0] * 1000     # floor -58: threshold -48
    assert voice_threshold(quiet_room, -42, 10) == -48
    noisy_room = [-30.0] * 1300                      # never ask for louder than -35
    assert voice_threshold(noisy_room, -42, 10) == -35
    digital_silence = [-120.0] * 300 + [-20.0] * 300
    assert voice_threshold(digital_silence, -42, 10) == -60
    assert voice_threshold([-50.0] * 10, -42, 10) == -42  # too little audio yet: fallback


SNAPS = os.path.join(os.path.dirname(__file__), "frames")


@pytest.mark.parametrize("name,expected", [("break.jpg", True), ("speaker.jpg", False)])
def test_break_screen_detection(name, expected):
    assert frames.is_break(Image.open(os.path.join(SNAPS, name))) is expected


def make_ts(tone_s=2.0, silence_s=2.0, rate=48000):
    import av

    buf = io.BytesIO()
    with av.open(buf, "w", format="mpegts") as c:
        st = c.add_stream("aac", rate=rate)
        st.layout = "mono"
        n = int(rate * (tone_s + silence_s))
        x = np.zeros(n, dtype=np.float32)
        k = int(rate * tone_s)
        x[:k] = 0.1 * np.sin(2 * math.pi * 220 * np.arange(k) / rate)  # about -23 dBFS
        x[k:] = np.random.default_rng(1).normal(0, 0.0005, n - k)      # room hiss, about -66 dBFS
        for i in range(0, n, 1024):
            fr = av.AudioFrame.from_ndarray(x[None, i : i + 1024], format="fltp", layout="mono")
            fr.sample_rate = rate
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode(None):
            c.mux(pkt)
    return buf.getvalue()


def test_segment_audio_loudness():
    w = frames.segment_audio(make_ts(), 100.0)
    assert len(w) >= 7
    assert w[0][0] == pytest.approx(100.0, abs=0.1)
    loud = [db for t, _, db in w if t < 101.8]
    quiet = [db for t, _, db in w if t > 102.3]
    assert min(loud) > -30 and max(quiet) < -55
