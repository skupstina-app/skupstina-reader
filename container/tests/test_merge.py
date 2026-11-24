from reader.config import Config
from reader.hls import Seg
from reader.main import Backfill, Reader
from reader.match import Observation


def reader(**kw):
    cfg = Config.from_env()
    cfg.backfill = False
    r = Reader(cfg)
    r.tracker.reset("vid")
    return r


def bf_for(r, end_seq=100):
    b = Backfill(r, "vid", Seg(200, 5.0, "https://x.test/videoplayback/sq/200/file/seg.ts"), end_seq)
    r.backfill = b
    return b


def feed(tr, t0, mp, n=3):
    for i in range(n):
        tr.observe(t0 + i * 1.5, Observation("mp", mp, "x"))


def test_url_rewrites_sequence():
    r = reader()
    assert bf_for(r).url(7) == "https://x.test/videoplayback/sq/7/file/seg.ts"


def test_history_is_cut_where_live_has_its_first_speaker():
    r = reader()
    b = bf_for(r)  # backfill covers 0..500 s
    feed(b.tracker, 100, 1)
    feed(b.tracker, 300, 2)  # still speaking when the backfill reaches its end
    b.tracker.end(b.end_t)
    b.done = True
    feed(r.tracker, 520, 3)
    assert r.merged() == [
        {"from": 100.0, "to": 300.0, "mp": 1},
        {"from": 300.0, "to": 500.0, "mp": 2},
        {"from": 520.0, "to": None, "mp": 3},
    ]


def test_open_speech_runs_on_while_live_has_nobody_yet():
    r = reader()
    b = bf_for(r)
    feed(b.tracker, 450, 2)
    b.last_t = 499
    assert r.merged() == [{"from": 450.0, "to": 499.0, "mp": 2}]  # still backfilling: up to where it read
    b.done = True
    assert r.merged() == [{"from": 450.0, "to": None, "mp": 2}]  # done, nobody new live: still speaking


def test_unfinished_backfill_stops_at_what_it_has_read():
    r = reader()
    b = bf_for(r)
    feed(b.tracker, 100, 1)
    b.last_t = 160
    feed(r.tracker, 520, 3)
    assert r.merged()[0] == {"from": 100.0, "to": 160.0, "mp": 1}
