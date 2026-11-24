from reader.config import Config
from reader.hls import Seg
from reader.main import Backfill, Reader
from reader.match import Observation


def plain(x):
    return [{k: v for k, v in d.items() if k != "talk"} for d in x]


def reader():
    cfg = Config.from_env()
    cfg.backfill = False
    r = Reader(cfg)
    r.tracker.reset("vid")
    return r


def bf_for(r, end_seq=100, start_seq=0, workers=1):
    """History of [start_seq, end_seq) segments of 5 s, read by `workers` stretches."""
    b = Backfill(r, "vid", Seg(200, 5.0, "https://x.test/videoplayback/sq/200/file/seg.ts"), end_seq, start_seq, workers)
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
    c = bf_for(r).chunks[0]  # covers 0..500 s
    feed(c.tracker, 100, 1)
    feed(c.tracker, 300, 2)  # still speaking when the history ends
    c.last_t, c.done = 500, True
    feed(r.tracker, 520, 3)
    assert plain(r.merged()) == [
        {"from": 100.0, "to": 300.0, "mp": 1},
        {"from": 300.0, "to": 520.0, "mp": 2},
        {"from": 520.0, "to": None, "mp": 3},
    ]


def test_open_speech_runs_on_while_live_has_nobody_yet():
    r = reader()
    c = bf_for(r).chunks[0]
    feed(c.tracker, 450, 2)
    c.last_t = 499
    assert plain(r.merged()) == [{"from": 450.0, "to": 499.0, "mp": 2}]  # still reading: up to where it got
    c.done = True
    assert plain(r.merged()) == [{"from": 450.0, "to": None, "mp": 2}]  # done, nobody new live: still speaking


def test_unfinished_history_stops_at_what_it_has_read():
    r = reader()
    c = bf_for(r).chunks[0]
    feed(c.tracker, 100, 1)
    c.last_t = 160
    feed(r.tracker, 520, 3)
    assert plain(r.merged())[0] == {"from": 100.0, "to": 160.0, "mp": 1}


def test_restored_history_fills_until_backfill_reaches_it():
    r = reader()
    r.restored = [{"from": 441.0, "to": 19041.0, "mp": 1029}, {"from": 19041.0, "to": None, "mp": 1190}]
    c = bf_for(r, end_seq=4000).chunks[0]  # will cover 0..20000 s
    feed(c.tracker, 441, 1029)
    c.last_t = 1400
    feed(r.tracker, 19500, 7)
    got = plain(r.merged())
    assert got[0] == {"from": 441.0, "to": 1400.0, "mp": 1029}
    assert {"from": 19041.0, "to": 19500.0, "mp": 1190} in got
    assert got[-1] == {"from": 19500.0, "to": None, "mp": 7}
    assert all(sg["from"] != 441.0 or sg["to"] != 19041.0 for sg in got)  # the stale stretched one is gone


def test_resumed_backfill_keeps_history_before_where_it_restarted():
    r = reader()
    r.restored = [{"from": 441.0, "to": 900.0, "mp": 1029}, {"from": 900.0, "to": None, "mp": 1190}]
    b = bf_for(r, end_seq=4000, start_seq=200)  # resumes at 1000 s
    c = b.chunks[0]
    c.last_t = 1000.0
    got = plain(r.merged())
    assert got[0] == {"from": 441.0, "to": 900.0, "mp": 1029}
    assert got[1]["mp"] == 1190 and got[1]["from"] == 900.0
    feed(c.tracker, 1200, 7)
    c.last_t = 1300
    got = plain(r.merged())
    assert got[1] == {"from": 900.0, "to": 1200.0, "mp": 1190}
    assert got[2] == {"from": 1200.0, "to": 1300.0, "mp": 7}


def test_two_stretches_join_up():
    r = reader()
    b = bf_for(r, end_seq=1000, workers=2)  # 0..2500 s and 2500..5000 s
    a, c = b.chunks
    assert (a.start_t, a.end_t, c.start_t, c.end_t) == (0, 2500, 2500, 5000)
    feed(a.tracker, 2000, 1)  # still speaking at the end of the first stretch
    a.last_t, a.done = 2500, True
    feed(c.tracker, 2600, 2)  # the second stretch's first speaker ends it
    c.last_t = 3000
    got = plain(b.json())
    assert got[0] == {"from": 2000.0, "to": 2600.0, "mp": 1}
    assert got[1]["mp"] == 2
    assert b.last_t == 3000  # gap-free reach runs into the second stretch once the first is done


def test_break_at_start_of_next_stretch_ends_the_speech():
    r = reader()
    b = bf_for(r, end_seq=1000, workers=2)
    a, c = b.chunks
    feed(a.tracker, 2000, 1)
    a.last_t, a.done = 2500, True
    c.tracker.observe(2550, Observation("break"))
    c.tracker.observe(2551.5, Observation("break"))
    assert plain(b.json())[0] == {"from": 2000.0, "to": 2550.0, "mp": 1}
