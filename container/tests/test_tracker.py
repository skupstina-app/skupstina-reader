from reader.match import Observation
from reader.tracker import Tracker


def plain(x):
    return [{k: v for k, v in d.items() if k != "talk"} for d in x]



MP = lambda i: Observation("mp", i, f"mp{i}")
OTHER = lambda s: Observation("other", None, s)
NONE = Observation("none")


def feed(tr, start, obs_list, step=1.5):
    for i, o in enumerate(obs_list):
        tr.observe(start + i * step, o)


def test_needs_three_reads_and_backdates():
    tr = Tracker()
    feed(tr, 100, [MP(30), MP(30)])
    assert plain(tr.json()) == []
    feed(tr, 103, [MP(30)])
    assert plain(tr.json()) == [{"from": 100.0, "to": None, "mp": 30}]


def test_none_readings_do_not_break_or_end():
    tr = Tracker()
    feed(tr, 0, [MP(1), NONE, MP(1), NONE, MP(1)])
    assert plain(tr.json()) == [{"from": 0.0, "to": None, "mp": 1}]
    feed(tr, 10, [NONE] * 100)  # caption gone, speech goes on
    assert tr.open is not None


def test_switch_closes_previous_at_first_read_of_new():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    feed(tr, 60, [MP(2), MP(1), MP(2), MP(2), MP(2)])  # a stray read resets the count
    assert plain(tr.json()) == [{"from": 0.0, "to": 63.0, "mp": 1}, {"from": 63.0, "to": None, "mp": 2}]


def test_non_mp_speaker_closes_segment():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    feed(tr, 30, [OTHER("Милош Вучевић"), OTHER("Милош Вучеви"), OTHER("Милош Вучевић")])
    assert plain(tr.json()) == [{"from": 0.0, "to": 30.0, "mp": 1}]
    feed(tr, 40, [MP(1)] * 3)
    assert plain(tr.json())[-1] == {"from": 40.0, "to": None, "mp": 1}


def test_max_speech_and_end():
    tr = Tracker(max_speech=100)
    feed(tr, 0, [MP(1)] * 3)
    tr.observe(150, NONE)
    assert plain(tr.json()) == [{"from": 0.0, "to": 100.0, "mp": 1}]
    feed(tr, 200, [MP(2)] * 3)
    tr.end(250)
    assert plain(tr.json())[-1] == {"from": 200.0, "to": 250.0, "mp": 2}


def test_restore_only_same_stream_when_empty():
    tr = Tracker()
    tr.reset("abc")
    assert tr.restore("abc", [{"from": 1, "to": None, "mp": 5}])
    assert tr.current == ("mp", 5)
    assert not tr.restore("abc", [])
    t2 = Tracker()
    t2.reset("xyz")
    assert not t2.restore("abc", [{"from": 1, "to": 2, "mp": 5}])


def test_stray_other_name_does_not_reset_new_speaker():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    feed(tr, 60, [MP(2), MP(7), MP(2), NONE, MP(2)])  # one misread as someone else in the middle
    assert plain(tr.json()) == [{"from": 0.0, "to": 60.0, "mp": 1}, {"from": 60.0, "to": None, "mp": 2}]


def test_old_reads_expire():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    feed(tr, 60, [MP(2), MP(2)])
    feed(tr, 200, [MP(2)])  # minutes later: not the same caption
    assert plain(tr.json()) == [{"from": 0.0, "to": None, "mp": 1}]
