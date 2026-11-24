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


def MPT(i, elapsed):
    return Observation("mp", i, f"mp{i}", elapsed)


def test_timer_moves_start_back_to_when_they_began():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    # camera finds MP 2 at t=100, their timer already shows 0:30
    for k, t in enumerate([100.0, 101.0, 102.0]):
        tr.observe(t, MPT(2, 30 + k))
    assert plain(tr.json()) == [{"from": 0.0, "to": 70.0, "mp": 1}, {"from": 70.0, "to": None, "mp": 2}]


def test_one_odd_timer_reading_is_not_enough():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    tr.observe(100, MPT(2, 30))
    tr.observe(101, MP(2))
    tr.observe(102, MPT(2, 5))  # misread
    assert plain(tr.json())[-1] == {"from": 100.0, "to": None, "mp": 2}


def test_timer_read_later_still_moves_the_start():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    feed(tr, 100, [MP(2)] * 3)  # no timer while confirming
    tr.observe(110, MPT(2, 50))
    tr.observe(111, MPT(2, 51))
    assert plain(tr.json()) == [{"from": 0.0, "to": 60.0, "mp": 1}, {"from": 60.0, "to": None, "mp": 2}]
    assert not tr.wants_timer(2)  # pinned down: stop reading it


def test_timer_cannot_reach_before_previous_speech():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    for t in (100.0, 101.0, 102.0):
        tr.observe(t, MPT(2, int(t)))  # timer says they began at 0
    assert plain(tr.json())[-1]["from"] == 1.0


def test_cut_microphone_ends_speech_at_once():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    for k in range(20):
        tr.audio(10 + k * 0.5, 0.5, True)
    assert tr.audio(20.0, 0.5, False, silent=True)
    assert plain(tr.json()) == [{"from": 0.0, "to": 20.0, "mp": 1}]
    assert tr.current is None


def test_five_quiet_seconds_end_speech_short_pause_does_not():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    for k in range(6):
        tr.audio(10 + k * 0.5, 0.5, False)  # 3 s pause
    tr.audio(13, 0.5, True)
    assert tr.open is not None
    changed = [tr.audio(20 + k * 0.5, 0.5, False) for k in range(10)]
    assert changed[-1] and not any(changed[:-1])
    assert plain(tr.json()) == [{"from": 0.0, "to": 20.0, "mp": 1}]


def test_caption_still_up_after_cut_does_not_restart():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    tr.audio(20, 0.5, False, silent=True)
    feed(tr, 21, [MP(1)] * 5)
    assert plain(tr.json()) == [{"from": 0.0, "to": 20.0, "mp": 1}]


def test_cut_speaker_resumes_when_timer_still_counts_from_their_start():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    tr.observe(4.5, MPT(1, 4))
    tr.observe(5.5, MPT(1, 5))  # anchored at ~0.5 -> start stays 0
    tr.audio(20, 0.5, False, silent=True)
    tr.audio(22, 0.5, True)
    tr.observe(30, MPT(1, 30))
    assert plain(tr.json()) == [{"from": 0.0, "to": None, "mp": 1}]


def test_new_speaker_after_cut_starts_after_the_cut():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    tr.audio(20, 0.5, False, silent=True)
    tr.audio(25, 0.5, True)
    for t in (40.0, 41.0, 42.0):
        tr.observe(t, MPT(2, int(t - 30)))  # began at 30, after the cut at 20
    assert plain(tr.json()) == [{"from": 0.0, "to": 20.0, "mp": 1}, {"from": 30.0, "to": None, "mp": 2}]


def test_timer_before_the_cut_is_clamped_to_the_cut():
    tr = Tracker()
    feed(tr, 0, [MP(1)] * 3)
    tr.audio(20, 0.5, False, silent=True)
    tr.audio(25, 0.5, True)
    for t in (40.0, 41.0, 42.0):
        tr.observe(t, MPT(2, int(t - 10)))  # timer says they began at 10, but MP 1 spoke until 20
    assert plain(tr.json())[-1] == {"from": 20.0, "to": None, "mp": 2}
