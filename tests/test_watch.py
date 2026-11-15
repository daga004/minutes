"""Call detection as a pure state machine: streams + window titles in, actions out."""

from minutes.watch import Detector, Stream, describe

MEET = Stream("Helium", "helium")
DICTATION = Stream("pw-record", "pw-record")
OWN = Stream("minutes-me", "ffmpeg")


def run(det, seconds, streams, titles=(), rec=None, start=0, every=3):
    out = []
    for t in range(start, start + seconds, every):
        out += [(t, a[0]) for a in det.step(t, streams, list(titles), rec)]
    return out


def test_asks_once_after_min_seconds():
    det = Detector(min_seconds=15)
    acts = run(det, 60, [MEET])
    assert acts == [(15, "ask")]


def test_dictation_and_own_recorder_are_not_calls():
    det = Detector(min_seconds=15)
    assert run(det, 60, [DICTATION, OWN]) == []


def test_short_mic_use_is_not_a_call():
    det = Detector(min_seconds=15)
    assert run(det, 9, [MEET]) == [] and run(det, 30, [], start=9) == []


def test_no_question_while_recording():
    det = Detector(min_seconds=15)
    assert run(det, 60, [MEET], rec={"session_id": "s1", "trigger": "manual"}) == []


def test_stops_a_call_recording_after_the_call_ends():
    det = Detector(min_seconds=15, end_grace=60)
    rec = {"session_id": "s1", "trigger": "detector"}
    run(det, 120, [MEET], rec=rec)
    acts = run(det, 120, [], rec=rec, start=120)
    assert acts == [(177, "stop")]  # 60 s after the mic was last heard (t=117), and only once


def test_open_meet_window_keeps_the_call_alive_when_muted():
    det = Detector(min_seconds=15, end_grace=60)
    rec = {"session_id": "s1", "trigger": "join"}
    run(det, 60, [MEET], rec=rec)
    assert run(det, 300, [], titles=["Meet - abc-defg-hij - Helium"], rec=rec, start=60) == []


def test_manual_recording_without_a_call_is_never_stopped():
    det = Detector(min_seconds=15, end_grace=60)
    assert run(det, 600, [], rec={"session_id": "s2", "trigger": "manual"}) == []


def test_asks_again_for_the_next_call():
    det = Detector(min_seconds=15, end_grace=60)
    run(det, 60, [MEET])
    run(det, 90, [], start=60)  # call over
    assert run(det, 30, [MEET], start=150) == [(165, "ask")]


def test_describe_names_the_kind_of_call():
    assert describe("Helium", ["Meet - abc-defg-hij - Helium"], None)[1].startswith("Google Meet")
    assert describe("Slack", ["Huddle with Ravi"], None)[1].startswith("Slack huddle")
    assert describe("Helium", ["WhatsApp"], None)[1].startswith("WhatsApp call")
