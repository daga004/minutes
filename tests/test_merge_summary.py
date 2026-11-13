from minutes import merge, pipeline, summarize
from minutes import session as sess
from minutes.asr.base import Segment


def _write(s, backend, side, idx, segs):
    sess.atomic_write_json(s.transcript_dir(backend, side) / f"{idx:05d}.json",
                           {"segments": [Segment(*x).to_dict() for x in segs]})


def test_interleaves_by_absolute_time(cfg):
    s = sess.create(cfg)
    s.set_state(sess.RECORDED, chunk_seconds=30, streams={
        "me": {"started_epoch": 1000.0}, "them": {"started_epoch": 1000.5}})
    _write(s, "x", "me", 0, [(1, 4, "hello"), (20, 25, "how are you")])
    _write(s, "x", "them", 0, [(5, 9, "hi there")])           # t = 5.5
    _write(s, "x", "them", 1, [(2, 6, "I am good")])          # t = 32.5
    _write(s, "x", "me", 1, [(10, 12, "great"), (14, 15, "  ")])  # t = 40, blank dropped
    lines = merge.collect(s, "x", 30, merge.stream_offsets(s.read()))
    assert [(round(l.t, 1), l.side, l.text) for l in lines] == [
        (1.0, "me", "hello"), (5.5, "them", "hi there"), (20.0, "me", "how are you"),
        (32.5, "them", "I am good"), (40.0, "me", "great")]


def test_coalesce_and_render(cfg):
    s = sess.create(cfg)
    s.set_state(sess.RECORDED, chunk_seconds=30, duration_s=3725)
    _write(s, "x", "me", 0, [(0, 1, "a"), (2, 3, "b")])
    _write(s, "x", "them", 124, [(0, 1, "c")])  # t = 3720 -> 1:02:00
    path = merge.run(cfg, s, backend="x")
    md = path.read_text()
    assert "**[00:00] Me:** a b" in md
    assert "**[1:02:00] Them:** c" in md
    assert s.read()["stages"]["merge"]["lines"] == 3


def test_missing_chunks_warned(cfg):
    s = sess.create(cfg)
    (s.audio_dir("me") / "00000.flac").write_bytes(b"x")
    s.set_state(sess.RECORDED, chunk_seconds=30)
    md = merge.run(cfg, s, backend="x").read_text()
    assert "1 chunk(s) not transcribed: me/00000" in md


class FakeSummarizer:
    name = "fake"

    def __init__(self):
        self.seen = None

    def summarize(self, transcript_md):
        self.seen = transcript_md
        return "## Summary\nok\n"


def test_summary_written(cfg):
    s = sess.create(cfg)
    s.set_state(sess.RECORDED, chunk_seconds=30)
    _write(s, "x", "me", 0, [(0, 1, "namaste")])
    merge.run(cfg, s, backend="x")
    fake = FakeSummarizer()
    md = summarize.run(cfg, s, summarizer=fake).summary_md.read_text()
    assert "namaste" in fake.seen and "## Summary\nok" in md
    assert s.read()["stages"]["summarize"]["status"] == "done"


def test_summary_without_speech_skips_backend(cfg):
    s = sess.create(cfg)
    s.set_state(sess.RECORDED, chunk_seconds=30)
    md = summarize.run(cfg, s, summarizer=None).summary_md.read_text()  # would fail if it called claude
    assert "No speech" in md


def test_process_end_to_end_with_null(cfg, monkeypatch):
    import subprocess
    s = sess.create(cfg)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=1",
                    "-ar", "16000", "-ac", "1", str(s.audio_dir("me") / "00000.flac")], check=True)
    s.set_state(sess.RECORDED, chunk_seconds=30)
    monkeypatch.setattr(summarize, "get", lambda name, cfg: FakeSummarizer())
    monkeypatch.setattr(pipeline, "notify", lambda *a, **k: None)
    assert pipeline.process(cfg, s, asr_backend="null")
    assert s.state == sess.SUMMARIZED
    assert "[me chunk 00000]" in s.transcript_md.read_text()
    assert s.summary_md.exists()


class TitledSummarizer(FakeSummarizer):
    def summarize(self, transcript_md):
        return "# Drone survey schedule with vendor\n\n## Summary\nok\n"


def _titled_session(cfg):
    s = sess.create(cfg)
    s.set_state(sess.RECORDED, chunk_seconds=30, started_at="2026-11-13T11:47:36+05:30")
    _write(s, "x", "me", 0, [(0, 1, "drone work today")])
    merge.run(cfg, s, backend="x")
    return s


def test_summary_titles_and_renames_session(cfg):
    s = _titled_session(cfg)
    base = s.id
    new = summarize.run(cfg, s, summarizer=TitledSummarizer())
    assert new.id == f"{base}_drone-survey-schedule-with-vendor"
    assert not s.path.exists() and new.path.exists()
    md = new.summary_md.read_text()
    assert md.startswith("# Drone survey schedule with vendor\n") and md.count("# Drone") == 1
    meta = new.read()
    assert meta["title"] == "Drone survey schedule with vendor" and meta["id"] == new.id
    assert meta["base_id"] == base
    # the time-based id still finds it, and re-summarizing replaces (not stacks) the slug
    assert sess.resolve(cfg, base).path == new.path
    again = summarize.run(cfg, new, summarizer=TitledSummarizer())
    assert again.id == new.id


def test_untitled_summary_keeps_name(cfg):
    s = _titled_session(cfg)
    new = summarize.run(cfg, s, summarizer=FakeSummarizer())
    assert new.id == s.id and new.summary_md.read_text().startswith("# Meeting minutes")


def test_base_id_resolves_despite_neighbours(cfg):
    from datetime import datetime
    when = datetime(2026, 11, 13, 11, 41)
    a, b = sess.create(cfg, when), sess.create(cfg, when)  # 2026-11-13_1141 and ..._1141-2
    a2 = sess.retitle(a, "Budget review")
    sess.retitle(b, "Hiring sync")
    assert sess.resolve(cfg, "2026-11-13_1141").path == a2.path


def test_slugify():
    assert sess.slugify("Drone work: plan & owners!") == "drone-work-plan-owners"
    assert sess.slugify("ड्रोन") == ""
    assert len(sess.slugify("word " * 40)) <= 60
