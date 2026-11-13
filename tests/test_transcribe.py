import json
import subprocess

import pytest

from minutes import asr, transcribe
from minutes import session as sess
from minutes.asr.null import NullTranscriber


def _tone(path, seconds=2, volume=0.3, freq=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    src = f"sine=frequency={freq}:duration={seconds}" if volume else f"anullsrc=r=16000:cl=mono:d={seconds}"
    af = ["-af", f"volume={volume}"] if volume else []
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src, *af,
                    "-ac", "1", "-ar", "16000", str(path)], check=True)


@pytest.fixture
def recorded(cfg):
    s = sess.create(cfg)
    for i in range(2):
        _tone(s.audio_dir("me") / f"{i:05d}.flac")
    _tone(s.audio_dir("them") / "00000.flac")
    _tone(s.audio_dir("them") / "00001.flac", volume=0)  # digital silence
    s.set_state(sess.RECORDED, chunk_seconds=cfg.chunk_seconds)
    return s


def test_registry():
    assert {"null", "superwhisper", "tara"} <= set(asr.available())
    with pytest.raises(LookupError):
        asr.get("nope", None)


def test_peak_detection(tmp_path):
    loud, quiet = tmp_path / "loud.flac", tmp_path / "quiet.flac"
    _tone(loud, volume=0.3)
    _tone(quiet, volume=0)
    assert -35 < transcribe.peak_db(loud) < -20  # lavfi sine is -18 dBFS, x0.3 = -28.5
    assert transcribe.peak_db(quiet) < -80  # digital silence reads -91 dB


def test_transcribe_skips_silence_and_resumes(cfg, recorded):
    t = NullTranscriber(cfg)
    stats = transcribe.run(cfg, recorded, transcriber=t)
    assert (stats.done, stats.skipped_silent, stats.skipped_existing, stats.failed) == (3, 1, 0, [])
    assert len(t.calls) == 3  # silent chunk never reaches the backend
    silent = json.loads((recorded.transcript_dir("null", "them") / "00001.json").read_text())
    assert silent["skipped"] == "silence" and silent["segments"] == []
    assert recorded.read()["stages"]["transcribe"] == {**recorded.read()["stages"]["transcribe"],
                                                       "status": "done", "backend": "null"}

    t2 = NullTranscriber(cfg)
    stats = transcribe.run(cfg, recorded, transcriber=t2)
    assert stats.skipped_existing == 4 and stats.done == 0 and t2.calls == []

    t3 = NullTranscriber(cfg)
    stats = transcribe.run(cfg, recorded, transcriber=t3, force=True)
    assert stats.done == 3 and stats.skipped_silent == 1 and len(t3.calls) == 3


def test_failed_chunk_is_retried_later(cfg, recorded):
    class Flaky(NullTranscriber):
        def transcribe(self, path, lang_hint):
            if path.parent.name == "me" and path.stem == "00001":
                raise RuntimeError("network down")
            return super().transcribe(path, lang_hint)

    stats = transcribe.run(cfg, recorded, transcriber=Flaky(cfg))
    assert stats.done == 2 and len(stats.failed) == 1
    assert recorded.read()["stages"]["transcribe"]["status"] == "failed"

    t = NullTranscriber(cfg)
    stats = transcribe.run(cfg, recorded, transcriber=t)
    assert [p.stem for p in t.calls] == ["00001"] and stats.done == 1
    assert recorded.read()["stages"]["transcribe"]["status"] == "done"


def test_refuses_while_recording(cfg, recorded):
    recorded.set_state(sess.RECORDING)
    with pytest.raises(RuntimeError):
        transcribe.run(cfg, recorded, transcriber=NullTranscriber(cfg))


def _blip(path, total=30, at=26.2, length=0.36):
    """Digital silence with one short loud blip, like a notification ping (session 2026-11-13_1147)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=880:duration={total}:sample_rate=16000",
                    "-af", f"volume=enable='between(t,{at},{at + length})':volume=0.9,"
                           f"volume=enable='not(between(t,{at},{at + length}))':volume=0",
                    "-ac", "1", str(path)], check=True)


def test_voiced_seconds(tmp_path):
    blip, speechy = tmp_path / "blip.flac", tmp_path / "tone.flac"
    _blip(blip)
    _tone(speechy, seconds=3)
    assert transcribe.peak_db(blip) > -45  # the old peak check let this through
    assert transcribe.voiced_seconds(blip, -45) < 0.6
    assert 2.8 < transcribe.voiced_seconds(speechy, -45) <= 3.05


def test_blip_chunk_is_skipped(cfg):
    s = sess.create(cfg)
    _tone(s.audio_dir("me") / "00000.flac")
    _blip(s.audio_dir("them") / "00000.flac")
    s.set_state(sess.RECORDED, chunk_seconds=cfg.chunk_seconds)
    t = NullTranscriber(cfg)
    stats = transcribe.run(cfg, s, transcriber=t)
    assert (stats.done, stats.skipped_silent) == (1, 1)
    rec = json.loads((s.transcript_dir("null", "them") / "00000.json").read_text())
    assert rec["skipped"] == "silence" and rec["voiced_s"] < 0.6


LOOP = "आपका प्रशिक्षण कैसे होगा " * 20


def test_repetition_detection():
    real = "this is a meeting recording जो अभी मैं mixed language के साथ test करने वाला हूँ"
    assert transcribe.compression_ratio(LOOP) > 2.4
    assert transcribe.compression_ratio(real) < 2.4
    assert transcribe.collapse_repeats(LOOP) == "आपका प्रशिक्षण कैसे होगा"
    assert transcribe.collapse_repeats(real) == real
    assert transcribe.collapse_repeats("we will we will rock you") == "we will rock you"


def test_looping_backend_output_is_collapsed(cfg, recorded):
    t = NullTranscriber(cfg)
    t.transcribe = lambda path, lang: [asr.Segment(0, 30, LOOP)]
    transcribe.run(cfg, recorded, transcriber=t)
    rec = json.loads((recorded.transcript_dir("null", "me") / "00000.json").read_text())
    assert rec["segments"][0]["text"] == "आपका प्रशिक्षण कैसे होगा"
    assert rec["repetition_raw"].startswith("आपका")
