"""TaraTranscriber's worker protocol, exercised with a fake worker (no OpenVINO needed)."""

import os
import sys
import textwrap
from pathlib import Path

import pytest

from minutes import config
from minutes.asr import tara

FAKE = textwrap.dedent('''
    import json, sys
    mode = "{mode}"
    print("noise on stdout before dup is fine?", file=sys.stderr)
    if mode == "crash":
        sys.exit(3)
    print(json.dumps({{"ready": True, "device": "GPU"}}), flush=True)
    import base64
    for line in sys.stdin:
        req = json.loads(line)
        req["path"] = base64.b64decode(req["pcm_b64"]).decode()
        if req["path"].endswith("bad.flac"):
            print(json.dumps({{"error": "CalledProcessError: ffmpeg"}}), flush=True)
        elif req["path"].endswith("quiet.flac"):
            print(json.dumps({{"text": "  ", "duration": 30.0}}), flush=True)
        else:
            print(json.dumps({{"text": "मैं कल meeting में था", "duration": 29.5, "lang": req["lang"]}}), flush=True)
''')


def make(tmp_path: Path, monkeypatch, mode: str = "ok", **opts) -> tara.TaraTranscriber:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(tara, "pcm16", lambda path: (str(path).encode(), 1.0))  # fake audio = path bytes
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE.format(mode=mode))
    monkeypatch.setattr(tara, "WORKER", worker)
    model = tmp_path / "model"
    model.mkdir()
    (model / "openvino_encoder_model.xml").write_text("")
    cfg = config.Config(storage_dir=tmp_path / "store")
    cfg.asr_options = {"tara": {"python": sys.executable, "model_dir": str(model), "timeout": 10,
                                "remote": "", **opts}}
    return tara.TaraTranscriber(cfg)


def test_transcribes_and_reuses_one_worker(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    a = t.transcribe(tmp_path / "00000.flac", "hi")
    pid = t._proc.pid
    b = t.transcribe(tmp_path / "00001.flac", "hi")
    assert a[0].text == "मैं कल meeting में था" and a[0].end == 29.5
    assert b and t._proc.pid == pid  # model stays loaded between chunks
    assert t.engine_label == "local: GPU"
    t.close()
    assert t._proc is None


def test_empty_text_is_no_segments(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    assert t.transcribe(tmp_path / "quiet.flac", "hi") == []
    t.close()


def test_worker_error_raises_but_worker_survives(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="tara: CalledProcessError"):
        t.transcribe(tmp_path / "bad.flac", "hi")
    assert t.transcribe(tmp_path / "00002.flac", "hi")
    t.close()


def test_worker_crash_is_reported(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, mode="crash")
    with pytest.raises(RuntimeError, match="exited"):
        t.transcribe(tmp_path / "00000.flac", "hi")
    # the machine-wide lock was released, so a new worker can start
    t.close()
    assert t._gpu_lock is None


def test_missing_engine_says_how_to_fix(tmp_path):
    cfg = config.Config(storage_dir=tmp_path)
    cfg.asr_options = {"tara": {"remote": ""}}
    with pytest.raises(RuntimeError, match="minutes setup tara"):
        tara.TaraTranscriber(cfg)


def test_asr_tables_are_loaded(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[asr]\nbackend = "tara"\n[asr.tara]\ndevice = "CPU"\n')
    cfg = config.load(p)
    assert cfg.asr_options == {"tara": {"device": "CPU"}}


def test_uses_remote_when_its_gpu_is_free(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, remote="gpu-box")
    monkeypatch.setattr(tara, "remote_free_mb", lambda host: 10068)
    assert t.choose() == "remote"


def test_uses_local_when_remote_busy_or_down(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, remote="gpu-box")
    monkeypatch.setattr(tara, "remote_free_mb", lambda host: 4000)
    assert t.choose() == "local"
    monkeypatch.setattr(tara, "remote_free_mb", lambda host: None)
    assert t.choose() == "local"


def test_remote_failure_falls_back_to_local(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, remote="gpu-box")
    monkeypatch.setattr(tara, "remote_free_mb", lambda host: 10068)
    real_start = t._start

    def start(where=None):
        where = where or t.choose()
        t.where = where
        if where == "remote":
            raise RuntimeError("ssh: connect to host gpu-box: No route to host")
        real_start("local")

    monkeypatch.setattr(t, "_start", start)
    seg = t.transcribe(tmp_path / "00000.flac", "hi")
    assert seg and t.where == "local" and t._remote_failed
    assert t.choose() == "local"  # stays local for the rest of the run
    t.close()


def test_live_only_takes_closed_chunks(tmp_path):
    from minutes import live
    from minutes import session as sess

    cfg = config.Config(storage_dir=tmp_path)
    s = sess.create(cfg)
    for side, n in (("me", 3), ("them", 1)):
        for i in range(n):
            (s.audio_dir(side) / f"{i:05d}.flac").write_bytes(b"x")
    got = [f"{c.parent.name}/{c.stem}" for c in live.closed_chunks(s)]
    assert got == ["me/00000", "me/00001"]  # newest per side is still being written
