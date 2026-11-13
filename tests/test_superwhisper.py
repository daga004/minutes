import pytest

from minutes.asr import superwhisper as sw


@pytest.fixture
def backend(monkeypatch, cfg):
    monkeypatch.setattr(sw.shutil, "which", lambda name: "/bin/true")
    return sw.SuperwhisperTranscriber(cfg, timeout=5)


def _fake(monkeypatch, backend, last):
    def fake_json(*args):
        if args[0] == "transcribe":
            return {"ok": True, "session_id": 42}
        return {"phase": "idle", "last": {"session_id": 42, **last}}
    monkeypatch.setattr(backend, "_json", fake_json)
    monkeypatch.setattr(sw, "_duration", lambda p: 29.0)


def test_text_becomes_one_segment(monkeypatch, backend, tmp_path):
    _fake(monkeypatch, backend, {"text": " नमस्ते, hello "})
    segs = backend.transcribe(tmp_path / "00000.flac", "hi")
    assert [(s.start, s.end, s.text) for s in segs] == [(0.0, 29.0, "नमस्ते, hello")]


def test_no_speech_is_empty_not_error(monkeypatch, backend, tmp_path):
    _fake(monkeypatch, backend, {"text": "", "error": "No speech detected"})
    assert backend.transcribe(tmp_path / "00002.flac", "hi") == []


def test_other_errors_raise(monkeypatch, backend, tmp_path):
    _fake(monkeypatch, backend, {"text": "", "error": "network unreachable"})
    with pytest.raises(RuntimeError, match="network unreachable"):
        backend.transcribe(tmp_path / "00000.flac", "hi")
