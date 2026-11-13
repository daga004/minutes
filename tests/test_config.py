from pathlib import Path

from minutes import config


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    cfg = config.load(tmp_path / "missing.toml")
    assert cfg.asr_backend == "tara"
    assert cfg.summary_backend == "claude-cli"
    assert cfg.chunk_seconds == 30
    assert cfg.sessions_dir == tmp_path / "minutes" / "sessions"


def test_overrides(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[asr]\nbackend = "null"\nparallel = 5\n'
        '[summary]\nmodel = "sonnet"\n'
        '[capture]\nchunk_seconds = 10\n'
        '[transcribe]\nsilence_peak_db = -40\n'
        f'[storage]\ndir = "{tmp_path / "store"}"\n'
    )
    cfg = config.load(p)
    assert cfg.asr_backend == "null"
    assert cfg.asr_parallel == 5
    assert cfg.summary_model == "sonnet"
    assert cfg.chunk_seconds == 10
    assert cfg.silence_peak_db == -40.0
    assert cfg.storage_dir == Path(tmp_path / "store")
