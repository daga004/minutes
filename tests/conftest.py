import pytest

from minutes.config import Config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    return Config(storage_dir=tmp_path / "data", asr_backend="null", chunk_seconds=30)
