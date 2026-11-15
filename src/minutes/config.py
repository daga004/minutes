"""Configuration loaded from ~/.config/minutes/config.toml (all keys optional)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else Path.home() / fallback


def config_path() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / "minutes" / "config.toml"


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/minutes-{os.getuid()}"
    return Path(base) / "minutes"


@dataclass
class Config:
    storage_dir: Path
    asr_backend: str = "tara"
    lang_hint: str = "hi"  # meetings are Hindi / Hinglish / English
    asr_parallel: int = 3
    summary_backend: str = "claude-cli"
    summary_model: str | None = None
    chunk_seconds: int = 30
    sample_rate: int = 16000
    silence_peak_db: float = -45.0  # chunks whose loudest sample is below this (dBFS) are skipped
    min_voiced_seconds: float = 1.0  # chunks with less sound than this are treated as silent
    live: bool = True  # transcribe chunks while recording
    asr_options: dict[str, dict] = field(default_factory=dict)  # [asr.<backend>] tables
    clean_transcript: bool = True  # [summary] clean_transcript: Claude tidies the full transcript
    clean_language: str = "as-spoken"  # [summary] clean_language: "as-spoken" or "english"
    calendar: dict = field(default_factory=dict)  # [calendar] (see calendar.py)
    outline: dict = field(default_factory=dict)  # [outline] (see outline.py); off unless vault is set
    watch: dict = field(default_factory=dict)  # [watch] (see watch.py)

    @property
    def sessions_dir(self) -> Path:
        return self.storage_dir / "sessions"


def load(path: Path | None = None) -> Config:
    path = path or config_path()
    data: dict = {}
    if path.exists():
        with path.open("rb") as fh:
            data = tomllib.load(fh)

    asr = data.get("asr", {})
    summary = data.get("summary", {})
    capture = data.get("capture", {})
    transcribe = data.get("transcribe", {})
    storage = data.get("storage", {})

    default_storage = _xdg("XDG_DATA_HOME", ".local/share") / "minutes"
    cfg = Config(storage_dir=Path(storage.get("dir", default_storage)).expanduser())
    cfg.asr_backend = asr.get("backend", cfg.asr_backend)
    cfg.lang_hint = asr.get("lang_hint", cfg.lang_hint)
    cfg.asr_parallel = int(asr.get("parallel", cfg.asr_parallel))
    cfg.asr_options = {k: v for k, v in asr.items() if isinstance(v, dict)}
    cfg.summary_backend = summary.get("backend", cfg.summary_backend)
    cfg.summary_model = summary.get("model", cfg.summary_model)
    cfg.chunk_seconds = int(capture.get("chunk_seconds", cfg.chunk_seconds))
    cfg.sample_rate = int(capture.get("sample_rate", cfg.sample_rate))
    cfg.silence_peak_db = float(transcribe.get("silence_peak_db", cfg.silence_peak_db))
    cfg.min_voiced_seconds = float(transcribe.get("min_voiced_seconds", cfg.min_voiced_seconds))
    cfg.live = bool(transcribe.get("live", cfg.live))
    cfg.clean_transcript = bool(summary.get("clean_transcript", cfg.clean_transcript))
    cfg.clean_language = str(summary.get("clean_language", cfg.clean_language))
    cfg.calendar = dict(data.get("calendar", {}))
    cfg.outline = dict(data.get("outline", {}))
    cfg.watch = dict(data.get("watch", {}))
    return cfg
