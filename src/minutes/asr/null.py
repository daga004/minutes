"""Fake backend: deterministic text, no network. For tests and offline pipeline checks."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import Segment, register


class NullTranscriber:
    name = "null"
    max_parallel = 8

    def __init__(self, cfg: Config):
        self.chunk_seconds = cfg.chunk_seconds
        self.calls: list[Path] = []

    def transcribe(self, path: Path, lang_hint: str | None) -> list[Segment]:
        self.calls.append(path)
        side = path.parent.name
        return [Segment(0.0, float(self.chunk_seconds), f"[{side} chunk {path.stem}]")]


@register("null")
def _factory(cfg: Config) -> NullTranscriber:
    return NullTranscriber(cfg)
