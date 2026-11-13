"""The speech-to-text plug-in interface.

A backend is anything with a `name` and `transcribe(path, lang_hint) -> list[Segment]`.
Segment times are seconds relative to the start of the chunk file. Register a factory with
`@register("name")`; it receives the Config so backends can read their own settings / keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from ..config import Config


@dataclass
class Segment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(float(d["start"]), float(d["end"]), str(d["text"]))


@runtime_checkable
class Transcriber(Protocol):
    name: str
    max_parallel: int  # how many chunks may be in flight at once for this backend

    def transcribe(self, path: Path, lang_hint: str | None) -> list[Segment]: ...


Factory = Callable[[Config], Transcriber]
_REGISTRY: dict[str, Factory] = {}


def register(name: str) -> Callable[[Factory], Factory]:
    def deco(factory: Factory) -> Factory:
        _REGISTRY[name] = factory
        return factory

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, cfg: Config) -> Transcriber:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise LookupError(f"unknown ASR backend {name!r}; available: {', '.join(available())}") from None
    return factory(cfg)
