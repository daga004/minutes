"""Session folder layout and atomic session.json state.

A session is a folder; every pipeline stage reads and writes plain files inside it:

    session.json
    audio/{me,them}/00000.flac ...
    transcripts/<backend>/{me,them}/00000.json ...
    transcript.md
    summary.md
"""

from __future__ import annotations

import json
import os
import unicodedata
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config

SIDES = ("me", "them")

# Session-level states
RECORDING = "recording"
RECORDED = "recorded"
INTERRUPTED = "interrupted"  # recorder died without a clean stop; audio is kept
PROCESSING = "processing"
TRANSCRIBED = "transcribed"
SUMMARIZED = "summarized"
FAILED = "failed"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Session:
    path: Path

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def meta_path(self) -> Path:
        return self.path / "session.json"

    @property
    def transcript_md(self) -> Path:
        return self.path / "transcript.md"

    @property
    def summary_md(self) -> Path:
        return self.path / "summary.md"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    def audio_dir(self, side: str) -> Path:
        return self.path / "audio" / side

    def transcript_dir(self, backend: str, side: str) -> Path:
        return self.path / "transcripts" / backend / side

    def chunks(self, side: str) -> list[Path]:
        d = self.audio_dir(side)
        if not d.exists():
            return []
        return sorted(f for f in d.glob("[0-9]*.flac") if f.stat().st_size > 0)

    # --- metadata -------------------------------------------------------
    def read(self) -> dict[str, Any]:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def write(self, meta: dict[str, Any]) -> None:
        atomic_write_json(self.meta_path, meta)

    def update(self, **fields: Any) -> dict[str, Any]:
        meta = self.read()
        meta.update(fields)
        self.write(meta)
        return meta

    @property
    def state(self) -> str | None:
        return self.read().get("state")

    def set_state(self, state: str, **fields: Any) -> dict[str, Any]:
        return self.update(state=state, updated_at=now_iso(), **fields)

    def set_stage(self, stage: str, status: str, **fields: Any) -> dict[str, Any]:
        """Record per-stage status, e.g. stages.transcribe = {status: done, backend: ...}."""
        meta = self.read()
        stages = meta.setdefault("stages", {})
        entry = {"status": status, "at": now_iso(), **fields}
        if status != "failed":
            entry.pop("error", None)
        stages[stage] = entry
        meta["updated_at"] = entry["at"]
        self.write(meta)
        return meta


_BASE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}(?:-\d+)?")


def base_id(name: str) -> str:
    """The time-based part of a session id: '2026-11-13_1147_drone-work' -> '2026-11-13_1147'."""
    m = _BASE_RE.match(name)
    return m.group(0) if m else name


def slugify(title: str, max_len: int = 60) -> str:
    """'Drone work: plan & owners' -> 'drone-work-plan-owners' (ASCII; empty if nothing usable)."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rsplit("-", 1)[0]
    return slug


def retitle(s: "Session", title: str) -> "Session":
    """Record the meeting title and rename the folder to '<base id>_<slug>'. Returns the new Session.

    Re-running the summary replaces an older title. Lookups by the base id keep working.
    """
    s.update(title=title, base_id=base_id(s.id))
    slug = slugify(title)
    if not slug:
        return s
    name = f"{base_id(s.id)}_{slug}"
    if name == s.id:
        return s
    target, n = s.path.with_name(name), 2
    while target.exists():
        target, n = s.path.with_name(f"{name}-{n}"), n + 1
    os.rename(s.path, target)
    new = Session(target)
    new.update(id=new.id)
    return new


def create(cfg: Config, when: datetime | None = None) -> Session:
    when = when or datetime.now()
    base = when.strftime("%Y-%m-%d_%H%M")
    cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
    name, n = base, 1
    while True:
        path = cfg.sessions_dir / name
        try:
            path.mkdir()
            break
        except FileExistsError:
            n += 1
            name = f"{base}-{n}"
    s = Session(path)
    for side in SIDES:
        s.audio_dir(side).mkdir(parents=True)
    s.logs_dir.mkdir()
    s.write({"id": s.id, "created_at": now_iso(), "state": None, "stages": {}})
    return s


def all_sessions(cfg: Config) -> list[Session]:
    if not cfg.sessions_dir.exists():
        return []
    return [Session(p) for p in sorted(cfg.sessions_dir.iterdir()) if (p / "session.json").exists()]


def resolve(cfg: Config, ref: str | None = None) -> Session:
    """Find a session by exact id, unique prefix, or 'latest' / None."""
    sessions = all_sessions(cfg)
    if not sessions:
        raise LookupError(f"no sessions in {cfg.sessions_dir}")
    if ref in (None, "", "latest", "last"):
        return sessions[-1]
    exact = [s for s in sessions if s.id == ref] or [s for s in sessions if base_id(s.id) == ref]
    if exact:
        return exact[0]
    matches = [s for s in sessions if s.id.startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise LookupError(f"no session matches {ref!r}")
    raise LookupError(f"{ref!r} is ambiguous: {', '.join(s.id for s in matches)}")
