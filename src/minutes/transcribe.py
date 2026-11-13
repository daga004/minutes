"""Transcribe every audio chunk of a session with one backend.

Output: transcripts/<backend>/<side>/<chunk>.json, one file per chunk, written atomically.
A chunk whose output exists is skipped (resume) unless force=True. Chunks whose peak level
is below cfg.silence_peak_db, or that hold less than cfg.min_voiced_seconds of sound above it
(e.g. silence plus one notification ping), are recorded as skipped without calling the backend, which
saves cost and avoids ASR models inventing text on silence.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import asr
from . import session as sess
from .config import Config

_MAX_RE = re.compile(r"max_volume:\s*(-?inf|-?[\d.]+) dB")
_SIL_START = re.compile(r"silence_start: (-?[\d.]+)")
_SIL_END = re.compile(r"silence_end: ([\d.]+)")
_DUR_RE = re.compile(r"time=(\d+):(\d+):([\d.]+)")


def voiced_seconds(path: Path, threshold_db: float) -> float:
    """Seconds of audio louder than threshold_db (everything not inside a >=0.3 s silence).

    A single click or notification ping is a fraction of a second, so a chunk that is all
    silence plus a blip scores ~0.3 s and is skipped instead of letting the model hallucinate.
    """
    r = subprocess.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path),
         "-af", f"silencedetect=n={threshold_db}dB:d=0.3", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    stamps = _DUR_RE.findall(r.stderr)
    if not stamps:
        return 0.0
    h, m, sec = stamps[-1]
    total = int(h) * 3600 + int(m) * 60 + float(sec)
    silent, start = 0.0, None
    for line in r.stderr.splitlines():
        if (ms := _SIL_START.search(line)):
            start = max(0.0, float(ms.group(1)))
        elif (me := _SIL_END.search(line)) and start is not None:
            silent += float(me.group(1)) - start
            start = None
    if start is not None:  # silence ran to the end
        silent += total - start
    return max(0.0, total - silent)


def compression_ratio(text: str) -> float:
    """Whisper's own hallucination test: looping text compresses far better than speech (>2.4)."""
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data))) if data else 0.0


def collapse_repeats(text: str, max_n: int = 12) -> str:
    """Keep one copy of a phrase repeated back to back ("a b c a b c a b c" -> "a b c")."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        out.append(words[i])
        i += 1
        for n in range(1, max_n + 1):
            if len(out) >= n and words[i:i + n] == out[-n:]:
                while words[i:i + n] == out[-n:]:
                    i += n
                break
    return " ".join(out)


def peak_db(path: Path) -> float:
    """Loudest sample in dBFS (-inf for digital silence or undecodable audio)."""
    r = subprocess.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    m = _MAX_RE.search(r.stderr)
    if not m:
        return -math.inf
    return -math.inf if "inf" in m.group(1) else float(m.group(1))


@dataclass
class Stats:
    backend: str
    done: int = 0
    skipped_existing: int = 0
    skipped_silent: int = 0
    failed: list[str] = field(default_factory=list)

    def describe(self) -> str:
        s = (f"{self.backend}: transcribed {self.done}, already done {self.skipped_existing}, "
             f"silent {self.skipped_silent}, failed {len(self.failed)}")
        if self.failed:
            s += "\n  " + "\n  ".join(self.failed)
        return s


def output_path(s: sess.Session, backend: str, chunk: Path) -> Path:
    return s.transcript_dir(backend, chunk.parent.name) / f"{chunk.stem}.json"


def _one(cfg: Config, t: asr.Transcriber, s: sess.Session, chunk: Path) -> str:
    out = output_path(s, t.name, chunk)
    peak = peak_db(chunk)
    record = {"backend": t.name, "side": chunk.parent.name, "chunk": chunk.stem,
              "peak_db": None if math.isinf(peak) else round(peak, 1), "skipped": None,
              "segments": [], "at": sess.now_iso()}
    voiced = voiced_seconds(chunk, cfg.silence_peak_db) if peak >= cfg.silence_peak_db else 0.0
    record["voiced_s"] = round(voiced, 2)
    if peak < cfg.silence_peak_db or voiced < cfg.min_voiced_seconds:
        record["skipped"] = "silence"
        sess.atomic_write_json(out, record)
        return "silent"
    segments = t.transcribe(chunk, cfg.lang_hint)
    for seg in segments:  # last line of defence if the backend still looped
        if compression_ratio(seg.text) > 2.4:
            record["repetition_raw"] = seg.text
            seg.text = collapse_repeats(seg.text)
    record["segments"] = [seg.to_dict() for seg in segments]
    if getattr(t, "engine_label", None):
        record["engine"] = t.engine_label  # where it ran, e.g. "gpu-box: NVIDIA GeForce RTX 5060 Ti"
    sess.atomic_write_json(out, record)
    return "done"


def run(cfg: Config, s: sess.Session, backend: str | None = None, force: bool = False,
        transcriber: asr.Transcriber | None = None) -> Stats:
    if s.state == sess.RECORDING:
        raise RuntimeError(f"{s.id} is still recording; stop it first")
    t = transcriber or asr.get(backend or cfg.asr_backend, cfg)
    stats = Stats(t.name)
    todo: list[Path] = []
    for side in sess.SIDES:
        for chunk in s.chunks(side):
            if not force and output_path(s, t.name, chunk).exists():
                stats.skipped_existing += 1
            else:
                todo.append(chunk)

    s.set_stage("transcribe", "running", backend=t.name)
    workers = max(1, min(cfg.asr_parallel, getattr(t, "max_parallel", 1)))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, cfg, t, s, c): c for c in todo}
            for fut in as_completed(futures):
                chunk = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:  # one bad chunk must not stop the rest
                    stats.failed.append(f"{chunk.parent.name}/{chunk.name}: {e}")
                    continue
                if result == "silent":
                    stats.skipped_silent += 1
                else:
                    stats.done += 1
    finally:
        if transcriber is None and hasattr(t, "close"):
            t.close()  # e.g. unload a local model worker

    if stats.failed:
        s.set_stage("transcribe", "failed", backend=t.name, error=f"{len(stats.failed)} chunk(s) failed")
    else:
        s.set_stage("transcribe", "done", backend=t.name)
    return stats


def load_segments(s: sess.Session, backend: str, side: str) -> dict[int, list[asr.Segment]]:
    """chunk index -> segments, for every chunk transcribed by `backend` on `side`."""
    d = s.transcript_dir(backend, side)
    out: dict[int, list[asr.Segment]] = {}
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        data = json.loads(f.read_text(encoding="utf-8"))
        out[int(f.stem)] = [asr.Segment.from_dict(x) for x in data.get("segments", [])]
    return out
