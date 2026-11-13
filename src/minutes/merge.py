"""Interleave Me / Them transcripts by absolute time into transcript.md."""

from __future__ import annotations

from dataclasses import dataclass

from . import session as sess
from . import transcribe
from .config import Config

LABEL = {"me": "Me", "them": "Them"}


@dataclass
class Line:
    t: float
    side: str
    text: str


def pick_backend(cfg: Config, s: sess.Session, backend: str | None) -> str:
    return backend or s.read().get("stages", {}).get("transcribe", {}).get("backend") or cfg.asr_backend


def stream_offsets(meta: dict) -> dict[str, float]:
    """Seconds each stream started after the earliest one (ffmpeg processes start ~ms apart)."""
    starts = {side: v.get("started_epoch") for side, v in meta.get("streams", {}).items()}
    known = [v for v in starts.values() if v]
    base = min(known) if known else 0.0
    return {side: (starts.get(side) or base) - base for side in sess.SIDES}


def collect(s: sess.Session, backend: str, chunk_seconds: float, offsets: dict[str, float]) -> list[Line]:
    lines: list[Line] = []
    for side in sess.SIDES:
        for idx, segs in transcribe.load_segments(s, backend, side).items():
            for seg in segs:
                if seg.text.strip():
                    lines.append(Line(offsets.get(side, 0.0) + idx * chunk_seconds + seg.start, side, seg.text.strip()))
    lines.sort(key=lambda l: (l.t, sess.SIDES.index(l.side)))
    return lines


def coalesce(lines: list[Line]) -> list[Line]:
    """Join consecutive lines from the same speaker into one paragraph."""
    out: list[Line] = []
    for l in lines:
        if out and out[-1].side == l.side:
            out[-1] = Line(out[-1].t, l.side, f"{out[-1].text} {l.text}")
        else:
            out.append(Line(l.t, l.side, l.text))
    return out


def fmt_time(t: float) -> str:
    t = max(0, int(t))
    h, rem = divmod(t, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def render(s: sess.Session, backend: str, lines: list[Line], missing: list[str]) -> str:
    meta = s.read()
    dur = meta.get("duration_s")
    head = [f"# Meeting transcript: {s.id}", ""]
    info = [f"Started {meta.get('started_at', '?')}"]
    if dur:
        info.append(f"duration {fmt_time(dur)}")
    info.append(f"ASR: {backend}")
    head.append("_" + " · ".join(info) + "_")
    head.append("")
    head.append("_Me = the person recording (microphone). Them = everyone else (system audio)._")
    if missing:
        head += ["", f"> **Warning:** {len(missing)} chunk(s) not transcribed: {', '.join(missing)}. "
                 f"Run `minutes transcribe {s.id}` to retry."]
    head.append("")
    if not lines:
        head.append("_No speech was transcribed._")
    body = [f"**[{fmt_time(l.t)}] {LABEL[l.side]}:** {l.text}\n" for l in coalesce(lines)]
    return "\n".join(head) + "\n" + "\n".join(body)


def run(cfg: Config, s: sess.Session, backend: str | None = None):
    backend = pick_backend(cfg, s, backend)
    meta = s.read()
    chunk_seconds = float(meta.get("chunk_seconds") or cfg.chunk_seconds)
    lines = collect(s, backend, chunk_seconds, stream_offsets(meta))
    missing = [f"{side}/{c.stem}" for side in sess.SIDES for c in s.chunks(side)
               if not transcribe.output_path(s, backend, c).exists()]
    sess.atomic_write_text(s.transcript_md, render(s, backend, lines, missing))
    s.set_stage("merge", "done", backend=backend, lines=len(lines), missing=len(missing))
    return s.transcript_md
