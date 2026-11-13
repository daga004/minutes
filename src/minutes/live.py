"""`minutes live <id>`: transcribe a meeting while it is being recorded, then finish it.

Started (detached) by `minutes start`. While the session is recording, every chunk that is
closed (a newer chunk exists on the same side) and not yet transcribed is sent to the ASR
backend, so the model stays loaded and the backlog stays near zero. When the recording stops,
the same process runs the normal pipeline (remaining chunks, merge, summary) with the already
loaded model; `minutes stop` sees this process alive and does not spawn a second one.

If this process dies, nothing is lost: `minutes stop` falls back to `minutes process`, which
resumes from whatever chunks were already transcribed.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

from . import asr, pipeline, transcribe
from . import session as sess
from .config import Config, runtime_dir

POLL = 2.0


def pidfile(s: sess.Session) -> Path:
    return runtime_dir() / f"live-{s.id}.pid"


def running(s: sess.Session) -> bool:
    try:
        pid = int(pidfile(s).read_text())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def closed_chunks(s: sess.Session) -> list[Path]:
    """Chunks the recorder has finished writing: all but the newest on each side."""
    out: list[Path] = []
    for side in sess.SIDES:
        out += s.chunks(side)[:-1]
    return sorted(out, key=lambda c: (c.stem, c.parent.name))  # oldest first, me/them interleaved


def run(cfg: Config, s: sess.Session, backend: str | None = None) -> bool:
    runtime_dir().mkdir(parents=True, exist_ok=True)
    pidfile(s).write_text(str(os.getpid()))
    t = asr.get(backend or cfg.asr_backend, cfg)
    failed: dict[str, int] = {}
    try:
        while s.state == sess.RECORDING:
            todo = [c for c in closed_chunks(s)
                    if not transcribe.output_path(s, t.name, c).exists() and failed.get(str(c), 0) < 2]
            if not todo:
                time.sleep(POLL)
                continue
            chunk = todo[0]
            try:
                transcribe._one(cfg, t, s, chunk)
                lag = len(todo) - 1
                print(f"live: {chunk.parent.name}/{chunk.name} done"
                      + (f" ({lag} behind)" if lag else "")
                      + (f" on {t.engine_label}" if getattr(t, "engine_label", None) else ""), flush=True)
            except Exception as e:  # leave it for the final pass
                failed[str(chunk)] = failed.get(str(chunk), 0) + 1
                print(f"live: {chunk.parent.name}/{chunk.name} failed: {e}", flush=True)
        # Recording stopped (stop() finalizes headers before changing state). Wait for it to settle.
        deadline = time.monotonic() + 60
        while s.state not in (sess.RECORDED, sess.INTERRUPTED) and time.monotonic() < deadline:
            time.sleep(0.5)
        if s.state not in (sess.RECORDED, sess.INTERRUPTED):
            print(f"live: session ended in state {s.state}; not processing", flush=True)
            return False
        return pipeline.process(cfg, s, asr_backend=t.name, transcriber=t)
    except Exception:
        traceback.print_exc()
        return False
    finally:
        if hasattr(t, "close"):
            t.close()
        pidfile(s).unlink(missing_ok=True)
