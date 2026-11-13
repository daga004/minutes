"""`minutes process`: transcribe -> merge -> summarize, with notifications."""

from __future__ import annotations

import traceback

from . import merge, summarize, transcribe
from . import session as sess
from .config import Config
from .notify import notify


def process(cfg: Config, s: sess.Session, asr_backend: str | None = None, force: bool = False,
            transcriber=None) -> bool:
    s.set_state(sess.PROCESSING)
    try:
        stats = transcribe.run(cfg, s, backend=asr_backend, force=force, transcriber=transcriber)
        print(stats.describe(), flush=True)
        s.set_state(sess.TRANSCRIBED)
        merge.run(cfg, s, backend=stats.backend)
        s = summarize.run(cfg, s)  # may rename the folder to include the meeting title
        path = s.summary_md
        s.set_state(sess.SUMMARIZED)
    except Exception as e:
        traceback.print_exc()
        s.set_state(sess.FAILED, error=str(e)[:500])
        notify("Meeting processing failed", f"{s.id}: {e}\nRetry: minutes process {s.id}", urgency="critical")
        return False
    if stats.failed:
        notify("Meeting summary ready (incomplete)",
               f"{len(stats.failed)} chunk(s) failed to transcribe. Retry: minutes process {s.id}\n{path}",
               urgency="critical")
    else:
        notify("Meeting summary ready", str(path))
    print(path, flush=True)
    return not stats.failed
