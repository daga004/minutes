"""`minutes process`: transcribe -> merge -> summarize, with notifications."""

from __future__ import annotations

import traceback

from . import clean, merge, outline, summarize, transcribe
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
    finish(cfg, s, failed_chunks=len(stats.failed))
    print(path, flush=True)
    return not stats.failed


def finish(cfg: Config, s: sess.Session, failed_chunks: int = 0) -> None:
    """After the summary: clean the transcript, publish the note, tell the user where it is.
    Neither step can fail the session; the summary and raw transcript are already saved."""
    problems = []
    if failed_chunks:
        problems.append(f"{failed_chunks} chunk(s) failed to transcribe; retry: minutes process {s.id}")
    if cfg.clean_transcript and s.read().get("stages", {}).get("merge", {}).get("lines"):
        try:
            clean.run(cfg, s)
        except Exception as e:
            traceback.print_exc()
            problems.append(f"transcript not cleaned ({str(e)[:120]}); the note has the raw one")
    link = None
    try:
        outline.write_local(cfg, s)
    except Exception as e:
        traceback.print_exc()
        problems.append(f"notes.md not written: {str(e)[:120]}")
    try:
        published = outline.publish(cfg, s)
        if published:
            note, url = published
            link = url or f"{note} (waiting for Outline sync)"
    except Exception as e:
        traceback.print_exc()
        problems.append(f"Outline note not written: {str(e)[:120]}")
    title = s.read().get("title") or s.id
    body = "\n".join([title, link + (" (link copied)" if link and link.startswith("http")
                                       and cfg.outline.get("copy_link", True) else "")
                       if link else str(s.notes_md if s.notes_md.exists() else s.summary_md), *problems])
    notify("Meeting notes ready" + (" (with problems)" if problems else ""), body,
           urgency="critical" if problems else "normal")
