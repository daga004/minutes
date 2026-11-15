"""Summary stage: transcript.md -> summary.md via a pluggable Summarizer."""

from __future__ import annotations

import re

from .. import calendar, merge
from .. import session as sess
from ..config import Config
from . import claude_cli  # noqa: F401  (registration)
from .base import Summarizer, available, get, register

__all__ = ["Summarizer", "available", "get", "register", "run"]


def split_facts(body: str) -> tuple[str | None, str]:
    """Take a leading 'Key facts: ...' line off the summary body ('none' -> None)."""
    lines = body.lstrip().splitlines()
    if lines and lines[0].lower().lstrip("*_ ").startswith("key facts:"):
        facts = lines[0].split(":", 1)[1].strip().strip("*_ ")
        rest = "\n".join(lines[1:]).lstrip("\n")
        empty = not facts or re.match(r"(?i)none\b", facts)  # "none", "None recorded", "none (only ...)"
        return (None if empty else facts), rest
    return None, body


def split_title(body: str) -> tuple[str | None, str]:
    """Take the leading '# Title' line off the summarizer output."""
    lines = body.lstrip().splitlines()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip().strip("*_`\"' ").rstrip(".")
        return (title or None), "\n".join(lines[1:]).lstrip("\n")
    return None, body


def run(cfg: Config, s: sess.Session, backend: str | None = None,
        summarizer: Summarizer | None = None) -> sess.Session:
    """Write summary.md and title the session. Returns the session, which may have been renamed
    to '<base id>_<title-slug>' (the old Session object then points at a path that is gone)."""
    if not s.transcript_md.exists():
        merge.run(cfg, s)
    transcript = s.transcript_md.read_text(encoding="utf-8")
    meta = s.read()
    event = calendar.load(s)
    prompt_input = (f"<meeting>\n{calendar.describe(event)}\n</meeting>\n\n" if event else "") \
        + f"<transcript>\n{transcript}\n</transcript>"
    title = None
    if not meta.get("stages", {}).get("merge", {}).get("lines"):
        body = "## Summary\nNo speech was transcribed in this recording.\n"
        name = "none"
    else:
        sm = summarizer or get(backend or cfg.summary_backend, cfg)
        name = sm.name
        s.set_stage("summarize", "running", backend=name)
        try:
            title, body = split_title(sm.summarize(prompt_input))
            facts, body = split_facts(body)
            s.update(key_facts=facts)
            if facts:
                body = f"**Key facts:** {facts}\n\n" + body
        except Exception as e:
            s.set_stage("summarize", "failed", backend=name, error=str(e)[:500])
            raise
    base = sess.base_id(s.id)
    topic = title
    if event and len(event.title) >= 3:
        title = event.title  # the name colleagues know the meeting by
    info = [base, f"started {meta.get('started_at', '?')}"]
    if event and topic and topic.lower() != event.title.lower():
        info.insert(0, f"**{topic}**")
    if event and event.others:
        info.append("with " + ", ".join(a.label for a in event.others))
    info += [f"summary by {name}", "[transcript](transcript.md)"]
    header = f"# {title or 'Meeting minutes'}\n\n_" + " · ".join(info) + "_\n\n"
    sess.atomic_write_text(s.summary_md, header + body)
    s.set_stage("summarize", "done", backend=name)
    if topic:
        s.update(topic=topic)
    if title:
        s = sess.retitle(s, title)
    return s
