"""Publish a meeting as one note in a Markdown vault that is synced to Outline.

The note is the summary followed by the full (cleaned) transcript. It is written into a folder
that an Outline sync daemon (e.g. outline-sync-daemon) mirrors; the daemon pushes it and writes
the document's `outline_url` into the frontmatter, which is waited for and copied to the clipboard.

    [outline]
    vault = "~/Documents/outline_and_notes"   # unset = don't publish
    folder = "Notes/meetings"                 # inside the vault; a folder the daemon maps
    wait_seconds = 120                        # how long to wait for the daemon's outline_url
    copy_link = true

Re-publishing a session rewrites the same file and keeps the daemon's `outline_*` keys, so the
Outline document is updated in place rather than duplicated.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from . import calendar, merge
from . import session as sess
from .config import Config

URL = re.compile(r"^outline_url:\s*[\"']?(https?://[^\s\"']+)", re.M)


def enabled(cfg: Config) -> bool:
    return bool(cfg.outline.get("vault"))


def folder(cfg: Config) -> Path:
    return Path(cfg.outline["vault"]).expanduser() / cfg.outline.get("folder", "meetings")


def split_frontmatter(text: str) -> tuple[list[str], str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[4:end].splitlines(), text[end + 5:]
    return [], text


def kept_keys(lines: list[str]) -> list[str]:
    """The sync daemon's own (outline_*) and the editor's (_*) frontmatter, with continuation lines."""
    out, keep = [], False
    for l in lines:
        if l and not l[0].isspace() and not l.startswith("-"):
            keep = l.startswith(("outline_", "_"))
        if keep:
            out.append(l)
    return out


def note_path(cfg: Config, s: sess.Session, title: str) -> Path:
    prev = s.read().get("outline_note")
    if prev and Path(prev).exists():
        return Path(prev)
    meta = s.read()
    date = (meta.get("started_at") or sess.base_id(s.id))[:10]
    stem = f"{date}-{sess.slugify(title) or sess.base_id(s.id)}"
    p, n = folder(cfg) / f"{stem}.md", 2
    while p.exists():
        p, n = folder(cfg) / f"{stem}-{n}.md", n + 1
    return p


def summary_body(s: sess.Session) -> str:
    """summary.md without its H1 and the local info line (the note has its own)."""
    lines = s.summary_md.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and (not lines[0].strip() or lines[0].startswith("_")):
        lines = lines[1:]
    return "\n".join(lines).strip()


def transcript_body(s: sess.Session) -> tuple[str, bool]:
    """(speaker lines, cleaned?) from transcript.clean.md, or the raw transcript if not cleaned."""
    from . import clean

    src = s.clean_md if s.clean_md.exists() else s.transcript_md
    _, blocks = clean.split(src.read_text(encoding="utf-8")) if src.exists() else ("", [])
    return "\n\n".join(blocks), src == s.clean_md


def render(cfg: Config, s: sess.Session, keep: list[str]) -> str:
    meta = s.read()
    event = calendar.load(s)
    title = meta.get("title") or (event.title if event else "") or "Meeting"
    started = meta.get("started_at", "")
    try:
        when = datetime.fromisoformat(started)
        when_s = when.strftime("%a %d %b %Y, %H:%M")
    except ValueError:
        when_s = started or sess.base_id(s.id)
    dur = meta.get("duration_s")
    people = [a.label for a in event.others] if event else []

    fm = ["type: Meeting", f"date: {started[:10] or sess.base_id(s.id)[:10]}",
          f"minutes_session: {json.dumps(sess.base_id(s.id))}"]
    if dur:
        fm.append(f"duration_min: {round(dur / 60)}")
    if people:
        fm.append("attendees:")
        fm += [f"  - {json.dumps(p, ensure_ascii=False)}" for p in people]
    if event and event.series:
        fm.append(f"series: {json.dumps(event.series)}")
    fm += keep

    info = [when_s] + ([merge.fmt_time(dur)] if dur else []) + (["with " + ", ".join(people)] if people else [])
    if meta.get("topic") and meta["topic"].lower() != title.lower():
        info.insert(0, f"**{meta['topic']}**")
    transcript, cleaned = transcript_body(s)
    label = ("Cleaned by Claude from the automatic transcript; nothing is summarized."
             if cleaned else "Automatic transcript (not cleaned).")
    parts = [f"# {title}", "_" + " · ".join(info) + "_", summary_body(s), "## Transcript",
             f"_{label} Me = the recorder; Them = everyone else on the call._", transcript or "_No speech._"]
    return "---\n" + "\n".join(fm) + "\n---\n" + "\n\n".join(parts) + "\n"


def read_url(path: Path) -> str | None:
    try:
        fm, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    m = URL.search("\n".join(fm))
    return m.group(1) if m else None


def copy_to_clipboard(text: str) -> None:
    """wl-copy has to keep running to serve the clipboard, so it gets its own transient unit
    (a child of this process would be killed with the background job's unit)."""
    if not shutil.which("wl-copy"):
        return
    cmd = ["wl-copy", "--foreground", "--", text]
    if shutil.which("systemd-run"):
        cmd = ["systemd-run", "--user", "--quiet", "--collect", *cmd]
    try:
        subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def publish(cfg: Config, s: sess.Session, wait: bool = True) -> tuple[Path, str | None] | None:
    """Write (or rewrite) the note. Returns (path, Outline URL or None), or None if disabled."""
    if not enabled(cfg) or not s.summary_md.exists():
        return None
    meta = s.read()
    path = note_path(cfg, s, meta.get("title") or "meeting")
    keep: list[str] = []
    if path.exists():
        keep = kept_keys(split_frontmatter(path.read_text(encoding="utf-8"))[0])
    sess.atomic_write_text(path, render(cfg, s, keep))
    s.update(outline_note=str(path))
    url = read_url(path) if keep else None  # an update keeps the URL it already had
    if not url and wait:
        deadline = time.monotonic() + float(cfg.outline.get("wait_seconds", 120))
        while time.monotonic() < deadline and not (url := read_url(path)):
            time.sleep(2)
    if url:
        s.update(outline_url=url)
        if cfg.outline.get("copy_link", True):
            copy_to_clipboard(url)
    s.set_stage("publish", "done" if url else "pending", note=str(path), url=url)
    return path, url
