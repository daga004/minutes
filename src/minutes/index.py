"""One add-only index of all meetings: a Markdown table, newest first.

| Date | Meeting | Duration | Key facts |

"Meeting" links to the meeting's note (summary + full transcript): its Outline link once the sync
daemon has pushed it, otherwise a [[wikilink]] the daemon converts. With [outline] set, the index is
<vault>/<folder>/index.md and syncs to Outline like any note; otherwise <storage>/index.md with
links to each session's notes.md. Each finished meeting adds one row at the top of the table;
existing rows are never rewritten, and a meeting already listed is not added again.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import merge, outline
from . import session as sess
from .config import Config

TITLE = "Meeting index"


def path(cfg: Config) -> Path:
    return outline.folder(cfg) / "index.md" if outline.enabled(cfg) else cfg.storage_dir / "index.md"


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def row(cfg: Config, s: sess.Session) -> dict | None:
    meta = s.read()
    note = meta.get("outline_note") if outline.enabled(cfg) else None
    if not (note or s.notes_md.exists()):
        return None  # not finished (or a test recording without speech)
    return {"date": (meta.get("started_at") or sess.base_id(s.id))[:16].replace("T", " "),
            "title": meta.get("title") or sess.base_id(s.id), "url": meta.get("outline_url"),
            "note": note, "local": s.notes_md, "dur": meta.get("duration_s"),
            "facts": meta.get("key_facts") or ""}


def link(cfg: Config, r: dict) -> str:
    title = cell(r["title"]).replace("]", ")").replace("[", "(")
    if r["url"]:
        return f"[{title}]({r['url']})"
    if r["note"]:
        return f"[[{Path(r['note']).stem}|{title}]]"
    return f"[{title}]({r['local'].relative_to(cfg.storage_dir)})"


def keys(r: dict) -> list[str]:
    """Ways an existing row for this meeting can show up in the file (the sync daemon may have
    turned a wikilink into an Outline link or back)."""
    return [k for k in (r["url"], r["note"] and Path(r["note"]).stem, str(r["local"])) if k]


HEADER = (f"---\ntype: Note\n---\n# {TITLE}\n\n"
          "_Every recorded meeting, newest first; new meetings are added at the top. Each link "
          "opens the summary with the full transcript below it. Rows are added by minutes and "
          "never rewritten, so notes added here are kept._\n\n"
          "| Date | Meeting | Duration | Key facts |\n|---|---|---|---|\n")


SEPARATOR = "|---|---|---|---|"
# The header's divider row, in whatever form Outline saved it ("|------|:---|" ...).
DIVIDER = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)


def append(cfg: Config, sessions: list[sess.Session]) -> tuple[Path, int]:
    """Add a row at the top of the table for each finished session that isn't listed yet
    (later sessions in the list end up higher). Existing text is kept byte for byte."""
    p = path(cfg)
    if not p.exists():
        sess.atomic_write_text(p, HEADER)
    if any(row(cfg, s) for s in sessions) and outline.enabled(cfg):
        outline.pull_first(cfg, p)  # take colleagues' edits (and Outline's reformatting) first
    text = p.read_text(encoding="utf-8")
    added = 0
    for s in sessions:
        r = row(cfg, s)
        if not r or any(k in text for k in keys(r)):
            continue
        dur = merge.fmt_time(r["dur"]) if r["dur"] else ""
        line = f"| {r['date']} | {link(cfg, r)} | {dur} | {cell(r['facts'])} |"
        m = DIVIDER.search(text)
        if not m:  # table header edited away: start a new table at the end
            text = text.rstrip("\n") + "\n\n| Date | Meeting | Duration | Key facts |\n" + SEPARATOR + "\n"
            m = DIVIDER.search(text)
        cut = m.end() + 1 if m.end() < len(text) else len(text)
        if cut > len(text):
            text += "\n"
        text = text[:cut] + line + "\n" + text[cut:]
        added += 1
    if added:
        sess.atomic_write_text(p, text)
    return p, added


def backfill(cfg: Config) -> tuple[Path, int]:
    """Add every finished meeting that is missing, oldest first."""
    ss = sorted(sess.all_sessions(cfg), key=lambda s: s.read().get("started_at") or s.id)
    return append(cfg, ss)
