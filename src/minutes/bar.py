"""Everything the bar widget shows, as one JSON object (`minutes bar`), and the Join action."""

from __future__ import annotations

import shutil
import subprocess
import time

from . import calendar, prefs, stats
from . import session as sess
from .config import Config

JOIN_LEAD_S = 5 * 60  # the Join button appears this long before a meeting starts


def joinable(e: calendar.Event, now: float) -> bool:
    return bool(e.conference) and e.start - JOIN_LEAD_S <= now < e.end


def state(cfg: Config, now: float | None = None) -> dict:
    from . import capture

    now = time.time() if now is None else now
    rec = capture.current(cfg)
    try:
        events = calendar.upcoming(cfg, now=now, hours=14)
    except Exception:
        events = []
    nxt = [{"uid": e.uid, "title": e.title[:120], "start": e.start, "end": e.end,
            "conference": e.conference, "people": len(e.others), "joinable": joinable(e, now)}
           for e in events if e.end > now][:6]
    last = None
    for s in reversed(sess.all_sessions(cfg)):
        meta = s.read()
        if meta.get("summary_md") or s.summary_md.exists():
            last = {"session": sess.base_id(s.id), "title": (meta.get("title") or s.id)[:120],
                    "url": meta.get("outline_url") or "", "started_at": meta.get("started_at", "")}
            break
    return {
        "recording": ({"session": rec.get("session_id"), "title": (rec.get("title") or "")[:120],
                       "started_at": rec.get("started_at")} if rec else None),
        "record_on_join": prefs.load(cfg)["record_on_join"],
        "next": nxt,
        "stats": stats.compute(cfg),
        "last": last,
    }


def open_url(url: str) -> None:
    if not url.startswith(("https://", "http://")):
        raise RuntimeError(f"not a web link: {url!r}")
    opener = shutil.which("xdg-open")
    if not opener:
        raise RuntimeError("xdg-open not found")
    subprocess.Popen([opener, url], start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
