"""Time spent in recorded meetings: today, this week (Mon-Sun), this month, all time."""

from __future__ import annotations

from datetime import datetime, timedelta

from . import session as sess
from .config import Config


def _started(meta: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(meta["started_at"]).astimezone()
    except (KeyError, TypeError, ValueError):
        return None


def compute(cfg: Config, now: datetime | None = None, min_seconds: float = 60) -> dict:
    """Recordings shorter than `min_seconds` (tests, accidental presses) are not meetings."""
    now = (now or datetime.now()).astimezone()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    starts = {"today": day, "week": day - timedelta(days=day.weekday()), "month": day.replace(day=1)}
    out = {k: {"seconds": 0.0, "meetings": 0} for k in (*starts, "all")}
    for s in sess.all_sessions(cfg):
        meta = s.read()
        when, dur = _started(meta), meta.get("duration_s")
        if not when or not dur or dur < min_seconds:
            continue
        for k, since in starts.items():
            if when >= since:
                out[k]["seconds"] += dur
                out[k]["meetings"] += 1
        out["all"]["seconds"] += dur
        out["all"]["meetings"] += 1
    for v in out.values():
        v["seconds"] = round(v["seconds"])
    return out


def hm(seconds: float) -> str:
    m = int(seconds) // 60
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m}m"
