"""Calendar events for a recording, read from OmaCal's local database (read-only).

OmaCal (https://omacal.app) syncs Google Calendar / CalDAV into SQLite. Its schema is private to
the app, so every read here tolerates missing columns and a missing or locked database: no
calendar just means an unscheduled recording, never a failed one.

    [calendar]
    backend = "omacal"        # or "none"
    db = "~/.local/share/com.omacal.app/omacal.db"
    match_minutes = 10        # an event matches a recording started up to this far before/after it
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import Config

DEFAULT_DB = Path.home() / ".local/share/com.omacal.app/omacal.db"
MEET_CODE = re.compile(r"\b([a-z]{3}-[a-z]{4}-[a-z]{3})\b")


@dataclass
class Attendee:
    email: str
    name: str = ""
    response: str = ""
    optional: bool = False
    is_self: bool = False

    @property
    def label(self) -> str:
        return self.name or self.email.split("@")[0]


@dataclass
class Event:
    uid: str
    title: str
    start: float  # epoch seconds
    end: float
    description: str = ""
    location: str = ""
    conference: str = ""
    organizer: str = ""
    series: str = ""
    response: str = ""  # my RSVP
    attendees: list[Attendee] = field(default_factory=list)

    @property
    def others(self) -> list[Attendee]:
        return [a for a in self.attendees if not a.is_self and a.response != "declined"]

    @property
    def meet_code(self) -> str | None:
        m = MEET_CODE.search(self.conference or self.location or "")
        return m.group(1) if m and "meet.google.com" in (self.conference or self.location) else None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Event":
        d = dict(d)
        d["attendees"] = [Attendee(**a) for a in d.get("attendees", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def plain_text(desc: str, limit: int = 4000) -> str:
    """Google stores descriptions as HTML; keep the text and line breaks."""
    t = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</div>", "\n", desc or "")
    t = re.sub(r"(?i)<li[^>]*>", "- ", t)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(l.rstrip() for l in t.splitlines())).strip()
    return t[:limit]


class OmaCal:
    def __init__(self, db: Path = DEFAULT_DB):
        self.db = Path(db).expanduser()

    def _rows(self, where: str, params: tuple) -> list[dict]:
        if not self.db.exists():
            return []
        try:
            con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=2)
            con.row_factory = sqlite3.Row
            try:
                return [dict(r) for r in con.execute(f"SELECT * FROM events WHERE {where}", params)]
            finally:
                con.close()
        except sqlite3.Error:
            return []

    @staticmethod
    def _event(r: dict) -> Event | None:
        if r.get("is_all_day") or r.get("status") == "cancelled":
            return None
        try:
            people = json.loads(r.get("attendees_json") or "[]")
        except ValueError:
            people = []
        attendees = [Attendee(email=str(p.get("email", "")), name=str(p.get("display_name") or ""),
                              response=str(p.get("response_status") or ""), optional=bool(p.get("optional")),
                              is_self=bool(p.get("is_self")))
                     for p in people if isinstance(p, dict) and p.get("email")]
        uid = str(r.get("google_id") or r.get("ical_uid") or r.get("id"))
        series = str(r.get("recurring_event_id") or (uid if r.get("recurrence") else ""))
        return Event(uid=uid,
                     title=str(r.get("summary") or "").strip(),
                     start=r["start_utc"] / 1000, end=r["end_utc"] / 1000,
                     description=plain_text(r.get("description") or ""), location=str(r.get("location") or ""),
                     conference=str(r.get("conference_uri") or ""), organizer=str(r.get("organizer_email") or ""),
                     series=series, response=str(r.get("self_response") or ""),
                     attendees=attendees)

    @staticmethod
    def _occurrences(master: dict, t0: float, t1: float, moved: set[int]) -> list[dict]:
        """Rows for each occurrence of a recurring master overlapping [t0, t1]. OmaCal stores only
        the first occurrence plus moved/cancelled ones (`moved`: their original starts, ms)."""
        from dateutil.rrule import rrulestr

        try:
            tz = ZoneInfo(master.get("start_tz") or "UTC")
        except (ZoneInfoNotFoundError, ValueError):
            tz = timezone.utc
        dur = (master["end_utc"] - master["start_utc"]) / 1000
        dtstart = datetime.fromtimestamp(master["start_utc"] / 1000, tz)
        try:
            rule = rrulestr(str(master["recurrence"]).replace("\\n", "\n"), dtstart=dtstart,
                            forceset=True, unfold=True)
        except (ValueError, TypeError):
            return []
        lo = datetime.fromtimestamp(t0 - dur, timezone.utc)
        hi = datetime.fromtimestamp(t1, timezone.utc)
        out = []
        for occ in rule.between(lo, hi, inc=True)[:500]:
            start_ms = int(occ.timestamp() * 1000)
            if start_ms in moved:
                continue
            stamp = occ.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out.append({**master, "google_id": f"{master['google_id']}_{stamp}", "start_utc": start_ms,
                        "end_utc": start_ms + int(dur * 1000), "recurring_event_id": master["google_id"]})
        return out

    def _moved(self, series: list[str]) -> set[int]:
        if not series:
            return set()
        marks = ",".join("?" * len(series))
        rows = self._rows(f"recurring_event_id IN ({marks}) AND original_start_utc IS NOT NULL", tuple(series))
        return {r["original_start_utc"] for r in rows}

    def between(self, t0: float, t1: float) -> list[Event]:
        """Timed, non-cancelled events overlapping [t0, t1] (epoch seconds), recurrences expanded."""
        lo, hi = int(t0 * 1000), int(t1 * 1000)
        single = self._rows("recurrence IS NULL AND start_utc < ? AND end_utc > ?", (hi, lo))
        masters = self._rows("recurrence IS NOT NULL AND start_utc < ?", (hi,))
        moved = self._moved([m["google_id"] for m in masters])
        rows = single + [o for m in masters for o in self._occurrences(m, t0, t1, moved)]
        events = {e.uid: e for e in map(self._event, rows)
                  if e and e.start < t1 and e.end > t0}  # a shared event appears once per calendar
        return sorted(events.values(), key=lambda e: e.start)

    def get(self, uid: str) -> Event | None:
        rows = self._rows("google_id = ? OR ical_uid = ?", (uid, uid))
        e = next((e for e in map(self._event, rows) if e), None)
        if e or "_" not in uid:
            return e
        master_id, _, stamp = uid.rpartition("_")  # an expanded occurrence: <master>_<UTC start>
        try:
            at = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
        return next((e for e in self.between(at, at + 60) if e.uid == uid), None)


def source(cfg: Config) -> OmaCal | None:
    opts = cfg.calendar
    if opts.get("backend", "omacal") != "omacal":
        return None
    return OmaCal(Path(opts.get("db", DEFAULT_DB)))


def window_titles() -> list[str]:
    try:
        r = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=3)
        return [str(c.get("title", "")) for c in json.loads(r.stdout or "[]")]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def score(e: Event, now: float, titles: list[str]) -> float:
    s = 0.0
    low = [t.lower() for t in titles]
    if e.meet_code and any(e.meet_code in t for t in low):
        s += 100  # the call itself is open in a window
    elif len(e.title) >= 4 and any(e.title.lower() in t for t in low):
        s += 50
    if e.conference:
        s += 5
    if e.others:
        s += 3
    if e.response == "declined":
        s -= 20
    elif e.response in ("accepted", ""):
        s += 2
    return s - abs(e.start - now) / 600  # closer start wins ties (-1 per 10 min)


def match(cfg: Config, now: float | None = None, titles: list[str] | None = None) -> Event | None:
    """The calendar event this recording most likely belongs to, or None."""
    cal = source(cfg)
    if not cal:
        return None
    now = time.time() if now is None else now
    slack = float(cfg.calendar.get("match_minutes", 10)) * 60
    candidates = [e for e in cal.between(now - slack, now + slack)
                  if e.start - slack <= now <= e.end + 60]
    if not candidates:
        return None
    titles = window_titles() if titles is None else titles
    best = max(candidates, key=lambda e: score(e, now, titles))
    return best if score(best, now, titles) > -10 else None


def upcoming(cfg: Config, now: float | None = None, hours: float = 12) -> list[Event]:
    cal = source(cfg)
    now = time.time() if now is None else now
    return [e for e in cal.between(now, now + hours * 3600) if e.response != "declined"] if cal else []


# -- the event a session belongs to: a frozen copy in <session>/meeting.json ----------------------
def load(s) -> Event | None:
    p = s.path / "meeting.json"
    try:
        return Event.from_json(json.loads(p.read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError, TypeError):
        return None


def save(s, e: Event | None) -> None:
    from . import session as sess

    p = s.path / "meeting.json"
    if e is None:
        p.unlink(missing_ok=True)
        s.update(event=None)
        return
    sess.atomic_write_json(p, e.to_json())
    s.update(event={"uid": e.uid, "title": e.title})


def describe(e: Event) -> str:
    """The calendar facts given to Claude next to the transcript."""
    lines = [f"Title: {e.title}",
             f"Scheduled: {time.strftime('%Y-%m-%d %H:%M', time.localtime(e.start))}"
             f"-{time.strftime('%H:%M', time.localtime(e.end))}"]
    me = next((a for a in e.attendees if a.is_self), None)
    if me:
        lines.append(f"Me (the recorder): {me.label}")
    if e.organizer:
        org = next((a.label for a in e.attendees if a.email == e.organizer), e.organizer.split("@")[0])
        lines.append(f"Organizer: {org}")
    if e.others:
        people = [a.label + (" (optional)" if a.optional else "") + (" (not answered)" if a.response == "needsAction" else "")
                  for a in e.others]
        lines.append("Other invitees (not all may have joined): " + ", ".join(people))
        if len(e.others) == 1:
            lines.append(f'This is a 1:1, so "Them" is {e.others[0].label}.')
    if e.description:
        lines += ["Agenda / description:", e.description]
    return "\n".join(lines)
