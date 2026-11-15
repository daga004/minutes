"""Calendar matching, cleaned transcript, Outline note and stats, all on synthetic data."""

import json
import sqlite3
from datetime import datetime, timezone

from minutes import calendar, clean, outline, stats, summarize
from minutes import session as sess
from minutes.asr.base import Segment
from minutes import merge

T0 = datetime(2026, 11, 16, 4, 30, tzinfo=timezone.utc).timestamp()  # Mon 10:00 IST

SCHEMA = """CREATE TABLE events (id INTEGER PRIMARY KEY, google_id TEXT, ical_uid TEXT, summary TEXT,
 description TEXT, location TEXT, start_utc INTEGER, end_utc INTEGER, start_tz TEXT, is_all_day INTEGER,
 recurrence TEXT, recurring_event_id TEXT, original_start_utc INTEGER, status TEXT, organizer_email TEXT,
 self_response TEXT, conference_uri TEXT, attendees_json TEXT)"""

PEOPLE = json.dumps([
    {"email": "me@example.com", "display_name": "Asha", "response_status": "accepted", "is_self": True},
    {"email": "ravi@example.com", "display_name": "Ravi Kumar", "response_status": "accepted"},
])


def make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    for r in rows:
        base = dict(ical_uid=None, description="", location="", start_tz="Asia/Kolkata", is_all_day=0,
                    recurrence=None, recurring_event_id=None, original_start_utc=None, status="confirmed",
                    organizer_email="ravi@example.com", self_response="accepted", conference_uri="",
                    attendees_json=PEOPLE)
        base.update(r)
        con.execute(f"INSERT INTO events ({','.join(base)}) VALUES ({','.join('?' * len(base))})",
                    tuple(base.values()))
    con.commit()
    con.close()


def ms(t):
    return int(t * 1000)


def cal_cfg(cfg, tmp_path, rows):
    db = tmp_path / "omacal.db"
    make_db(db, rows)
    cfg.calendar = {"backend": "omacal", "db": str(db)}
    return cfg


def test_weekly_series_expands_and_moved_occurrence_wins(cfg, tmp_path):
    week = 7 * 86400
    cal_cfg(cfg, tmp_path, [
        dict(google_id="drone", summary="Drone sync", start_utc=ms(T0 - week), end_utc=ms(T0 - week + 1800),
             recurrence="RRULE:FREQ=WEEKLY;BYDAY=MO", conference_uri="https://meet.google.com/abc-defg-hij"),
        # this week's occurrence moved 15 minutes earlier
        dict(google_id="drone_20261116T043000Z", summary="Drone sync", start_utc=ms(T0 - 900),
             end_utc=ms(T0 + 900), recurring_event_id="drone", original_start_utc=ms(T0)),
    ])
    evs = calendar.OmaCal(tmp_path / "omacal.db").between(T0 - 3600, T0 + week + 3600)
    assert [(e.uid, e.start - T0) for e in evs] == [
        ("drone_20261116T043000Z", -900), ("drone_20261123T043000Z", week)]
    assert all(e.series == "drone" for e in evs)
    nxt = calendar.OmaCal(tmp_path / "omacal.db").get("drone_20261123T043000Z")
    assert nxt and nxt.start == T0 + week and nxt.others[0].label == "Ravi Kumar"


def test_match_prefers_the_open_meet_window(cfg, tmp_path):
    cal_cfg(cfg, tmp_path, [
        dict(google_id="a", summary="Budget review", start_utc=ms(T0), end_utc=ms(T0 + 3600),
             conference_uri="https://meet.google.com/aaa-bbbb-ccc"),
        dict(google_id="b", summary="Hiring sync", start_utc=ms(T0), end_utc=ms(T0 + 3600),
             conference_uri="https://meet.google.com/xyz-wxyz-xyz"),
        dict(google_id="c", summary="Cancelled thing", start_utc=ms(T0), end_utc=ms(T0 + 3600), status="cancelled"),
    ])
    e = calendar.match(cfg, now=T0 + 120, titles=["Meet - xyz-wxyz-xyz - Helium"])
    assert e.uid == "b"
    assert calendar.match(cfg, now=T0 + 5 * 3600, titles=[]) is None  # nothing near


def test_no_calendar_is_not_an_error(cfg, tmp_path):
    cfg.calendar = {"backend": "omacal", "db": str(tmp_path / "missing.db")}
    assert calendar.match(cfg, now=T0, titles=[]) is None


def test_describe_and_html_agenda():
    e = calendar.Event(uid="x", title="Drone sync", start=T0, end=T0 + 1800,
                       description="<p>1. Survey <b>dates</b></p><ul><li>Vendor</li></ul>",
                       organizer="ravi@example.com",
                       attendees=[calendar.Attendee("me@example.com", "Asha", "accepted", is_self=True),
                                  calendar.Attendee("ravi@example.com", "Ravi Kumar", "accepted")])
    e.description = calendar.plain_text(e.description)
    text = calendar.describe(e)
    assert "Organizer: Ravi Kumar" in text and '"Them" is Ravi Kumar' in text
    assert "1. Survey dates" in text and "- Vendor" in text and "<" not in text


def _session_with_speech(cfg, event=None):
    s = sess.create(cfg)
    s.set_state(sess.RECORDED, chunk_seconds=30, started_at="2026-11-16T10:00:05+05:30", duration_s=1500)
    sess.atomic_write_json(s.transcript_dir("x", "me") / "00000.json",
                           {"segments": [Segment(0, 2, "अः drone survey kab hai").to_dict()]})
    sess.atomic_write_json(s.transcript_dir("x", "them") / "00000.json",
                           {"segments": [Segment(5, 9, "next monday, vendor confirm karega").to_dict()]})
    merge.run(cfg, s, backend="x")
    if event:
        calendar.save(s, event)
    return s


EVENT = calendar.Event(uid="drone_1", title="Drone sync", start=T0, end=T0 + 1800, series="drone",
                       attendees=[calendar.Attendee("me@example.com", "Asha", is_self=True),
                                  calendar.Attendee("ravi@example.com", "Ravi Kumar", "accepted")])


class Fake:
    name = "fake"

    def __init__(self):
        self.prompts = []

    def summarize(self, text):
        self.prompts.append(text)
        return "# Survey dates and vendor\n\n## Summary\nSurvey next Monday.\n"

    def complete(self, prompt):
        self.prompts.append(prompt)
        return ("```\n**[00:00] Me:** drone survey kab hai\n\n"
                "**[00:05] Ravi Kumar:** next monday, vendor confirm karega\n```")


def test_summary_uses_calendar_title_and_context(cfg):
    s = _session_with_speech(cfg, EVENT)
    fake = Fake()
    s = summarize.run(cfg, s, summarizer=fake)
    assert "<meeting>" in fake.prompts[0] and "Ravi Kumar" in fake.prompts[0]
    assert s.id.endswith("_drone-sync")
    md = s.summary_md.read_text()
    assert md.startswith("# Drone sync\n") and "**Survey dates and vendor**" in md and "with Ravi Kumar" in md


def test_clean_keeps_lines_and_renames_them_in_a_1to1(cfg):
    s = _session_with_speech(cfg, EVENT)
    fake = Fake()
    clean.run(cfg, s, completer=fake)
    assert 'instead of "Them"' in fake.prompts[0] and "Do not translate" in fake.prompts[0]
    out = s.clean_md.read_text()
    assert "**[00:05] Ravi Kumar:** next monday" in out and "```" not in out
    assert s.transcript_md.read_text().count("अः") == 1  # raw transcript untouched


def test_clean_parts_split_long_transcripts():
    blocks = [f"**[00:{i:02d}] Me:** " + "x" * 1000 for i in range(20)]
    chunks = clean.parts(blocks, limit=6000)
    assert len(chunks) == 4 and sum(map(len, chunks)) == 20


def test_outline_note_and_republish_keeps_sync_keys(cfg, tmp_path):
    s = _session_with_speech(cfg, EVENT)
    fake = Fake()
    s = summarize.run(cfg, s, summarizer=fake)
    clean.run(cfg, s, completer=fake)
    cfg.outline = {"vault": str(tmp_path / "vault"), "folder": "Me/meetings", "copy_link": False}
    path, url = outline.publish(cfg, s, wait=False)
    assert path == tmp_path / "vault/Me/meetings/2026-11-16-drone-sync.md" and url is None
    text = path.read_text()
    assert outline.write_local(cfg, s).read_text() == text  # notes.md in the session = the Outline note
    assert text.startswith("---\ntype: Meeting\ndate: 2026-11-16\n")
    assert '  - "Ravi Kumar"' in text and "\n# Drone sync\n" in text
    assert text.index("## Summary") < text.index("## Transcript") < text.index("**[00:05] Ravi Kumar:**")
    # the sync daemon adds its keys; a re-publish rewrites the body but keeps them and the URL
    synced = text.replace("---\n# Drone", "outline_id: abc\noutline_url: https://docs.example/doc/x\n---\n# Drone", 1)
    path.write_text(synced)
    path2, url2 = outline.publish(cfg, s, wait=False)
    assert path2 == path and url2 == "https://docs.example/doc/x"
    assert path.read_text().count("outline_id: abc") == 1
    assert s.read()["outline_url"] == url2


def test_stats_ignores_short_tests(cfg):
    now = datetime(2026, 11, 18, 12, 0).astimezone()  # Wednesday
    for started, dur in (("2026-11-18T09:00:00", 1800), ("2026-11-16T09:00:00", 3600),
                         ("2026-11-02T09:00:00", 600), ("2026-11-18T10:00:00", 20)):
        s = sess.create(cfg)
        s.update(started_at=datetime.fromisoformat(started).astimezone().isoformat(), duration_s=dur)
    st = stats.compute(cfg, now=now)
    assert st["today"] == {"seconds": 1800, "meetings": 1}
    assert st["week"] == {"seconds": 5400, "meetings": 2}
    assert st["month"]["seconds"] == 6000 and st["all"]["meetings"] == 3
    assert stats.hm(5400) == "1h 30m" and stats.hm(600) == "10m"
