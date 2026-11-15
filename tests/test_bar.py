"""`minutes bar` state, the record-on-join setting and `minutes join`, without a browser or recorder."""

import time
from types import SimpleNamespace

from minutes import bar, cli, prefs
from test_calendar_outline import cal_cfg, ms


def test_prefs_default_and_toggle(cfg):
    assert prefs.load(cfg)["record_on_join"] is True
    prefs.set_(cfg, "record_on_join", False)
    assert prefs.load(cfg)["record_on_join"] is False


def test_state_marks_meetings_joinable_from_five_minutes_before(cfg, tmp_path):
    now = time.time()
    cal_cfg(cfg, tmp_path, [
        dict(google_id="soon", summary="Soon", start_utc=ms(now + 240), end_utc=ms(now + 2000),
             conference_uri="https://meet.google.com/abc-defg-hij"),
        dict(google_id="later", summary="Later", start_utc=ms(now + 3600), end_utc=ms(now + 5400),
             conference_uri="https://meet.google.com/abc-defg-hik"),
        dict(google_id="nolink", summary="In person", start_utc=ms(now - 60), end_utc=ms(now + 600)),
    ])
    st = bar.state(cfg, now=now)
    got = {e["title"]: e["joinable"] for e in st["next"]}
    assert got == {"In person": False, "Soon": True, "Later": False}
    assert st["recording"] is None and st["record_on_join"] is True and "today" in st["stats"]


def _join(cfg, tmp_path, monkeypatch, record=None):
    now = time.time()
    tmp_path = tmp_path / str(time.monotonic_ns())  # one calendar file per call
    tmp_path.mkdir()
    cal_cfg(cfg, tmp_path, [dict(google_id="m", summary="Design sync", start_utc=ms(now), end_utc=ms(now + 1800),
                                 conference_uri="https://meet.google.com/abc-defg-hij")])
    opened, started = [], []
    monkeypatch.setattr(cli, "_cfg", lambda: cfg)
    monkeypatch.setattr(bar, "open_url", opened.append)
    monkeypatch.setattr(cli, "cmd_start", lambda args: started.append(args.trigger) or 0)
    cli.cmd_join(SimpleNamespace(event="m", record=record))
    return opened, started


def test_join_opens_link_and_records_when_toggle_on(cfg, tmp_path, monkeypatch):
    opened, started = _join(cfg, tmp_path, monkeypatch)
    assert opened == ["https://meet.google.com/abc-defg-hij"] and started == ["join"]


def test_join_only_when_toggle_off_or_right_click(cfg, tmp_path, monkeypatch):
    opened, started = _join(cfg, tmp_path, monkeypatch, record=False)
    assert opened and started == []
    prefs.set_(cfg, "record_on_join", False)
    opened, started = _join(cfg, tmp_path, monkeypatch)
    assert opened and started == []


def test_open_url_refuses_non_web_links():
    import pytest

    with pytest.raises(RuntimeError):
        bar.open_url("file:///etc/passwd")
