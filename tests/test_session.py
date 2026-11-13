from datetime import datetime

import pytest

from minutes import cli
from minutes import session as sess


def test_create_layout_and_unique_ids(cfg):
    when = datetime(2026, 11, 13, 15, 30)
    a = sess.create(cfg, when)
    b = sess.create(cfg, when)
    assert a.id == "2026-11-13_1530"
    assert b.id == "2026-11-13_1530-2"
    assert a.audio_dir("me").is_dir() and a.audio_dir("them").is_dir()
    assert a.read()["id"] == a.id


def test_state_and_stage(cfg):
    s = sess.create(cfg)
    s.set_state(sess.RECORDING)
    s.set_stage("transcribe", "failed", error="boom")
    s.set_stage("transcribe", "done", backend="null")
    meta = s.read()
    assert meta["state"] == "recording"
    assert meta["stages"]["transcribe"]["status"] == "done"
    assert "error" not in meta["stages"]["transcribe"]
    assert not list(s.path.glob(".*.tmp"))


def test_resolve(cfg):
    a = sess.create(cfg, datetime(2026, 11, 12, 9, 0))
    b = sess.create(cfg, datetime(2026, 11, 13, 10, 0))
    assert sess.resolve(cfg).id == b.id
    assert sess.resolve(cfg, "2026-11-12").id == a.id
    with pytest.raises(LookupError):
        sess.resolve(cfg, "2026-11")
    with pytest.raises(LookupError):
        sess.resolve(cfg, "1999")


def test_cli_parses_all_commands():
    p = cli.build_parser()
    for argv in (["toggle"], ["stop", "--no-process"], ["transcribe", "x", "-b", "null", "-f"],
                 ["process"], ["open", "--print"], ["list", "-n", "3"], ["backends"]):
        assert p.parse_args(argv).func
