"""`minutes` command line. Each subcommand is one pipeline stage (or a convenience wrapper)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as config_mod
from . import session as sess


def _cfg() -> config_mod.Config:
    return config_mod.load()


def _event_arg(cfg, uid: str | None):
    from . import calendar

    if not uid:
        return None
    cal = calendar.source(cfg)
    e = cal.get(uid) if cal else None
    if e is None:
        raise LookupError(f"no calendar event {uid!r} (see `minutes events`)")
    return e


def cmd_start(args) -> int:
    from . import capture

    cfg = _cfg()
    event = _event_arg(cfg, getattr(args, "event", None))
    s = capture.start(cfg, event=event, match=not getattr(args, "no_match", False),
                      trigger=getattr(args, "trigger", None) or "manual")
    title = s.read().get("event", {}) or {}
    print(f"recording {s.id}  ({s.path})" + (f"  · {title.get('title')}" if title.get("title") else ""))
    if cfg.live and not getattr(args, "no_live", False):
        _spawn(s, ["live", s.id], "live")
    return 0


def cmd_stop(args) -> int:
    from . import capture

    s = capture.stop(_cfg())
    if s is None:
        print("not recording")
        return 1
    print(f"stopped {s.id}: {len(s.chunks('me'))} me / {len(s.chunks('them'))} them chunks")
    if not args.no_process:
        from . import live

        if live.running(s):
            print(f"live transcriber is finishing {s.id} (transcript + summary)")
        else:
            _spawn(s, ["process", s.id], "process")
    return 0


def cmd_live(args) -> int:
    from . import live

    cfg = _cfg()
    return 0 if live.run(cfg, sess.resolve(cfg, args.session), backend=args.backend) else 1


def cmd_toggle(args) -> int:
    from . import capture

    cfg = _cfg()
    if capture.is_recording(cfg):
        args.no_process = False
        return cmd_stop(args)
    return cmd_start(args)


def cmd_status(args) -> int:
    from . import capture

    cfg = _cfg()
    rec = capture.current(cfg)
    if rec:
        print(f"recording {rec['session_id']} since {rec['started_at']}")
    else:
        print("idle")
    return 0


def cmd_list(args) -> int:
    from . import capture

    cfg = _cfg()
    capture.recover(cfg)
    for s in sess.all_sessions(cfg)[-args.limit :]:
        meta = s.read()
        stages = meta.get("stages", {})
        flags = " ".join(
            f"{k}:{v.get('status')}" for k, v in stages.items() if k in ("transcribe", "summarize")
        )
        dur = meta.get("duration_s")
        dur_s = f"{int(dur) // 60:>3}m{int(dur) % 60:02d}s" if dur else "      -"
        title = meta.get("title") or ("(no summary yet)" if not s.summary_md.exists() else "")
        print(f"{sess.base_id(s.id):<18} {meta.get('state') or '-':<12} {dur_s}  {title}"
              + (f"  [{flags}]" if "failed" in flags or "running" in flags else ""))
    return 0


def cmd_transcribe(args) -> int:
    from . import transcribe

    cfg = _cfg()
    s = sess.resolve(cfg, args.session)
    stats = transcribe.run(cfg, s, backend=args.backend, force=args.force)
    print(stats.describe())
    return 0 if not stats.failed else 2


def cmd_merge(args) -> int:
    from . import merge

    cfg = _cfg()
    s = sess.resolve(cfg, args.session)
    path = merge.run(cfg, s, backend=args.backend)
    print(path)
    return 0


def cmd_summarize(args) -> int:
    from . import summarize

    cfg = _cfg()
    s = sess.resolve(cfg, args.session)
    s = summarize.run(cfg, s, backend=args.backend)
    print(s.summary_md)
    return 0


def cmd_process(args) -> int:
    from . import pipeline

    cfg = _cfg()
    s = sess.resolve(cfg, args.session)
    ok = pipeline.process(cfg, s, asr_backend=args.backend, force=args.force)
    return 0 if ok else 2


def cmd_link(args) -> int:
    """Show, set or clear the calendar event a session belongs to."""
    import time as _time

    from . import calendar

    cfg = _cfg()
    s = sess.resolve(cfg, args.session)
    if args.none:
        calendar.save(s, None)
        print(f"{s.id}: unlinked")
        return 0
    if args.event:
        e = _event_arg(cfg, args.event)
    else:
        cur = calendar.load(s)
        started = s.read().get("started_epoch") or _time.time()
        cal = calendar.source(cfg)
        near = cal.between(started - 3 * 3600, started + 3 * 3600) if cal else []
        print(f"{s.id}: " + (f"linked to {cur.title!r} ({cur.uid})" if cur else "not linked"))
        for ev in near:
            print(f"  {_time.strftime('%a %H:%M', _time.localtime(ev.start))}  {ev.uid}  {ev.title}")
        if near:
            print(f"link with: minutes link {sess.base_id(s.id)} --event <uid>")
        return 0
    calendar.save(s, e)
    print(f"{s.id}: linked to {e.title!r}; re-run `minutes summarize {sess.base_id(s.id)}` "
          f"and `minutes publish {sess.base_id(s.id)}` to use it")
    return 0


def cmd_events(args) -> int:
    import json as _json
    import time as _time

    from . import calendar

    cfg = _cfg()
    evs = calendar.upcoming(cfg, hours=args.hours)
    if args.json:
        print(_json.dumps([{"uid": e.uid, "title": e.title, "start": e.start, "end": e.end,
                            "conference": e.conference, "people": len(e.others)} for e in evs]))
        return 0
    for e in evs:
        when = _time.strftime("%a %H:%M", _time.localtime(e.start))
        print(f"{when}  {e.title}  ({len(e.others)} others){'  ' + e.conference if e.conference else ''}"
              f"\n         {e.uid}")
    return 0


def cmd_clean(args) -> int:
    from . import clean

    cfg = _cfg()
    s = clean.run(cfg, sess.resolve(cfg, args.session))
    print(s.clean_md)
    return 0


def cmd_publish(args) -> int:
    from . import outline

    cfg = _cfg()
    if not outline.enabled(cfg):
        raise RuntimeError("Outline publishing is off; set [outline] vault (and folder) in "
                           f"{config_mod.config_path()}")
    s = sess.resolve(cfg, args.session)
    path, url = outline.publish(cfg, s, wait=not args.no_wait)
    print(path)
    print(url or "no Outline link yet (is the sync daemon running?)")
    return 0


def cmd_stats(args) -> int:
    import json as _json

    from . import stats

    st = stats.compute(_cfg())
    if args.json:
        print(_json.dumps(st))
        return 0
    for k, label in (("today", "Today"), ("week", "This week"), ("month", "This month"), ("all", "All time")):
        print(f"{label:<11} {stats.hm(st[k]['seconds']):>8}  {st[k]['meetings']} meeting(s)")
    return 0


def cmd_bar(args) -> int:
    import json as _json

    from . import bar

    print(_json.dumps(bar.state(_cfg()), ensure_ascii=False))
    return 0


def cmd_join(args) -> int:
    """Open the meeting's call link; record it too if `record_on_join` is on (and not --no-record)."""
    from . import bar, capture, prefs

    cfg = _cfg()
    e = _event_arg(cfg, args.event)
    if not e.conference:
        raise RuntimeError(f"{e.title!r} has no call link")
    bar.open_url(e.conference)
    print(f"opened {e.conference}")
    record = prefs.load(cfg)["record_on_join"] if args.record is None else args.record
    if record and not capture.is_recording(cfg):
        args.no_live, args.no_match, args.trigger = False, True, "join"
        return cmd_start(args)
    return 0


def cmd_watch(args) -> int:
    from . import watch

    watch.run(_cfg())
    return 0


def cmd_set(args) -> int:
    from . import prefs

    value = args.value.lower() in ("1", "true", "on", "yes")
    print(prefs.set_(_cfg(), args.key, value))
    return 0


def cmd_open(args) -> int:
    cfg = _cfg()
    s = sess.resolve(cfg, args.session)
    target = s.summary_md if s.summary_md.exists() else s.transcript_md if s.transcript_md.exists() else s.path
    if args.print or not shutil.which("xdg-open"):
        print(target)
        return 0
    subprocess.Popen(["xdg-open", str(target)], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def cmd_backends(args) -> int:
    from . import asr, summarize

    cfg = _cfg()
    print("asr:       " + ", ".join(
        f"{n}{' (default)' if n == cfg.asr_backend else ''}" for n in asr.available()))
    print("summary:   " + ", ".join(
        f"{n}{' (default)' if n == cfg.summary_backend else ''}" for n in summarize.available()))
    return 0


def cmd_setup(args) -> int:
    """Install the Tara engine venv and its OpenVINO model (idempotent)."""
    from .asr import tara

    cfg = _cfg()
    p = tara.paths(cfg)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv not found (needed to build the engine venv)")
    py = p["python"]
    if not py.exists():
        print(f"creating engine venv {p['engine']} (Python 3.13)")
        subprocess.run([uv, "venv", "--no-project", "--python", "3.13", str(p["engine"])], check=True)
    print("installing engine packages (torch CPU build, then OpenVINO)")
    subprocess.run([uv, "pip", "install", "--python", str(py), *tara.TORCH], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(py), "-r", str(tara.REQUIREMENTS)], check=True)
    # optimum pulls in a PyPI torchvision built against a different torch; importing it breaks
    # transformers ("operator torchvision::nms does not exist"). Tara doesn't need it.
    subprocess.run([uv, "pip", "uninstall", "--python", str(py), "torchvision"], check=False)
    md = p["model_dir"]
    if (md / "openvino_encoder_model.xml").exists():
        print(f"model present: {md}")
    elif args.from_dir:
        src = Path(args.from_dir).expanduser()
        md.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                subprocess.run(["cp", "--reflink=auto", str(f), str(md / f.name)], check=True)
        print(f"copied model from {src} to {md}")
    else:
        print(f"exporting Trelis/tara to OpenVINO in {md} (downloads ~3 GB)")
        md.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(py), "-I", str(tara.WORKER), "--model-dir", str(md), "--export"], check=True,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    print("checking the engine loads ...")
    t = tara.TaraTranscriber(cfg)
    try:
        t._start("local")
        print(f"tara ready on {t.engine_label}")
    finally:
        t.close()
    return 0


def _spawn(s: sess.Session, minutes_args: list[str], kind: str) -> None:
    """Run `minutes <args>` detached so it outlives the hotkey invocation."""
    cmd = [sys.executable, "-m", "minutes", *minutes_args]
    if shutil.which("systemd-run"):
        unit = f"minutes-{kind}-{s.id}".replace("_", "-")
        full = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}",
                "--setenv=PATH", "--setenv=HOME", *(
                    [f"--setenv=XDG_CONFIG_HOME"] if "XDG_CONFIG_HOME" in os.environ else []), *(
                    [f"--setenv=XDG_DATA_HOME"] if "XDG_DATA_HOME" in os.environ else []), *cmd]
        r = subprocess.run(full, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            print(f"{kind} in background: journalctl --user -u {unit}")
            return
        print(f"systemd-run failed ({r.stderr.strip()}); falling back to a detached process", file=sys.stderr)
    log = (s.logs_dir / f"{kind}.log").open("ab")
    subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    print(f"{kind} in background: {s.logs_dir / f'{kind}.log'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="minutes", description="Record both sides of a meeting, transcribe, summarize.")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=fn)
        return sp

    add("toggle", cmd_toggle, "start recording, or stop and process (bind this to a hotkey)")
    st = add("start", cmd_start, "start recording mic + system audio")
    st.add_argument("--no-live", action="store_true", help="don't transcribe while recording")
    st.add_argument("--event", help="calendar event uid to link (see `minutes events`)")
    st.add_argument("--no-match", action="store_true", help="don't look up the calendar")
    st.add_argument("--trigger", help=argparse.SUPPRESS)
    add("stop", cmd_stop, "stop recording (then process in background)").add_argument(
        "--no-process", action="store_true", help="only stop; don't transcribe/summarize")
    add("status", cmd_status, "show whether a recording is running")
    add("list", cmd_list, "list sessions").add_argument("-n", "--limit", type=int, default=20)

    for name, fn, help_ in [
        ("transcribe", cmd_transcribe, "transcribe audio chunks (resumable)"),
        ("merge", cmd_merge, "merge Me/Them transcripts into transcript.md"),
        ("summarize", cmd_summarize, "summarize transcript.md into summary.md"),
        ("process", cmd_process, "transcribe + merge + summarize"),
    ]:
        sp = add(name, fn, help_)
        sp.add_argument("session", nargs="?", help="session id or prefix (default: latest)")
        sp.add_argument("-b", "--backend", help="backend name (see `minutes backends`)")
        if name in ("transcribe", "process"):
            sp.add_argument("-f", "--force", action="store_true", help="redo chunks already transcribed")

    for name, fn, help_ in [("clean", cmd_clean, "tidy the full transcript with Claude (transcript.clean.md)"),
                            ("publish", cmd_publish, "write the Outline note (summary + transcript)")]:
        sp = add(name, fn, help_)
        sp.add_argument("session", nargs="?", help="session id or prefix (default: latest)")
        if name == "publish":
            sp.add_argument("--no-wait", action="store_true", help="don't wait for the Outline link")
    lk = add("link", cmd_link, "show or set the calendar event of a session")
    lk.add_argument("session", nargs="?")
    g = lk.add_mutually_exclusive_group()
    g.add_argument("--event", help="event uid")
    g.add_argument("--none", action="store_true", help="unlink")
    ev = add("events", cmd_events, "upcoming calendar events")
    ev.add_argument("--hours", type=float, default=12)
    ev.add_argument("--json", action="store_true")
    sa = add("stats", cmd_stats, "time spent in meetings")
    sa.add_argument("--json", action="store_true")

    add("bar", cmd_bar, "state for the bar widget, as JSON")
    jn = add("join", cmd_join, "open a meeting's call link (and record it if record_on_join is on)")
    jn.add_argument("event", help="event uid (see `minutes events`)")
    rg = jn.add_mutually_exclusive_group()
    rg.add_argument("--record", dest="record", action="store_true", default=None, help="record regardless")
    rg.add_argument("--no-record", dest="record", action="store_false", help="only open the link")
    add("watch", cmd_watch, "detect calls and offer to record them (run as a user service)")
    se = add("set", cmd_set, "change a bar setting, e.g. `minutes set record_on_join off`")
    se.add_argument("key", choices=["record_on_join"])
    se.add_argument("value", choices=["on", "off", "true", "false"])

    lv = add("live", cmd_live, "transcribe while recording, then finish (started by `start`)")
    lv.add_argument("session", nargs="?")
    lv.add_argument("-b", "--backend")

    op = add("open", cmd_open, "open summary.md (or transcript / folder)")
    op.add_argument("session", nargs="?")
    op.add_argument("--print", action="store_true", help="print the path instead of opening it")
    add("backends", cmd_backends, "list available ASR and summary backends")
    su = add("setup", cmd_setup, "install a local ASR engine (tara)")
    su.add_argument("engine", choices=["tara"])
    su.add_argument("--from", dest="from_dir", help="copy an existing OpenVINO export instead of exporting")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LookupError, RuntimeError) as e:
        print(f"minutes: {e}", file=sys.stderr)
        return 1
