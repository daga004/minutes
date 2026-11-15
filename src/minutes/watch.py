"""`minutes watch`: notice calls (any app using the mic) and offer to record them.

A call is an app holding a microphone stream (PipeWire source-output) for `min_seconds`.
Dictation tools and minutes' own recorder are ignored. When a call starts and nothing is being
recorded, a notification asks "Record / Ignore" once per call. A recording that belongs to a call
(started from this prompt or the bar's Join, or running while a call was seen) is stopped when
the call ends: no call stream and no call window for `end_grace_seconds`.

    [watch]
    min_seconds = 15
    end_grace_seconds = 60
    auto_stop = true
    ignore = ["pw-record", "parecord", "arecord", "superwhisper"]   # app or binary names
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

from . import calendar
from .config import Config

IGNORE = ["pw-record", "parecord", "arecord", "superwhisper", "minutes-me", "minutes-them", "minutes"]
CALL_TITLE = re.compile(r"(meet - |\b[a-z]{3}-[a-z]{4}-[a-z]{3}\b|huddle|zoom meeting|microsoft teams|"
                        r"whatsapp.*(call|voice|video)|(call|voice|video).*whatsapp)", re.I)
BROWSERS = re.compile(r"helium|chrom|brave|firefox|vivaldi|edge|opera|zen", re.I)


@dataclass
class Stream:
    app: str  # application.name, else binary
    binary: str = ""


@dataclass
class Call:
    app: str
    since: float
    last_seen: float
    asked: bool = False


@dataclass
class Detector:
    min_seconds: float = 15
    end_grace: float = 60
    auto_stop: bool = True
    ignore: list[str] = field(default_factory=lambda: list(IGNORE))
    calls: dict[str, Call] = field(default_factory=dict)
    call_window_seen: float = 0.0
    call_heard: float = 0.0  # last poll with a call app holding the mic
    tied: str | None = None  # session id of a recording that belongs to a call
    stopped: set[str] = field(default_factory=set)  # recordings this detector already stopped

    def relevant(self, s: Stream) -> bool:
        names = {s.app.lower(), s.binary.lower()}
        return not any(n and (n == i.lower() or n.startswith("minutes")) for n in names for i in self.ignore)

    def step(self, now: float, streams: list[Stream], titles: list[str], recording: dict | None) -> list[tuple]:
        """Advance one poll. Returns actions: ("ask", app, titles), ("stop", session_id)."""
        actions: list[tuple] = []
        live = {s.app or s.binary for s in streams if self.relevant(s)}
        for app in live:
            c = self.calls.setdefault(app, Call(app, since=now, last_seen=now))
            c.last_seen = now
            self.call_heard = now
        call_window = any(CALL_TITLE.search(t or "") for t in titles)
        if call_window:
            self.call_window_seen = now
        for app, c in list(self.calls.items()):
            if app not in live and now - c.last_seen >= self.end_grace:
                del self.calls[app]  # that call is over; a new one will be asked about again
        in_call = any(c.last_seen == now and now - c.since >= self.min_seconds for c in self.calls.values())

        rec_id = recording.get("session_id") if recording else None
        if rec_id and rec_id != self.tied and rec_id not in self.stopped and (in_call or recording.get("trigger") in ("join", "detector")):
            self.tied = rec_id
        if not rec_id:
            self.tied = None
        if in_call and not rec_id:
            for c in self.calls.values():
                if not c.asked and c.last_seen == now and now - c.since >= self.min_seconds:
                    c.asked = True
                    actions.append(("ask", c.app, titles))
                    break
        if in_call and rec_id:
            for c in self.calls.values():
                c.asked = True  # already recording: nothing to ask
        quiet_since = max(self.call_heard, self.call_window_seen)
        if (self.auto_stop and rec_id and rec_id == self.tied and not live and not call_window
                and quiet_since and now - quiet_since >= self.end_grace):
            actions.append(("stop", rec_id))
            self.stopped.add(rec_id)
            self.tied = None
        return actions


# -- the real world ------------------------------------------------------------------------------
def mic_streams() -> list[Stream]:
    """Apps recording from a real input (not a sink monitor)."""
    def pactl(*args):
        r = subprocess.run(["pactl", "-f", "json", *args], capture_output=True, text=True, timeout=5)
        return json.loads(r.stdout or "[]")

    try:
        monitors = {s["index"] for s in pactl("list", "sources")
                    if str(s.get("name", "")).endswith(".monitor") or s.get("monitor_of_sink") not in (None, "n/a")}
        out = []
        for o in pactl("list", "source-outputs"):
            if o.get("source") in monitors:
                continue
            p = o.get("properties", {})
            out.append(Stream(app=str(p.get("application.name") or ""),
                              binary=str(p.get("application.process.binary") or "")))
        return out
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def describe(app: str, titles: list[str], event) -> tuple[str, str]:
    """(notification title, body) for a detected call."""
    low = " | ".join(titles).lower()
    kind = ("Google Meet" if ("meet - " in low or re.search(r"\b[a-z]{3}-[a-z]{4}-[a-z]{3}\b", low)) and BROWSERS.search(app)
            else "Slack huddle" if "huddle" in low or "slack" in app.lower()
            else "WhatsApp call" if "whatsapp" in low or "whatsapp" in app.lower()
            else "Zoom" if "zoom" in app.lower() or "zoom meeting" in low
            else "Teams" if "teams" in app.lower() or "microsoft teams" in low
            else f"{app} is using the microphone")
    if event:
        return f"Call detected: {event.title}", f"{kind} · record this meeting?"
    return "Call detected", f"{kind} · record it?"


def ask(title: str, body: str, timeout_s: int = 60) -> str:
    """Show a notification with Record / Ignore; returns "record", "ignore" or "" (dismissed)."""
    if not shutil.which("notify-send"):
        return ""
    try:
        r = subprocess.run(["notify-send", "-a", "minutes", "-i", "audio-input-microphone", "-u", "critical",
                            "-t", str(timeout_s * 1000), "-A", "record=Record", "-A", "ignore=Ignore", title, body],
                           capture_output=True, text=True, timeout=timeout_s + 5)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _minutes(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "minutes", *args], check=False, timeout=60)


def run(cfg: Config, poll: float = 3.0) -> None:
    from . import capture

    opts = cfg.watch
    det = Detector(min_seconds=float(opts.get("min_seconds", 15)), end_grace=float(opts.get("end_grace_seconds", 60)),
                   auto_stop=bool(opts.get("auto_stop", True)), ignore=list(opts.get("ignore", IGNORE)) + ["minutes"])
    print(f"minutes watch: polling every {poll:.0f}s", flush=True)

    def on_ask(app: str, titles: list[str]) -> None:
        try:
            event = calendar.match(cfg, titles=titles)
        except Exception:
            event = None
        title, body = describe(app, titles, event)
        print(f"asking: {title} / {body}", flush=True)
        if ask(title, body) == "record" and not capture.is_recording(cfg):
            args = ["start", "--trigger", "detector"] + (["--event", event.uid] if event else ["--no-match"])
            _minutes(*args)

    while True:
        now = time.time()
        rec = capture.current(cfg)
        titles = calendar.window_titles()
        for action in det.step(now, mic_streams(), titles, rec):
            if action[0] == "ask":
                threading.Thread(target=on_ask, args=action[1:], daemon=True).start()
            elif action[0] == "stop":
                print(f"call ended; stopping {action[1]}", flush=True)
                _minutes("stop")
        time.sleep(poll)
