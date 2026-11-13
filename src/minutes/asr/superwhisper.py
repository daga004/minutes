"""Placeholder backend wrapping the local `superwhisper` CLI (cloud S1-Voice, Hindi-capable).

`superwhisper transcribe FILE` is asynchronous: it returns {"ok": true, "session_id": N} and
the text appears later in `superwhisper status` under "last" once last.session_id == N.
Because that result slot is global, only one chunk may be in flight at a time (enforced
with a lock file, so concurrent `minutes` processes also queue up).
Side effect: each chunk shows up in Superwhisper's own history.
"""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

from ..config import Config, runtime_dir
from .base import Segment, register


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    ).stderr
    # last "time=HH:MM:SS.xx" is the decoded length (FLAC segment headers can lie)
    stamps = [t for t in out.split("time=")[1:]]
    if not stamps:
        return 0.0
    h, m, s = stamps[-1].split()[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


class SuperwhisperTranscriber:
    name = "superwhisper"
    max_parallel = 1

    def __init__(self, cfg: Config, timeout: float = 180.0):
        exe = shutil.which("superwhisper")
        if not exe:
            raise RuntimeError("superwhisper CLI not found")
        self.exe = exe
        self.timeout = timeout

    def _json(self, *args: str) -> dict:
        r = subprocess.run([self.exe, *args], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"superwhisper {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
        return json.loads(r.stdout)

    def transcribe(self, path: Path, lang_hint: str | None) -> list[Segment]:
        runtime_dir().mkdir(parents=True, exist_ok=True)
        with open(runtime_dir() / "superwhisper.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            started = self._json("transcribe", str(path))
            if not started.get("ok"):
                raise RuntimeError(f"superwhisper refused {path.name}: {started}")
            job = started["session_id"]
            deadline = time.monotonic() + self.timeout
            while True:
                status = self._json("status")
                last = status.get("last") or {}
                if last.get("session_id") == job and status.get("phase") == "idle":
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"superwhisper timed out on {path.name} (job {job})")
                time.sleep(0.5)
        err = last.get("error") or (last.get("warning") if not last.get("text") else None)
        if err and "no speech" in str(err).lower():
            return []  # a quiet or very short chunk; not a failure
        if err:
            raise RuntimeError(f"superwhisper error on {path.name}: {err}")
        text = (last.get("text") or "").strip()
        if not text:
            return []
        return [Segment(0.0, round(_duration(path), 2), text)]


@register("superwhisper")
def _factory(cfg: Config) -> SuperwhisperTranscriber:
    return SuperwhisperTranscriber(cfg)
