"""Two-stream chunked capture: mic ("me") and the default sink's monitor ("them").

Each stream is an independent ffmpeg process writing fixed-length FLAC segments, so every
finished chunk is a complete, playable file even if the recorder later crashes.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from . import session as sess
from .config import Config, runtime_dir
from .notify import notify


def pidfile() -> Path:
    return runtime_dir() / "recording.json"


def _pactl(*args: str) -> str:
    out = subprocess.run(["pactl", *args], check=True, capture_output=True, text=True, timeout=5)
    return out.stdout.strip()


def default_devices() -> dict[str, str]:
    """Resolve devices at start time so headset changes between meetings are picked up."""
    return {"me": _pactl("get-default-source"), "them": _pactl("get-default-sink") + ".monitor"}


def ffmpeg_cmd(device: str, out_dir: Path, cfg: Config, client_name: str) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-f", "pulse", "-name", client_name, "-i", device,
        "-ac", "1", "-ar", str(cfg.sample_rate), "-c:a", "flac",
        "-f", "segment", "-segment_time", str(cfg.chunk_seconds), "-reset_timestamps", "1",
        "-segment_format_options", "flush_packets=1",  # partial chunk survives a crash
        str(out_dir / "%05d.flac"),
    ]


def _alive(pid: int) -> bool:
    try:
        # Reap if it happens to be our child (same-process start/stop, e.g. tests).
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Guard against PID reuse: it must still be an ffmpeg.
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip() == "ffmpeg"
    except OSError:
        return False


def _read_pidfile() -> dict[str, Any] | None:
    try:
        return json.loads(pidfile().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fix_chunk_headers(s: sess.Session) -> int:
    """Re-encode chunks so each FLAC header carries its own length.

    ffmpeg's segment muxer leaves STREAMINFO wrong (unknown, or the whole recording's length).
    ffmpeg ignores that, but libsndfile-based readers (soundfile, many ASR servers) trust it and
    fail. Re-encoding a 30 s chunk takes milliseconds; on any failure the original is kept.
    """
    fixed = 0
    for side in sess.SIDES:
        for chunk in s.chunks(side):
            tmp = chunk.with_name(f".{chunk.stem}.fix.flac")
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", str(chunk),
                                "-c:a", "flac", str(tmp)], capture_output=True, timeout=120)
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, chunk)
                fixed += 1
            else:
                tmp.unlink(missing_ok=True)
    return fixed


def _finalize(cfg: Config, s: sess.Session, state: str) -> None:
    fix_chunk_headers(s)
    meta = s.read()
    n = max((len(s.chunks(side)) for side in sess.SIDES), default=0)
    started = meta.get("started_epoch")
    duration = (time.time() - started) if started and state == sess.RECORDED else None
    if duration is None:
        # Upper bound from chunk count; good enough for an interrupted session.
        duration = n * cfg.chunk_seconds
    s.set_state(state, stopped_at=sess.now_iso(), duration_s=round(duration, 1),
                chunks={side: len(s.chunks(side)) for side in sess.SIDES})


def recover(cfg: Config) -> sess.Session | None:
    """If the pidfile points at dead recorders, mark that session interrupted and clear it."""
    rec = _read_pidfile()
    if rec is None:
        return None
    pids = rec.get("pids", {})
    if any(_alive(p) for p in pids.values()):
        return None
    pidfile().unlink(missing_ok=True)
    s = sess.Session(Path(rec["session_path"]))
    if s.meta_path.exists() and s.state == sess.RECORDING:
        _finalize(cfg, s, sess.INTERRUPTED)
        notify("Meeting recording interrupted", f"{s.id}: audio kept ({len(s.chunks('me'))} chunks)",
               urgency="critical")
    return s


def current(cfg: Config) -> dict[str, Any] | None:
    recover(cfg)
    return _read_pidfile()


def is_recording(cfg: Config) -> bool:
    return current(cfg) is not None


def start(cfg: Config) -> sess.Session:
    if is_recording(cfg):
        raise RuntimeError(f"already recording {_read_pidfile()['session_id']}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    devices = default_devices()
    s = sess.create(cfg)
    pids: dict[str, int] = {}
    streams: dict[str, Any] = {}
    procs: list[subprocess.Popen] = []
    try:
        for side in sess.SIDES:
            log = (s.logs_dir / f"ffmpeg-{side}.log").open("ab")
            p = subprocess.Popen(
                ffmpeg_cmd(devices[side], s.audio_dir(side), cfg, f"minutes-{side}"),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            )
            log.close()
            procs.append(p)
            pids[side] = p.pid
            streams[side] = {"device": devices[side], "started_epoch": time.time()}
        time.sleep(0.5)
        dead = [side for side, p in zip(sess.SIDES, procs) if p.poll() is not None]
        if dead:
            logs = "; ".join((s.logs_dir / f"ffmpeg-{d}.log").read_text().strip()[-300:] for d in dead)
            raise RuntimeError(f"ffmpeg for {', '.join(dead)} exited immediately: {logs}")
    except BaseException as e:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        s.set_state(sess.FAILED, error=str(e))
        notify("Meeting recording failed", str(e), urgency="critical")
        raise

    s.set_state(sess.RECORDING, started_at=sess.now_iso(), started_epoch=time.time(),
                chunk_seconds=cfg.chunk_seconds, sample_rate=cfg.sample_rate, streams=streams)
    runtime_dir().mkdir(parents=True, exist_ok=True)
    sess.atomic_write_json(pidfile(), {"session_id": s.id, "session_path": str(s.path),
                                       "pids": pids, "started_at": sess.now_iso()})
    notify("Recording meeting", f"Mic: {devices['me']}\nSystem: {devices['them']}")
    return s


def stop(cfg: Config, timeout: float = 8.0) -> sess.Session | None:
    recover(cfg)
    rec = _read_pidfile()
    if rec is None:
        return None
    pids: dict[str, int] = rec["pids"]
    for pid in pids.values():
        if _alive(pid):
            os.kill(pid, signal.SIGTERM)  # ffmpeg finalizes the current segment on SIGTERM
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(_alive(p) for p in pids.values()):
        time.sleep(0.1)
    for pid in pids.values():
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
    pidfile().unlink(missing_ok=True)
    s = sess.Session(Path(rec["session_path"]))
    _finalize(cfg, s, sess.RECORDED)
    meta = s.read()
    notify("Meeting recording stopped",
           f"{s.id}: {meta.get('duration_s', 0) / 60:.1f} min. Transcribing…")
    return s
