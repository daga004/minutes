"""Hindi-English ASR: Trelis/tara (Whisper fine-tuned on code-mixed speech).

Benchmarked on CoSHE-Eval (bench/): 20.0% WER on CUDA, 20.1% on the Iris Xe via OpenVINO, vs
44.6% for Superwhisper (which drops or translates Hindi).

Where it runs is decided when the worker starts (once per run):
  1. a remote NVIDIA box over ssh (`remote`, off by default) if its GPU has more than
     `remote_min_free_mb` (5000) MiB free, i.e. nobody else is using it;
  2. otherwise the local Intel GPU through OpenVINO (engine venv from `minutes setup tara`).
If the remote worker dies mid-run, the chunk is retried locally and the run stays local.

A worker keeps the model loaded and gets chunks one at a time as JSON lines; audio is sent as
16 kHz int16 PCM (decoded here with ffmpeg), so the remote box needs no ffmpeg or shared disk.

Config (~/.config/minutes/config.toml, all optional):

    [asr.tara]
    device = "GPU"                  # local OpenVINO device: GPU (Intel iGPU) or CPU
    remote = "gpu-box"              # ssh host (alias) with an NVIDIA GPU; unset = local only
    remote_python = "~/tara-venv/bin/python"   # venv there with torch (CUDA) + transformers
    remote_min_free_mb = 5000
    python = "..."                  # local engine interpreter
    model_dir = "..."               # local OpenVINO IR
    timeout = 300                   # seconds per chunk before a worker is considered stuck
"""

from __future__ import annotations

import atexit
import base64
import fcntl
import json
import os
import select
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..config import Config, runtime_dir
from .base import Segment, register

WORKER = Path(__file__).with_name("tara_worker.py")
REQUIREMENTS = Path(__file__).with_name("tara-requirements.txt")
TORCH = ["torch==2.15.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"]  # no CUDA/XPU libs
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=10",
       "-o", "ServerAliveCountMax=3"]
REMOTE_WORKER = ".cache/minutes/tara_worker.py"  # relative to the remote home


def paths(cfg: Config) -> dict[str, Path]:
    opts = cfg.asr_options.get("tara", {})
    return {
        "engine": cfg.storage_dir / "engines" / "tara",
        "python": Path(opts.get("python", cfg.storage_dir / "engines" / "tara" / "bin" / "python")).expanduser(),
        "model_dir": Path(opts.get("model_dir", cfg.storage_dir / "models" / "tara-ov-fp16")).expanduser(),
        "cache_dir": cfg.storage_dir / "cache" / "openvino",
        "log": cfg.storage_dir / "logs" / "tara.log",
    }


def pcm16(path: Path) -> tuple[bytes, float]:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True, timeout=120,
    ).stdout
    return raw, len(raw) / 32000


def remote_free_mb(host: str, timeout: float = 8.0) -> int | None:
    """Free memory on the remote's emptiest NVIDIA GPU, or None if unreachable."""
    try:
        r = subprocess.run([*SSH, host, "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()] if r.returncode == 0 else []
    return max(vals) if vals else None


class TaraTranscriber:
    name = "tara"
    max_parallel = 1  # one model per worker; chunks are queued

    def __init__(self, cfg: Config):
        opts = cfg.asr_options.get("tara", {})
        self.p = paths(cfg)
        self.device = str(opts.get("device", "GPU"))
        self.remote = str(opts.get("remote", ""))
        self.remote_python = str(opts.get("remote_python", "~/tara-venv/bin/python"))
        self.remote_min_free_mb = int(opts.get("remote_min_free_mb", 5000))
        self.timeout = float(opts.get("timeout", 300))
        self.load_timeout = float(opts.get("load_timeout", 900))  # first GPU compile is slow
        self.local_ok = (self.p["python"].exists()
                         and (self.p["model_dir"] / "openvino_encoder_model.xml").exists())
        if not self.local_ok and not self.remote:
            raise RuntimeError(f"tara engine not installed ({self.p['python']}); run `minutes setup tara`")
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._buf = b""
        self._gpu_lock = None
        self._remote_failed = False
        self.where: str | None = None  # "remote" | "local"
        self.engine_label: str | None = None  # e.g. "gpu-box: NVIDIA GeForce RTX 5060 Ti"
        atexit.register(self.close)

    # -- choosing and starting a worker ------------------------------------------------------
    def choose(self) -> str:
        if self.remote and not self._remote_failed:
            free = remote_free_mb(self.remote)
            if free is not None and free > self.remote_min_free_mb:
                self._log(f"{self.remote}: {free} MiB GPU free > {self.remote_min_free_mb}; using it")
                return "remote"
            self._log(f"{self.remote}: " + ("unreachable" if free is None else
                      f"only {free} MiB GPU free (busy)") + "; using the local GPU")
        if not self.local_ok:
            raise RuntimeError(f"{self.remote} unavailable and local tara engine not installed; "
                               "run `minutes setup tara`")
        return "local"

    def _log(self, msg: str) -> None:
        print(f"tara: {msg}", file=sys.stderr, flush=True)

    def _start(self, where: str | None = None) -> None:
        where = where or self.choose()
        self.where = where  # set before starting, so a failed remote start still falls back
        self.p["log"].parent.mkdir(parents=True, exist_ok=True)
        if where == "remote":
            subprocess.run([*SSH, self.remote, f"mkdir -p ~/.cache/minutes && cat > ~/{REMOTE_WORKER}"],
                           input=WORKER.read_bytes(), check=True, timeout=30, capture_output=True)
            cmd = [*SSH, self.remote,
                   f"HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false exec {self.remote_python} -I "
                   f"~/{REMOTE_WORKER} --engine torch --device cuda"]
            env = None
        else:
            # Only one local worker machine-wide: two loaded copies would not fit beside a meeting.
            runtime_dir().mkdir(parents=True, exist_ok=True)
            self._gpu_lock = open(runtime_dir() / "tara.lock", "w")
            fcntl.flock(self._gpu_lock, fcntl.LOCK_EX)
            self.p["cache_dir"].mkdir(parents=True, exist_ok=True)
            cmd = [str(self.p["python"]), "-I", str(WORKER), "--model-dir", str(self.p["model_dir"]),
                   "--device", self.device, "--cache-dir", str(self.p["cache_dir"])]
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        with open(self.p["log"], "ab") as log:
            log.write(f"\n== {time.strftime('%F %T')} start {where}: {shlex.join(cmd)}\n".encode())
            log.flush()
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env)
        self._buf = b""
        ready = self._read(self.load_timeout)
        if not ready.get("ready"):
            raise RuntimeError(f"tara worker failed to start: {ready}")
        dev = ready.get("device")
        self.engine_label = f"{self.remote}: {dev}" if where == "remote" else f"local: {dev}"
        self._log(f"ready on {self.engine_label}")

    def _read(self, timeout: float) -> dict:
        assert self._proc and self._proc.stdout
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                self.close()
                raise RuntimeError(f"tara worker timed out after {timeout:.0f}s (see {self.p['log']})")
            r, _, _ = select.select([fd], [], [], left)
            if r:
                data = os.read(fd, 65536)
                if not data:
                    code = self._proc.wait()
                    self._proc = None
                    self._release()
                    raise RuntimeError(f"tara worker exited ({code}); see {self.p['log']}")
                self._buf += data
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def _release(self) -> None:
        if self._gpu_lock:
            self._gpu_lock.close()  # releases the flock
            self._gpu_lock = None

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
                proc.wait()
        self._release()

    # -- Transcriber ------------------------------------------------------------------------
    def _request(self, req: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(req) + "\n").encode())
        self._proc.stdin.flush()
        return self._read(self.timeout)

    def transcribe(self, path: Path, lang_hint: str | None) -> list[Segment]:
        raw, _ = pcm16(Path(path))
        req = {"pcm_b64": base64.b64encode(raw).decode(), "lang": lang_hint}
        with self._lock:
            try:
                resp = self._request(req)
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                if self.where != "remote" or not self.local_ok:
                    raise
                self._log(f"remote worker failed ({e}); switching to the local GPU")
                self.close()
                self._remote_failed = True
                self._start("local")
                resp = self._request(req)
        if "error" in resp:
            raise RuntimeError(f"tara: {resp['error']}")
        text = resp.get("text", "").strip()
        return [Segment(0.0, float(resp.get("duration", 0.0)), text)] if text else []


@register("tara")
def _factory(cfg: Config) -> TaraTranscriber:
    return TaraTranscriber(cfg)
