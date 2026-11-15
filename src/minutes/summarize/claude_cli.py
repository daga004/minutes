"""Summarize with the Claude Code CLI in headless print mode, with all tools disabled."""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from ..config import Config
from .base import PROMPT, register


class ClaudeCLISummarizer:
    name = "claude-cli"

    def __init__(self, cfg: Config, timeout: float = 900.0):
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("claude CLI not found on PATH")
        self.exe = exe
        self.model = cfg.summary_model
        self.timeout = timeout

    def command(self) -> list[str]:
        cmd = [self.exe, "-p", "--tools", "", "--no-session-persistence", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def summarize(self, transcript_md: str) -> str:
        return self.complete(PROMPT.format(transcript=transcript_md))

    def complete(self, prompt: str) -> str:
        """One prompt in, the model's text out (no tools, no session saved)."""
        # Run from an empty dir so no project CLAUDE.md or settings leak into the prompt.
        with tempfile.TemporaryDirectory(prefix="minutes-claude-") as cwd:
            r = subprocess.run(self.command(), input=prompt,
                               capture_output=True, text=True, timeout=self.timeout, cwd=cwd)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(f"claude -p failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[:500]}")
        return r.stdout.strip() + "\n"


@register("claude-cli")
def _factory(cfg: Config) -> ClaudeCLISummarizer:
    return ClaudeCLISummarizer(cfg)
