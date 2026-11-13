"""Desktop notifications via notify-send. Never raises."""

from __future__ import annotations

import shutil
import subprocess


def notify(title: str, body: str = "", *, urgency: str = "normal", icon: str = "audio-input-microphone") -> None:
    exe = shutil.which("notify-send")
    if not exe:
        return
    try:
        subprocess.run(
            [exe, "-a", "minutes", "-u", urgency, "-i", icon, title, body],
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        pass
