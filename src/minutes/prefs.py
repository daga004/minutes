"""Settings the bar widget toggles, kept apart from the hand-edited config.toml.

<storage>/prefs.json, e.g. {"record_on_join": true}.
"""

from __future__ import annotations

import json

from . import session as sess
from .config import Config

DEFAULTS = {"record_on_join": True}


def load(cfg: Config) -> dict:
    try:
        data = json.loads((cfg.storage_dir / "prefs.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def set_(cfg: Config, key: str, value) -> dict:
    if key not in DEFAULTS:
        raise LookupError(f"unknown setting {key!r}; known: {', '.join(DEFAULTS)}")
    data = load(cfg)
    data[key] = type(DEFAULTS[key])(value)
    sess.atomic_write_json(cfg.storage_dir / "prefs.json", data)
    return data
