"""ASR backends. Importing this package registers the built-in ones."""

from . import null, superwhisper, tara  # noqa: F401  (registration side effects)
from .base import Segment, Transcriber, available, get, register

__all__ = ["Segment", "Transcriber", "available", "get", "register"]
