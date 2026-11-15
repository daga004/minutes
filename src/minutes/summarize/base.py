"""Summarizer plug-in interface: `summarize(transcript_md) -> markdown`."""

from __future__ import annotations

from typing import Callable, Protocol

from ..config import Config


class Summarizer(Protocol):
    name: str

    def summarize(self, transcript_md: str) -> str: ...


Factory = Callable[[Config], Summarizer]
_REGISTRY: dict[str, Factory] = {}


def register(name: str):
    def deco(factory: Factory) -> Factory:
        _REGISTRY[name] = factory
        return factory

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, cfg: Config) -> Summarizer:
    try:
        return _REGISTRY[name](cfg)
    except KeyError:
        raise LookupError(f"unknown summary backend {name!r}; available: {', '.join(available())}") from None


PROMPT = """\
You are writing meeting minutes from an automatic transcript.

About the transcript:
- "Me" is the person who recorded the meeting (their microphone). "Them" is everyone else on the
  call, mixed into one channel, so it may be several different people.
- It may be in English, Hindi (Devanagari), or Hinglish (mixed). Speech recognition errors are
  likely, especially for names, numbers and technical terms. Infer the intended meaning.
- If the recorder used speakers instead of headphones, some "Them" speech can be repeated under
  "Me". Treat near-duplicate lines as one utterance by "Them".

If a <meeting> block is given, it comes from the calendar invite. Use it to spell names and
terms correctly, to name action-item owners, and to order the key points by agenda item where the
discussion follows the agenda. Invitees may not have attended; only say someone said something
if the transcript shows it. Agenda items that were not discussed go under Open questions.

Write the minutes in English, as GitHub-flavoured Markdown.

The very first line must be the meeting title: "# " followed by 3-8 English words naming what the
meeting was about, specific enough to find it again months later (e.g. "# Drone survey schedule
with vendor", not "# Team meeting").

The second line must start with "Key facts: " followed by one sentence (at most 30 words) with the
facts someone scanning a list of meetings must not miss: decisions, deadlines, amounts, numbers,
commitments. Write "Key facts: none" if there are none. Then exactly these sections:

## Summary
3-6 sentences: purpose of the meeting and what was concluded.

## Key points
Bullets of the substantive discussion, grouped by topic.

## Decisions
Bullets. Write "None recorded." if there were none.

## Action items
A checklist: "- [ ] <owner (Me / Them / a name if stated)>: <task> (due <date> if stated)".
Write "None recorded." if there were none.

## Open questions
Bullets of unresolved questions or follow-ups. Write "None recorded." if there were none.

Do not invent facts that are not in the transcript. Output only the Markdown, no preamble.

{transcript}
"""
