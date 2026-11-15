"""Clean stage: transcript.md -> transcript.clean.md, the full transcript tidied by Claude.

Nothing is summarized or dropped: Claude fixes recognition errors (using the calendar's names and
agenda), removes fillers and the echo of "Them" picked up by the mic, and keeps the timestamps.
The raw transcript.md is never modified. Long meetings are cleaned in parts of ~CHUNK_CHARS.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from . import calendar
from . import session as sess
from .config import Config

CHUNK_CHARS = 6000
LINE = re.compile(r"^\*\*\[\d[\d:]*\] [^*]+:\*\*")

PROMPT = """\
You are cleaning up part {part} of {parts} of an automatic meeting transcript. It is not a summary:
the people reading it want everything that was said.

About the transcript:
- "Me" is the person who recorded the meeting (their microphone). "Them" is everyone else on the
  call, mixed into one channel.
- Speech recognition errors are likely, especially for names, numbers and technical terms.
- With speakers instead of headphones, "Them" speech is sometimes repeated under "Me".
{meeting}
Rewrite the lines below as a clean transcript:
- Keep every utterance and all of its content, in order. Do not summarize, shorten, merge across
  speakers, or add anything that was not said.
- Fix recognition errors where the intended word is clear from context; spell names and terms as
  in the meeting details. Mark words you cannot recover as [inaudible].
- Drop fillers, stutters and false starts ("um", "अः", repeated words).
- Drop a "Me" line only when it is clearly an echo of the "Them" line next to it.
- {language}
- Keep each line's format and timestamp exactly: **[mm:ss] Speaker:** text{rename}

Output only the cleaned lines, nothing else.

<lines>
{lines}
</lines>
"""

LANGUAGE = {
    "as-spoken": "Keep the language as spoken: Hindi in Devanagari, English in Latin script, Hinglish "
                 "as mixed. Do not translate.",
    "english": "Translate everything into natural English, keeping names and technical terms.",
}


def split(transcript: str) -> tuple[str, list[str]]:
    """(header, utterance blocks): the header is everything before the first speaker line."""
    lines = transcript.splitlines()
    first = next((i for i, l in enumerate(lines) if LINE.match(l)), len(lines))
    header = "\n".join(lines[:first]).rstrip()
    blocks = [l for l in lines[first:] if l.strip()]
    return header, blocks


def parts(blocks: list[str], limit: int = CHUNK_CHARS) -> list[list[str]]:
    out: list[list[str]] = [[]]
    size = 0
    for b in blocks:
        if out[-1] and size + len(b) > limit:
            out.append([])
            size = 0
        out[-1].append(b)
        size += len(b)
    return [p for p in out if p]


def prompts(cfg: Config, event: calendar.Event | None, chunks: list[list[str]]) -> list[str]:
    meeting = f"\nMeeting details from the calendar invite:\n{calendar.describe(event)}\n" if event else ""
    rename = ""
    if event and len(event.others) == 1:
        rename = f'\n- Write "{event.others[0].label}" instead of "Them" (this is a 1:1).'
    language = LANGUAGE.get(cfg.clean_language, LANGUAGE["as-spoken"])
    return [PROMPT.format(part=i + 1, parts=len(chunks), meeting=meeting, language=language, rename=rename,
                          lines="\n\n".join(c)) for i, c in enumerate(chunks)]


def tidy(text: str) -> str:
    """Model output -> one speaker line per paragraph; anything else (stray prose) is dropped."""
    text = re.sub(r"^```\w*\n|\n```$", "", text.strip())
    keep = [l.strip() for l in text.splitlines() if LINE.match(l.strip())]
    return "\n\n".join(keep)


def run(cfg: Config, s: sess.Session, completer=None) -> sess.Session:
    """Write transcript.clean.md. `completer` is anything with .complete(prompt) -> str."""
    raw = s.transcript_md.read_text(encoding="utf-8")
    header, blocks = split(raw)
    if not blocks:
        return s
    if completer is None:
        from .summarize import claude_cli

        completer = claude_cli.ClaudeCLISummarizer(cfg)
    chunks = parts(blocks)
    s.set_stage("clean", "running", parts=len(chunks))
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            cleaned = list(pool.map(completer.complete, prompts(cfg, calendar.load(s), chunks)))
    except Exception as e:
        s.set_stage("clean", "failed", error=str(e)[:500])
        raise
    body = [tidy(c) for c in cleaned]
    for i, b in enumerate(body):  # a part that came back empty is kept raw rather than lost
        if not b:
            body[i] = "\n\n".join(chunks[i])
    header = header.replace("# Meeting transcript:", "# Clean transcript:", 1)
    note = "_Cleaned by Claude from the raw transcript (transcript.md); nothing summarized._"
    sess.atomic_write_text(s.clean_md, f"{header}\n\n{note}\n\n" + "\n\n".join(body) + "\n")
    s.set_stage("clean", "done", parts=len(chunks))
    return s
