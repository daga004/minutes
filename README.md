# minutes

Composable meeting recorder for Omarchy / Linux. Records **both sides** of any call (your mic +
system audio), keeps every recording, transcribes with a swappable speech-to-text backend, and
writes English minutes with `claude -p`.

**Hotkey:** `SUPER + ALT + M` starts recording; press again to stop. Processing then runs in the
background and a notification points at `summary.md`.

## Commands

```
minutes toggle                 # what the hotkey runs
minutes start | stop [--no-process] | status
minutes list                   # sessions, state, duration
minutes process [ID]           # transcribe -> merge -> summarize (resumable)
minutes transcribe [ID] [-b BACKEND] [-f]   # -f redoes chunks already done
minutes merge [ID] [-b BACKEND]             # rebuild transcript.md from a backend's output
minutes summarize [ID]
minutes clean [ID]             # full transcript tidied by Claude -> transcript.clean.md
minutes publish [ID]           # write the Outline note, print its link
minutes link [ID] [--event UID | --none]    # show / set the calendar meeting of a session
minutes events [--hours N] [--json]         # upcoming calendar meetings
minutes start --event UID      # record and link to that meeting (otherwise matched automatically)
minutes stats [--json]         # time in meetings: today, this week, this month, all time
minutes join UID [--record | --no-record]   # open the call link; records if record_on_join is on
minutes set record_on_join on|off
minutes bar                    # JSON state for a status-bar widget
minutes open [ID] [--print]
minutes backends
```

`ID` is a session id or unique prefix; default is the latest session.
Background logs: `journalctl --user -u minutes-live-<id>` (transcribes while recording, then
finishes the summary) or `minutes-process-<id>` (when live transcription was off or died).

## Storage

```
~/.local/share/minutes/sessions/<YYYY-MM-DD_HHMM>_<title-slug>/
  session.json                       # state + per-stage status + title
  audio/{me,them}/00000.flac ...     # 30 s, 16 kHz mono FLAC chunks (never deleted)
  transcripts/<backend>/{me,them}/00000.json
  meeting.json                       # the calendar event (frozen copy), if any
  transcript.md                      # raw: interleaved Me / Them from the ASR
  transcript.clean.md                # the same, every line kept, tidied by Claude
  summary.md
```

The folder is named by start time until the summary is written; the summarizer then titles the
meeting and the folder gains the title's slug (e.g. `2026-11-13_1147_drone-survey-schedule`).
Commands accept either the full name or just the time part (`minutes open 2026-11-13_1147`).

State is plain files, not a database: each stage reads the previous stage's files, so any stage
can be re-run with any backend, and backends' outputs sit side by side for comparison.

## Config (optional) `~/.config/minutes/config.toml`

```toml
[asr]
backend = "tara"           # or "superwhisper", "null"
lang_hint = "hi"
parallel = 3               # capped by each backend's max_parallel

[summary]
backend = "claude-cli"
# model = "sonnet"

[capture]
chunk_seconds = 30

[transcribe]
silence_peak_db = -45      # sound quieter than this counts as silence
min_voiced_seconds = 1.0   # chunks with less sound than this are skipped (blips, pings)

[summary]
clean_transcript = true    # tidy the full transcript after the summary
clean_language = "as-spoken"   # or "english" to translate it

[calendar]
backend = "omacal"         # or "none"
# db = "~/.local/share/com.omacal.app/omacal.db"
match_minutes = 10

[outline]
# vault = "~/Documents/notes"   # unset = no Outline note
# folder = "Me/meetings"        # a folder your Outline sync daemon maps to a collection
wait_seconds = 120

[storage]
# dir = "~/.local/share/minutes"
```

## Calendar

With [OmaCal](https://omacal.app) installed, a recording is matched to the meeting happening
when it starts. The best match is the event whose Google Meet code is in an open window's
title; then events with a call link, other invitees and your acceptance; then the closest
start time. The event (title, agenda, attendees, organizer, series) is saved as
`meeting.json`, so later calendar edits don't change it. Its database is only read, never
written, and recurring meetings are expanded from their rules as OmaCal does.

The calendar gives the summary the agenda and the attendees' names, so names are spelled
right, action items get owners, and in a 1:1 "Them" becomes the other person. The calendar
title names the session folder and the note; Claude's own title is kept as the topic.
Wrong match? `minutes link ID` lists nearby events, `--event UID` relinks, then re-run
`minutes summarize ID && minutes publish ID`.

## Bar widget

`minutes bar` prints everything a status bar needs as one JSON object: the recording (if
any), today's meetings with a `joinable` flag (from 5 minutes before the start until the
end, if there is a call link), time-in-meetings stats, the record-on-join setting and the
last note's link. An Omarchy bar widget built on it shows:

- **Recording:** red dot and elapsed time. Click to stop; right-click for the popup.
- **A meeting to join:** "Join". Click opens the call and records it (when "record when I
  join" is on); right-click joins without recording.
- **Otherwise:** a dim mic icon that opens the popup.

The popup shows time in meetings, the upcoming meetings with Join buttons, the "record when
I join" switch, Record now / Stop, and a link to the last meeting's notes. Recordings under
a minute are not counted as meetings.

## Outline notes

After the summary, `minutes` writes one Markdown note into `[outline] vault/folder`: the
summary on top, then `## Transcript` with the full cleaned transcript. An Outline sync
daemon (for example outline-sync-daemon) pushes it and writes `outline_url` into the
frontmatter. `minutes` waits for that link, copies it to the clipboard and shows it in the
"Meeting notes ready" notification. Publishing again updates the same Outline document.
Everyone who can read that collection can read the transcript.

## Tara (default speech-to-text)

`tara` runs [Trelis/tara](https://huggingface.co/Trelis/tara), a Whisper model fine-tuned for
Hindi-English code-mixing, locally on the Intel GPU through OpenVINO. Nothing leaves the machine.
On 30 CoSHE-Eval clips (see `bench/`) it gets 20.1% word errors vs 44.6% for Superwhisper, which
drops or translates Hindi. It runs at about 0.5× real time per stream on the Iris Xe.

One-time install (separate Python 3.13 venv and an approximately 3 GB model under `~/.local/share/minutes/`):

```sh
minutes setup tara            # or: minutes setup tara --from bench/ov-models/tara-fp16
```

Options live under `[asr.tara]`: `device = "GPU" | "CPU"`, `python`, `model_dir`, `timeout`.

**Offloading to an NVIDIA machine.** Set `remote = "<ssh host>"` (key-based ssh, e.g. over
Tailscale) and `remote_python` (a venv there with CUDA torch + transformers and the `Trelis/tara`
weights in its Hugging Face cache). At the start of each run, if that GPU has more than
`remote_min_free_mb` (default 5000) MiB free, chunks are sent there (about 0.07× real time);
otherwise, or if it drops mid-meeting, the local Intel GPU is used.
The worker loads the model once per run, falls back to the CPU if the GPU fails, and logs to
`~/.local/share/minutes/logs/tara.log`.

## Adding a speech-to-text backend

Create `src/minutes/asr/<name>.py` with a class exposing `name`, `max_parallel` and
`transcribe(path, lang_hint) -> list[Segment]` (times relative to the chunk), register it with
`@register("<name>")`, and import it in `asr/__init__.py`. Then
`minutes transcribe <id> -b <name>` on any old session.

## Known v1 limitations

- **Speakers instead of headphones:** the mic picks up the other side. Quiet leaked audio can
  make the speech-to-text model invent stock phrases (e.g. "ご視聴ありがとうございました") in the
  "Me" track. The summary prompt is told to ignore duplicates and noise, but use headphones for
  clean transcripts.
- Switching the output device mid-meeting: `them` keeps recording the old device.
- Fixed 30 s chunks can split a word at the boundary.
- The Superwhisper placeholder is serial (one chunk at a time) and adds every chunk to
  Superwhisper's own history.
- A hard crash loses at most the last ~second of audio; the session is marked `interrupted`
  the next time `minutes` runs, and can still be processed.
