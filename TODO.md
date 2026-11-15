# TODO

Phase 1 (recording, Tara ASR, live transcription, titles, bar dot) is done and pushed.

## Phase 2 — calendar, sharing, detection

Decisions (2026-11-16): one **Join** button with a global "record when I join" toggle
(right-click = join only); calls not started from the bar get a **Record / Ignore** popup;
notes go to a `meetings/` folder in the personal Outline collection.

### 2a. Calendar link + Outline note
- [x] `calendar.py`: read-only OmaCal adapter (events, agenda, attendees, organizer, call link, series)
- [x] Match the recording to an event at start; snapshot `meeting.json`; `minutes link <id> [--event]`
- [x] Summary prompt gets title, agenda, attendee names; 1:1 calls name "Them"
- [x] Calendar title names the session folder (Claude's title only for unscheduled calls)
- [x] Cleaned full transcript (Claude fixes ASR errors and echo duplicates, keeps every line) -> `transcript.clean.md`; raw `transcript.md` kept
- [x] Outline note `<vault>/<collection>/meetings/<date>-<slug>.md`: summary, then cleaned transcript
- [x] Wait for the sync daemon's `outline_url`, copy it to the clipboard, show it in the notification
- [x] Tests (synthetic calendar DB, no personal data), README, commit + push

### 2b. Bar widget
- [x] Next meeting + **Join** button (opens the call link, records if the toggle is on; right-click = join only)
- [x] Global toggle "record when I join"
- [x] Popup stats: total time in meetings (today / this week / this month / all time)
- [x] Record in ~/dotfiles (change.sh), commit + push
- [ ] Live check during a real meeting (Join -> recording dot -> stop -> Outline link)

### 2c. Call detector
- [ ] Watch PipeWire mic users (ignore superwhisper / pw-record and minutes itself) + window titles
- [ ] Popup "Call detected: <title> · Record / Ignore", once per call; calendar match for Meet codes
- [ ] Auto-stop when the app releases the mic for 60 s
- [ ] systemd user service, tests, commit + push

### Later
- [ ] Analytics: time by meeting / person / series / project
- [ ] Follow-ups from the previous meeting in a series
- [ ] Attendee names as an ASR hint (only if the benchmark WER holds)
- [ ] Speaker separation for group calls (remote GPU box)
- [ ] Choose the Outline collection per meeting; MCP search
- [ ] Cleanups: old test sessions, `bench/ov-models/maya-fp16` (8.3 GB), `mise prune claude`
