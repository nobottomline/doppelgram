# Contributing

Thanks for helping. Doppelgram acts in a real Telegram account and talks to real people, so a change is judged
first by what it can make the agent do on its own.

## Setup

```
uv sync
uv run ruff check
uv run pytest
```

CI runs the same commands on Linux, macOS and Windows; a pull request needs all of them green. To see the
conversation agent at work without Telegram, use `uv run doppelgram -c config.toml test-chat` (see the README).

## Project layout

| Module | Responsibility |
|---|---|
| `agent.py` | one agent per dating bot: reads the screen, applies the rules, asks the model, presses buttons |
| `brain.py` | the rating prompt, decision validation, privacy sanitizers |
| `conversation.py` | chats with matches and people who write first: pace, plans, meetings, safety checks, the queue |
| `supervisor.py`, `control.py`, `settings.py`, `ui.py` | owner commands, cards and runtime settings |
| `telegram.py`, `transport.py`, `models.py` | the Telethon boundary, its error model and the shared data types |
| `llm.py`, `speech.py`, `media.py` | the OpenAI-compatible client, voice transcription, images |
| `schedule.py`, `storage.py`, `taste.py`, `config.py` | working hours, SQLite stores, the taste profile, configuration |
| `cli.py`, `simulate.py`, `evaluate.py` | the command line, the offline chat simulator and the trap scenarios (`doppelgram eval`) |
| `lookup.py`, `slang.py` | web lookups for what the model may not know; built-in chat slang |

## Rules

* **Tests use fakes only.** The suite drives the agents against simulated bots (`tests/fakes*.py`) and real
  Telethon TL objects. Never add tests or fixtures that need a live account, and never commit chat exports,
  session files, profile texts or photos of real people, or personal names and ids.
* **A behaviour change comes with a regression test** that fails without it.
* **Safety rules stay safe by default.** Anything that can pay, send personal data, contact a person or press
  an unknown button must be off or require the owner by default, and must be covered by tests.
* **Privacy.** Do not store other people's photos, ages, chat texts or transcripts; keep stored profile
  texts to matches and without digits; log button labels, actions and statuses, not texts.
* **Every automatic send passes the checks.** A new path that sends to a person goes through the conversation
  queue and its send-time gate, so takeovers, setting changes, quiet hours and caps apply to it.
* Python 3.11+, type hints, `ruff check` clean, line length 120. Keep comments about *why*, not *what*.
* Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` …).

## AI-assisted contributions

AI tools are welcome, but say so. If any part of a pull request (code, tests, docs, the description itself)
was written or substantially edited by an AI, the PR description must name:

* **the model**, with its exact version (for example `claude-opus-5-5`, `gpt-6-luna`), and
* **the harness**, i.e. the tool that ran it (for example Claude Code, Codex CLI, Cursor, Copilot, a chat UI),
* and briefly **which parts** it produced.

The pull request template has a section for this. You remain the author: you must have read, run and understood
every line you submit. A PR that hides AI use, or that its author cannot explain, will be closed.

## Adding a bot

Most bots need only a `[[bots]]` block in `config.toml` and a task file: the button lists, patterns and limits
are configuration. Add a simulated version of the bot to `tests/fakes_dating.py` and tests for its screens
before relying on it.
