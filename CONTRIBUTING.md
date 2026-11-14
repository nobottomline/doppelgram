# Contributing

Thanks for helping. Doppelgram acts in a real Telegram account, so changes are judged first by what they can
make the agent do on its own.

## Setup

```
uv sync
uv run ruff check
uv run pytest
```

CI runs the same commands on Linux, macOS and Windows; a pull request needs all of them green.

## Rules

* **Tests use fakes only.** The suite drives the agents against simulated bots (`tests/fakes*.py`) and real
  Telethon TL objects. Never add tests or fixtures that need a live account, and never commit chat exports,
  session files, profile texts or photos of real people, or personal names and ids.
* **A behaviour change comes with a regression test** that fails without it.
* **Safety rules stay safe by default.** Anything that can pay, send personal data, contact a person or press
  an unknown button must be off or require the owner by default, and must be covered by tests.
* **Privacy.** Do not store profiles, ages or other people's photos; log button labels and actions, not texts.
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
