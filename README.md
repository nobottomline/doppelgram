# Doppelgram

[![CI](https://github.com/nobottomline/doppelgram/actions/workflows/ci.yml/badge.svg)](https://github.com/nobottomline/doppelgram/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An LLM agent that lives in your own **Telegram user account** and helps you meet people in Telegram dating
bots. It browses profiles in [Дайвинчик](https://t.me/leomatchbot) and [Бибинто](https://t.me/bibinto_bot)
by **your** taste, tells you about mutual likes, and — if you switch it on — keeps up conversations with
your matches in your voice. You control it from your other Telegram account.

It runs on MTProto (Telethon), not the Bot API, and works with any OpenAI-compatible Chat Completions endpoint
that serves a vision model: OpenRouter, OpenAI, Z.ai, DeepSeek, Groq, vLLM, Ollama, LM Studio and others.
Parameters a provider rejects are adapted automatically. The default is `openai/gpt-6-luna` via OpenRouter, at
about $0.2 per 1000 profiles measured without reference photos (each reference photo adds to every request).
You bring your own API key.

## What it does

| | |
|---|---|
| **Swipes by your taste** | Looks at each profile photo with a vision model and decides like/dislike (Дайвинчик) or a 1–10 rating (Бибинто). |
| **Learns your taste** | From a text description, rules you add from Telegram (`/rule не нравятся полные`), reference photos you send with a `/like` or `/dislike` caption, and your answers. |
| **Learns before it acts** | `ручной` mode sends every profile to you first and you answer with one word; `умный` asks only when unsure; `авто` decides alone and occasionally asks you to rate a profile it already rated, to measure and improve agreement. |
| **Reports matches** | Mutual likes are recorded and sent to you with the link to the person. |
| **Chats with matches (opt-in)** | Drafts messages in your style from facts you wrote about yourself. `approve` mode: nothing is sent without your "ок". `auto` mode: it writes itself, but hands anything sensitive to you. |
| **Stays out of trouble** | Human-like pacing, hourly and daily caps, a daily $ budget, FLOOD_WAIT handling, and a circuit breaker that stops everything when Telegram restricts the account. |

## How it works

```
Telegram updates ──wake──▶ per-bot agent: re-read the bot chat → current card (photo, caption, keyboard)
                              │  hard rules first: age gate, blocked buttons, limits, match notices
                              ▼
            vision LLM (card + your taste text, rules, reference photos, past decisions) → JSON decision
                              │  validated against the real buttons; anything else → "ask the owner"
                              ▼
            supervised: card → you → your answer is clicked and saved as an example
            auto:       human-like delay → re-check the card did not change → press the button
```

* **Updates only wake the loop.** Every step re-reads the chat from the server; a card is identified by its
  content (text, picture, visible buttons), so edits in place, reconnects and duplicate updates are the same case.
* **Reply keyboards are first-class.** Both bots are driven by reply-keyboard buttons; the active keyboard is
  taken from the latest message that carries one (Бибинто sends it before the photo).
* **The model only chooses.** It picks a button id from the list it was shown. Paid features, reports and
  "send a message" traps are blocked per bot (`deny_buttons`), and URL, web-app, payment, phone and
  location buttons are never pressed.
* **Bot and chat content is untrusted.** Nothing in a profile or a message can change the agent's task.

## Controlling it from Telegram

Everything is controlled from your other account's DM with the agent (or the agent account's Saved Messages),
and the same options live in `config.toml`. A value set from Telegram is stored and overrides the file until
`/reset`; `/settings` shows every effective value and where it comes from.

* **Commands start with `/`.** Plain text is never a command, so both chats stay usable for your own notes.
  Russian aliases work too (`/старт`, `/стоп`, `/статус`, `/режим`, ...).
* **Answers to a profile are a Telegram reply** to the agent's message: a button (`❤️`), a number (`7`, a rating
  when the buttons are ratings), optionally with a comment (`7 руки кривые`), `ок` to accept the agent's choice,
  or `пропуск` to let the agent's choice stand without recording an example. Without a reply: `/ok`, `/skip`, `/7`.
* **One bot or all:** put the bot's name last (`/stop бибинто`) or right after `/set` (`/set бибинто режим авто`).
* **`/start` and `/stop` are idempotent:** they report "already running" / "already stopped".

| Command | Effect |
|---|---|
| `/start`, `/stop`, `/pause 2ч` | run, stop, pause (all bots, or one by name) |
| `/status`, `/log`, `/matches` | state, spend and accuracy; recent actions; mutual likes |
| `/mode ручной \| умный \| авто` | how profiles are rated (see below) |
| `/settings`, `/set <key> <value>`, `/reset <key\|all>` | runtime settings: `режим`, `порог`, `ожидание`, `проверки`, `проверок_в_час`, `возраст`, `числа`, `автопродолжение`, `после_лимита`, `часы_работы`, `часы`, `пояс`, `дневная_норма`, `норма`, `бюджет`, `лимит`, `задержка`, `отчёт`, `модель` |
| photo with caption `/like улыбка` or `/dislike` | add a reference photo to your taste |
| `/rule не нравятся полные`, `/taste`, `/refs`, `/delete правило 3`, `/delete фото 5` | manage the taste profile |
| `/chat вкл \| выкл \| подтверждение \| авто`, `/dialogs`, `/lead 123`, `/self 123` | chats with matches |
| `/help` | the full list, formatted |

Messages from the agent use Telegram formatting sparingly (bold, monospace commands), no tables.

### Rating modes

| Mode | Behaviour |
|---|---|
| **ручной** (`supervised`) | every profile comes to you first; your answer is pressed and saved as an example |
| **умный** (`smart`) | the agent rates by itself; profiles below `порог` (confidence) come to you. By default it waits for your answer; `/set ожидание 30` lets it apply its own choice after 30 minutes |
| **авто** (`auto`) | the agent rates everything itself |

In **умный** and **авто**, the agent occasionally sends you a profile it has already rated (`проверки`, default
5%, at most `проверок_в_час` per hour and three unanswered) and asks how *you* would rate it, before revealing its
own choice, so it does not anchor you. High, middle and low decisions are sampled evenly. Your answer never
presses anything; it becomes a labelled example, and `/status` shows how often you agree.

### Paid features

Offers to buy premium, VIP, stars or boosts are declined automatically (`premium_pattern` + `decline_buttons`
per bot): the agent presses an explicit refusal such as "Нет" or "Не сейчас" and keeps rating, without asking
you. "Продолжить" is deliberately not a refusal, since it can mean "agree". An offer without a refusal button,
or more than five offers in an hour, stops that bot and tells you. Paid and report buttons are also blocked
(`deny_buttons`), so the model can never press them.

### Bot limits

Free accounts can only rate so much: Leomatch says "Слишком много ❤️ за сегодня", Bibinto "Коала накушалась
твоих оценок". Such a notice (`limit_pattern`, text messages only) pauses that bot without asking the model, and
`/status` shows "лимит бота" with the time of the next try. With `автопродолжение` on (default) the agent tries
again every `после_лимита` hours (Leomatch 4, Bibinto 1): it presses a `resume_buttons` button such as
"Главное меню" or "🚀 Оценивать" (never a paid one), or sends `/start`. If the bot is still limited it waits
another interval; you get at most one limit message per six hours. `/start` tries again at once.
With `/set автопродолжение выкл` the bot stops on the limit and waits for your `/start`.

### Working hours, daily norm, warnings

* **Working hours** (on by default): the bots act only from 10:00 to 23:00 in `timezone` (Moscow by default)
  and rest at night; `/status` says when they start again. `/set часы 9-22`, `/set пояс +5`, or
  `/set часы_работы выкл` to work around the clock. A window may cross midnight (`22-02`).
* **Daily norm** (off by default): after `норма` rated profiles in a local day the bot waits for the next day
  (and for working hours), and tells you once. `/set дневная_норма вкл`, `/set норма 120`.
* **Warnings** such as "слишком быстро" or "подозрительная активность" never stop the agent: it waits
  `warning_pause_s` (8 s) and goes on. A warning that asks for an identity check still comes to you.

Volume is what matters most for the risk of a block: keep the norm modest and the hours natural.

## Conversations with matches

Off by default. When on, the agent only writes to people a dating bot reported as a mutual like — never to
anyone else. It writes from `about-me.md` (true facts about you, plus notes on what may be embellished and
what not to mention), toward the goal in `conversation.md`, in the style of `style.md` and of every message
you wrote yourself in reply to a draft. Output is post-processed to read like a person typing (lowercase,
no em dashes, no trailing periods).

**Who it writes to.** By default the agent never writes first: it answers only people who wrote to you.
With `write_first = true` (`/chat первым вкл`) it also opens the chat, but only with matches it rated at
least `write_first_min_score` (default 9 of 10, `/chat порог 8`) and at most `first_messages_per_day` a day.
The score is the agent's taste estimate made when it liked the profile (in Бибинто the rating it gave).

**How it talks.** Short messages in your style, reacting to what the person said before asking anything, a
balance of questions and things about you, threads picked up from their profile and earlier messages, a new
topic instead of chasing dry answers. It moves the chat toward the goal in `conversation.md` stage by stage;
when the chat is warm enough, or the person suggests it, the meeting proposal comes to you as a draft, and
you send it with «ок».

**Try it offline first.** `doppelgram -c config.toml test-chat` lets the real agent (your files, prompt and
guards) talk to a model playing a match, without Telegram. Options: `--card` (their profile), `--persona`
(how they behave, e.g. "отвечает сухо"), `--agent-first`, `--turns`.

These rules hold in every mode:

* **Facts only from your profile.** Embellishment only where you explicitly allowed it; nothing invented.
* **"Are you a bot?" is never answered by the agent.** It never claims or denies being an AI; the chat is
  paused and handed to you.
* **Meetings, money, contacts and links, explicit topics, media, and anything unclear** go to you with the
  context and the agent's draft; the chat waits for your answer.
* **Age.** No chats with matches younger than `min_match_age` (default 18), read from their profile card.
* **You can always take over.** If you write to the person yourself, the agent steps back from that chat.
* **Account safety.** Writing first is off by default and capped by `first_messages_per_day` when on (writing
  first to strangers is the main cause of `PEER_FLOOD` restrictions); an hourly send cap, quiet hours and
  human-like delays with a typing indicator.

## What is stored

Profiles stay in Telegram; Doppelgram keeps only what it needs to work, in `data/` on your server.

**Kept:**

* The Telegram session. Like any Telegram client, including the official apps, it caches the ids,
  usernames and names of chats it sees; without that cache it could not find the bots or you after a
  restart.
* Per bot:
  * hashes of the messages it handled;
  * the label and time of every button it pressed;
  * your answers with the model's short visual description of the photo (digits, and the name and city
    from the profile, removed);
  * links to mutual likes;
  * LLM spend.
* Your taste: the rules and the reference photos *you* sent.
* Only with conversations enabled, per match (a conversation needs memory):
  * name and username;
  * the profile text without digits;
  * the agent's notes without digits;
  * drafts waiting for your "ок" or for their send time (deleted once sent or dropped);
  * your style examples.

**Never kept:**

* other people's photos;
* profile captions;
* ages (the age is read from the caption only for the current decision);
* the model's explanations;
* the bots' pop-up answers;
* free text the agent typed to a bot (only commands, numbers and button labels are logged).

Errors from the LLM provider are logged as HTTP status and error code only.

The system log records actions and confidence only.

**Sent to your LLM provider:** the current profile photo and caption (ages per `age_in_prompt`; standalone
numbers 10–18 removed while `hide_numbers_10_18` is on, the default), your taste and past answers. Everything sent goes under your own API key.

The leading block of `#` lines in `taste.md`, `about-me.md`, `style.md` and the task files is a comment the
model never sees; headings and hashtags further down are kept. The `taste`, `about-me` and `style` templates
contain only that comment, so nothing is assumed about you until you write it. The task templates and
`conversation.example.md` hold working instructions for the bots and for chats; they are fine as they are.

## Quick start (macOS, Linux, Windows)

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.11+ itself) and
Telegram app credentials from <https://my.telegram.org> → *API development tools* (`api_id`, `api_hash`).
CI runs the test suite on all three systems.

macOS / Linux:

```sh
git clone https://github.com/nobottomline/doppelgram && cd doppelgram
uv sync
cp config.example.toml config.toml && cp .env.example .env && chmod 600 .env
for f in *.example.md; do cp -n "$f" "${f%.example.md}.md"; done
```

Windows (PowerShell):

```powershell
git clone https://github.com/nobottomline/doppelgram; cd doppelgram
uv sync
Copy-Item config.example.toml config.toml; Copy-Item .env.example .env
Get-ChildItem *.example.md | ForEach-Object { Copy-Item $_ ($_.Name -replace '\.example\.md$', '.md') }
```

Then fill in `.env` (Telegram credentials, LLM key) and `owner.user_ids` in `config.toml`, write your taste in
`taste.md`, and:

```sh
uv run doppelgram -c config.toml login     # once: QR code, Telegram → Settings → Devices → Link Desktop Device
uv run doppelgram -c config.toml run       # keep it running; control it from your other account with /start
```

Files are read as UTF-8 on every system. On Windows `chmod` does not apply: keep `.env`, `*.session` and
`data/` in a folder only your user can read (inside your user profile, not a shared drive).
The agent has to stay online, so for daily use a small always-on Linux server is the practical choice:

## Setup (Ubuntu VPS)

1. Create app credentials at <https://my.telegram.org> → *API development tools* (`api_id`, `api_hash`).
2. On the server, from a checkout of this repository: `sudo ./deploy/install.sh`.
3. Fill in `/opt/doppelgram/.env` (Telegram credentials and your OpenRouter key), then `config.toml`
   (`owner.user_ids` — find your other account's id with `doppelgram resolve @username`), `taste.md`, and,
   for conversations, `about-me.md`, `conversation.md`, `style.md`.
4. Log in once (QR: Telegram → Settings → Devices → Link Desktop Device):
   `sudo -u doppelgram /opt/doppelgram/.venv/bin/doppelgram -c /opt/doppelgram/config.toml login`
5. `systemctl enable --now doppelgram`, then send `/start` from your other account (`/help` lists everything).
6. Logs: `journalctl -u doppelgram -f`. State: `data/*.sqlite3`.

Check the model on local pictures before going live:

```
doppelgram -c config.toml test-llm photo.jpg --bot бибинто --buttons "1,2,3,4,5,6,7,8,9,10"
```

## Development

```
uv sync
uv run ruff check
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md).

Tests run the agents against simulated bots that follow the public protocols of Дайвинчик and Бибинто
(reply keyboards, keyboard-before-photo, edits in place, match notices, limits), a simulated chat partner for
conversations, and real Telethon TL objects of the current layer. No real chats are needed or read.

## Responsible use

Read this before you run it.

* **Telegram.** Accounts using unofficial clients are monitored, and automation can get an account limited
  or frozen. Use your own `api_id`, one stable IP and the default pacing; a secondary account is safer.
  Telegram's terms restrict using platform data for AI "deployment" without the consent of the people
  involved — this tool processes other users' profiles and messages, so you are responsible for how you use it.
* **The dating bots** do not allow automation explicitly; they can ban or shadow-ban accounts for mass likes
  or spam. Keep the limits low.
* **Other people's data.** Profile photos and messages of other users are sent to the LLM provider *you*
  configure, under *your* key. Doppelgram does not store their photos (only your own reference photos, locally);
  prefer providers with no-training / zero-retention options.
* **Honesty.** People you chat with believe they are talking to you. The agent never denies being an AI when
  asked and never invents facts about you; meetings and anything serious are yours to handle. Do not use this
  to deceive, harass or mass-message people.
