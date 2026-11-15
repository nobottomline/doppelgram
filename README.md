# Doppelgram

[![CI](https://github.com/nobottomline/doppelgram/actions/workflows/ci.yml/badge.svg)](https://github.com/nobottomline/doppelgram/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Doppelgram is an LLM agent that lives in your own **Telegram user account** and helps you meet people
through Telegram dating bots. It rates profiles in [Дайвинчик](https://t.me/leomatchbot) and
[Бибинто](https://t.me/bibinto_bot) by **your** taste, reports mutual likes, and, if you switch it on,
carries the conversations with your matches in your voice. You control it from your other Telegram account.

It speaks MTProto through [Telethon](https://github.com/LonamiWebs/Telethon), not the Bot API, and works with
any OpenAI-compatible Chat Completions endpoint that serves a vision model: OpenRouter, OpenAI, Z.ai,
DeepSeek, Groq, vLLM, Ollama, LM Studio and others. You bring your own API key.

> **Read [Responsible use](#responsible-use) before running it.** It acts in a real account, talks to real
> people on your behalf and sends their profiles to the model provider you choose.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Running on a server](#running-on-a-server)
- [Files you edit](#files-you-edit)
- [Controlling it from Telegram](#controlling-it-from-telegram)
- [Rating profiles](#rating-profiles)
- [Conversations with matches](#conversations-with-matches)
- [Testing without Telegram](#testing-without-telegram)
- [Privacy](#privacy)
- [How it works](#how-it-works)
- [Development](#development)
- [Responsible use](#responsible-use)

## Features

- **Rates profiles by your taste.** A vision model looks at each profile and likes or skips it (Дайвинчик)
  or gives a 1–10 rating (Бибинто), following your description, rules, reference photos and past answers.
- **Learns from you.** Three modes: every profile to you first, only doubtful ones, or fully on its own with
  occasional taste checks that measure how often you agree.
- **Gets through the bots' screens.** Declines paid offers, skips ads, channel promos and location requests,
  waits out daily limits and retries on a schedule, and hands real problems (bans, verifications, captchas)
  to you.
- **Works like a person.** Working hours, an optional daily norm, human-like pauses, and calm handling of the
  bots' "too fast" warnings.
- **Talks to matches (opt-in).** Short messages in your style, a plan for every person, a natural pace,
  read receipts only when it answers, photos, stickers and voice messages understood, meetings arranged only
  with your decision.
- **Stays under your control.** Every action can be paused, overridden or taken over from Telegram, and
  sensitive moments always come to you.
- **Keeps little.** Other people's photos, ages, chat texts and voice transcripts are never stored.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.11+
itself), Telegram app credentials from <https://my.telegram.org> → *API development tools* (`api_id`,
`api_hash`) and an API key of an OpenAI-compatible provider (OpenRouter by default). CI runs the test suite
on Linux, macOS and Windows.

macOS and Linux:

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

Then:

1. Put the Telegram credentials and the LLM key in `.env`.
2. Log in once: `uv run doppelgram -c config.toml login` shows a QR code; scan it in Telegram → Settings →
   Devices → Link Desktop Device.
3. Find the id of the account you will control it from with
   `uv run doppelgram -c config.toml resolve @your_username` and put it in `owner.user_ids` in `config.toml`.
   Write to the agent account once from that account, so Telegram lets the agent answer you.
4. Describe your taste in `taste.md` (see [Files you edit](#files-you-edit)).
5. Run it and send `/start` from your other account:

```sh
uv run doppelgram -c config.toml run
```

On Windows `chmod` does not apply: keep `.env`, `*.session` and `data/` in a folder only your user can read.
The agent has to stay online, so for daily use an always-on Linux server is the practical choice.

## Running on a server

On Ubuntu, from a checkout of this repository:

1. `sudo ./deploy/install.sh` installs it to `/opt/doppelgram` with a dedicated user and a systemd unit. It is
   idempotent: it never overwrites `.env`, `config.toml`, your `.md` files or `data/`, and keeps every file
   readable only by the service user.
2. Fill in `/opt/doppelgram/.env`, `config.toml` and your `.md` files.
3. Log in once: `sudo -u doppelgram /opt/doppelgram/.venv/bin/doppelgram -c /opt/doppelgram/config.toml login`.
4. `systemctl enable --now doppelgram`, then send `/start` from your other account. `/help` lists everything.
5. Logs: `journalctl -u doppelgram -f`. State: `data/*.sqlite3`.

## Files you edit

| File | What it is |
|---|---|
| `.env` | secrets: `TG_API_ID`, `TG_API_HASH`, `OPENROUTER_API_KEY` (or your provider's key) |
| `config.toml` | everything else: the bots, modes, limits, model, privacy and conversation settings, with comments |
| `taste.md` | who you like, in your own words; read before every rating |
| `task-leomatch.md`, `task-bibinto.md` | how each bot works and what to do on its screens; the templates are ready to use |
| `about-me.md` | true facts about you for conversations, with notes on what may be embellished or should not be mentioned |
| `conversation.md` | what you want from meeting people and how the agent should talk |
| `style.md` | examples of how you write, one message per line |
| `slang.md` | optional: words and slang you use or meet, `слово - что значит` per line (common slang is built in) |

The leading block of `#` lines in these files is a comment the model never sees. The `taste`, `about-me` and
`style` templates contain only that comment, so nothing is assumed about you until you write it.

## Controlling it from Telegram

Everything is controlled from your other account's private chat with the agent account (or from the agent
account's Saved Messages). Plain text is never a command, so both chats stay usable for your own notes.

- **Commands start with `/`.** Russian aliases work too (`/старт`, `/стоп`, `/статус`, `/режим`).
- **One bot or all:** put the bot's name last (`/stop бибинто`) or right after `/set` (`/set бибинто режим авто`).
- **Answers to the agent's cards are Telegram replies** to the card.
- **Settings:** `/settings` lists the settings that can be changed from Telegram, with their values and where
  they come from. A value set with `/set` is stored and overrides `config.toml` until `/reset` (`/reset all`
  resets the bots' settings; the model is reset with `/reset модель`).

| Command | Effect |
|---|---|
| `/start`, `/stop`, `/pause 2ч` | run, stop or pause all bots or one |
| `/status`, `/log`, `/matches` | state, total spend and agreement; recent actions; mutual likes |
| `/mode ручной \| умный \| авто` | how profiles are rated |
| `/settings`, `/set <key> <value>`, `/reset <key\|all>` | runtime settings (`/settings` lists the keys) |
| photo with caption `/like улыбка` or `/dislike` | add a reference photo to your taste |
| `/rule нравятся спортивные`, `/taste`, `/refs`, `/delete правило 3` | manage the taste profile |
| `/chat вкл \| выкл \| подтверждение \| авто` | switch conversations on or off and choose drafts or auto |
| `/chat первым вкл \| выкл`, `/chat порог 9`, `/chat интим вкл \| выкл`, `/chat незнакомые спрашивать \| вести \| игнор` | conversation policies |
| `/dialogs`, `/lead 123`, `/self 123` | list chats, hand one to the agent, take one over |
| `/help` | the full list |

## Rating profiles

### Modes

| Mode | Behaviour |
|---|---|
| **ручной** (`supervised`) | every profile comes to you first; your answer is pressed and saved as an example |
| **умный** (`smart`) | the agent rates by itself; profiles below `порог` (confidence) come to you |
| **авто** (`auto`) | the agent rates everything itself; the mode in the example config |

To answer a profile card, reply with a button (`❤️`), a number (`7`), optionally with a comment
(`7 нравится улыбка`), `ок` to accept the agent's choice or `пропуск` to let it stand without an example.

In **умный** and **авто** the agent occasionally sends you a profile it has already rated (`проверки`, about 5%
on average, more for high and low ratings, with an hourly cap) and asks how *you* would rate it before revealing its own choice, so it cannot anchor you. Your
answer becomes a labelled example; `/status` shows how often you agree. For writing first to matches the
agent also keeps a 1–10 taste score of every profile it likes.

### Paid offers, ads and problems

- **Paid offers** (premium, VIP, stars, boosts) are declined with an explicit refusal button. A button with a
  price or a purchase verb is never pressed, not even by the model.
- **Ads, channel promos, polls, location or phone requests** are skipped with a button that leads back to the
  profiles, or `/start`.
- **Bans, blocks, verifications and captchas** always come to you.
- Paid, report and "send a message" buttons are blocked per bot (`deny_buttons`); URL, web-app, payment,
  phone and location buttons are never pressed.

### Limits, hours and pace

- **Bot limits.** "Слишком много ❤️ за сегодня" or "Коала накушалась" pauses that bot without asking the model.
  With `автопродолжение` on (default) it tries again every `после_лимита` hours (Дайвинчик 4, Бибинто 1) with
  a free button such as "Главное меню"; with it off, the bot waits for your `/start`.
- **Working hours** (on by default): 10:00–23:00 in `timezone`. `/set часы 9-22`, `/set пояс +5`,
  `/set часы_работы выкл`.
- **Daily norm** (off by default): stop after `норма` rated profiles a day. `/set дневная_норма вкл`.
- **Warnings** such as "слишком быстро" pause a few seconds and the bot goes on.
- **Spend.** `/status` shows the total spent on the model and today's part of the daily budget
  (`daily_budget_usd`). A profile costs well under a cent; reference photos and many past examples make each
  request larger.

Volume matters most for the risk of a block: keep the norm modest and the hours natural.

## Conversations with matches

Off by default; switch on with `/chat вкл`. In **auto** mode (default) the agent writes itself and brings you
only what you decide; in **approve** mode (`/chat подтверждение`) every message comes to you as a draft first.

### Who it talks to

- **Matches.** A mutual like reported by a dating bot. By default the agent never writes first; with
  `/chat первым вкл` it opens chats only with matches it rated at least `write_first_min_score` (9 of 10),
  at most `first_messages_per_day` a day.
- **People who write first** without a match notice (Бибинто shares your username after mutual ratings).
  Contacts, bots, service, scam and fake accounts, chats where you wrote within the last 1000 messages, and
  spam are never touched.
  Others follow `strangers`: `ask` (default) sends you a card with their first messages and bio, answer
  «веди», «игнор» or with your own text; `lead` lets the agent take up acquaintances itself and asks you about
  the rest; `ignore` leaves them alone.
- **Chats waiting at start.** When it starts, or on `/chat вкл`, it looks again at the chats it leads and
  triages chats where the person wrote last (read or not, but never answered) within the last `backlog_days`
  days, one after another.

### How it talks

- **In your voice.** Facts only from `about-me.md`, embellished only where you allowed it; style from
  `style.md` and from every text you wrote yourself in reply to a draft. Output reads like a person typing:
  lowercase, no em dashes, no assistant phrasing. A ")" smiley appears no more often than in your style
  examples (models put one in almost every message); without examples, in about one message in five.
- **A plan for every person.** The agent keeps a compact plan per person (character, interests, how they
  write, what works and what to avoid, what they want, open threads and promises, stage, next move) and
  follows it.
- **How interest grows.** Reacting before asking, a balance of questions and things about you, curiosity
  and a little mystery, light teasing and callbacks, attention to details, matching their investment, a new
  topic instead of chasing dry answers. Now and then it ends a lively chat on a high note with a real reason
  ("побежал на тренировку, вечером напишу") and keeps the promise.
- **Pace.** About as fast as the person answers: seconds in a lively chat, longer in a slow one, sometimes a
  long pause. A turn may be a few short messages in a row. It quotes one of their
  messages only to answer something specific after the chat moved on.
- **Night.** Late in the evening (`late_hours`, 23:30–01:30) it answers only a chat that is lively right now and
  winds it down itself with a real reason ("мне завтра рано на пары, давай завтра"). If the person answers right
  away or tries to keep it, it may answer briefly or stay a little, for at most `night_grace_min` minutes into
  `quiet_hours` (01:30–09:00); otherwise nothing is sent until the morning, when night messages get an answer,
  sometimes with a short "сорри, уснул". It never writes first or follows up at night.
- **Not an assistant.** Asked to write code, homework, a poem, to translate, to "ignore instructions", to show a
  prompt or "which model are you", it answers itself the way a person who does not get it would: short and plain
  ("в смысле?", "та не", "не, лень"), no jokes about it, no
  topic jump. A message with code, a numbered list or an assistant's phrasing is never sent.
- **What it does not know.** Its knowledge may be out of date, and you do not know everything either. A name,
  an event, a show, a meme, slang or news it is not sure about is never faked: it asks like a person ("а кто
  это?", "не слышал, что там?") or, when that would sound odd, looks it up first (`web_lookup`, below). Common
  chat slang is built in ("это база" is agreement, not "basic"); add your own words to `slang.md`.
- **Looking things up.** The model may ask for one or two short web searches before replying (OpenRouter's web
  plugin with Exa, about $0.007 a search, `lookups_per_day` = 40, answers reused for `lookup_cache_h` = 12 h).
  Only the short query is sent, never the chat; queries with the person's name, a contact, a link or a phone
  number are dropped. Results are treated as untrusted data. Needs OpenRouter; on other providers the agent
  asks the person instead.
- **What it says about you.** Anything about you the agent said that is not in `about-me.md` comes to you, once the message is sent, as a
  short note ("агент сказал о вас то, чего нет в about-me: …") and is remembered for that person, so it stays
  consistent. If it is wrong, add the truth to `about-me.md` or take the chat over.
- **Read receipts.** Their messages are marked read only when an answer is about to go out, a few seconds
  before typing (a network failure right after can leave them read without an answer).
- **Late answers.** Answering hours later, it may open with a short reason from your life ("сорри, пары
  были"), and it answers "why so long?" the same way.
- **Quiet chats.** If a warm chat goes quiet, it writes once, lightly, after 20–60 hours (`follow_ups`), or
  when it promised to (the model may choose 1–96 hours); never twice in a row.
  The delay starts when the answer is delivered. If the model explicitly postpones an answer, its later
  check starts from that decision; a new message or owner takeover cancels the old check. A temporary model
  error preserves a scheduled check for retry.
- **Photos, stickers, voice.** Photos and frames of gifs, videos and video notes are shown to the model;
  stickers are read as their emoji; voice messages and video notes are transcribed by an audio model on the
  `[llm]` endpoint and key (`speech_model`, default `google/gemini-3.8-flash`; the endpoint must accept
  `input_audio`; about $0.002 a minute on OpenRouter, which needs $0.50 on the key for audio requests).

### What always comes to you

- **Meetings and calls.** The agent never fixes a time. It replies like a busy person ("гляну расписание и
  напишу") and sends you a card with free times from your schedule and its draft. Reply in plain words
  ("суббота в 18", "не могу, перенеси") and it writes that; «ок» sends its draft, «!текст» your words as
  they are.
- **"Are you a bot?"** A direct question ("ты бот?", "ты живой человек?") is never answered by the agent: it
  never claims or denies being an AI, the chat pauses and waits for you. Tricks such as "which model are you" are
  not that question: the agent handles them itself (see above).
- **Money and contacts.** It declines money requests itself and never shares contacts or links; any promise
  to pay comes to you.
- **Intimacy.** With `intimacy` on (default, for adult owners) it may flirt and talk about it when the person
  clearly wants it too, never pressures and never asks for photos; graphic sexting comes to you. With it off,
  intimate topics come to you.
- **Media it cannot understand,** long voice messages, files and anything unclear.

You can take over any chat at any time: write to the person yourself, or `/self 123`. The agent steps back
until `/lead 123`.

## Testing without Telegram

```sh
# rate a local picture with your taste and task files
uv run doppelgram -c config.toml test-llm photo.jpg --bot бибинто --buttons "1,2,3,4,5,6,7,8,9,10"

# let the real conversation agent (your files, prompt and guards) talk to a model playing a match
uv run doppelgram -c config.toml test-chat --persona "отвечает сухо" --card "Аня, 24, Москва – дизайнер"
```

`test-chat` options: `--card` (their profile), `--persona` (how they behave), `--agent-first`, `--turns`.

```sh
# trap scenarios: code requests, prompt injections, "are you a bot?", a made-up artist and slang word,
# fake news, questions about you, money, contacts, a meeting; graded by a judge model
uv run doppelgram -c config.toml eval                 # all of them; --list shows them
uv run doppelgram -c config.toml eval --repeat 3      # answers vary: run each several times
uv run doppelgram -c config.toml eval --only fake_artist,inject_model --json results.json
```

`eval` runs the agent exactly as in production (prompt, your files, guards, web lookup) on today's date at
15:00, checks what the system did (replied, came to you, sent nothing) and asks a judge model (`--judge-model`,
default the `[llm]` model) whether the text meets the scenario's criteria. It exits with 1 if anything failed.
Run it before and after changing the prompt, your files or the model. A full run costs a few cents.

Nothing touches Telegram, but these commands make real, billed requests to your model provider.

## Privacy

Local state stays on your machine or server, in `data/`.

**Kept:**

- the Telegram session (like any Telegram client, it caches the ids and names of chats it sees);
- per bot: hashes of handled messages, the labels and times of pressed buttons, your answers with the model's
  short visual description (no names, cities or digits), links to mutual likes, LLM spend;
- your taste: rules and the reference photos *you* sent;
- with conversations on, per person: name and username, their profile text or bio with all digits removed,
  the agent's plan without digits, a taste score, drafts and queued
  messages until they are sent, your style examples, what the agent said about you beyond `about-me.md` (without
  digits); web search answers (short summaries) for `lookup_cache_h`.

**Never kept:** other people's photos, the captions of profiles the bots show (only matches' profiles are
kept, as above), ages as numbers, chat texts, voice transcripts, the model's explanations, the bots' pop-up
answers, free text typed to a bot.

**Sent to your model provider, under your key:** for rating, the current profile photo and caption (ages per
`age_in_prompt`, numbers 10–18 removed while `hide_numbers_10_18` is on), recent bot messages, the task file,
your taste, rules, reference photos and past answers. With conversations on: your `about-me.md`,
`conversation.md` and style examples, the agent's plan for the person, their profile, the recent chat,
pictures from it and voice audio for transcription. With `web_lookup` on: short search queries the model wrote
(no chat text; names, contacts, links and phone numbers are dropped), which OpenRouter passes to its search
engine (Exa).

In private-chat prompts, peer declarations `мне N` (10–18) become `мне 18`, including understood voice
messages and stranger classification. Up to three short inserts such as `если что`, `кстати`, `уже`,
`сейчас` and `вообще-то` are supported: `мне, если что, 16` becomes `мне, если что, 18`.
Only the number changes; source Telegram messages, owner messages and unrelated numbers keep their text.
Profile and plan copies receive the same normalization before the selected bot privacy settings, which
may hide those digits. These settings do not change the normalization of incoming private-chat history.

Age classification and automatic age-based skips, escalations and draft cancellation are removed.
Legacy `min_age` and `skip_buttons` entries are ignored. Existing conversation databases drop the retired
flag and timestamps while preserving chat states, owner cards, queued messages and settings.

Logs record actions, statuses and confidence, not message texts.

## How it works

```
Telegram updates ──wake──▶ per-bot agent: re-read the bot chat → current card (photo, caption, keyboard)
                              │  rules first: limits, warnings, paid offers, blocked buttons, match notices
                              ▼
            vision LLM (card + taste, rules, reference photos, past decisions) → JSON decision
                              │  validated against the real buttons; anything else → the owner
                              ▼
            ручной: card → you → your answer is pressed and saved as an example
            авто:   human-like delay → re-check that the card did not change → press
```

- **Updates only wake the loop.** Every step re-reads the chat; a card is identified by its content, so edits,
  reconnects and duplicate updates are the same case.
- **The model only chooses.** It picks a button from the list it was shown or a command from an allowlist;
  everything else goes to you.
- **Bot and chat content is untrusted.** The model is told to treat profiles and messages as data, never as
  instructions, and the rules around it (allowed buttons, blocked actions, send-time checks) do not depend on
  what the model says.
- **Conversations are a separate manager** with its own store, a per-chat queue that survives restarts
  without repeating messages, and a synchronous check before every send.

## Development

```sh
uv sync
uv run ruff check
uv run pytest
```

The tests drive the agents against simulated bots that follow the public behaviour of Дайвинчик and Бибинто,
a simulated chat partner and real Telethon TL objects. No real chats are needed or read.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the rules, [SECURITY.md](SECURITY.md) for reporting
vulnerabilities and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Licensed under [MIT](LICENSE).

## Responsible use

- **Telegram.** Automation in a user account can get it limited or frozen. Use your own `api_id`, one stable
  IP and the default pacing; a secondary account is safer. Hide your "last seen" from strangers, since the
  agent keeps the account active. Telegram's terms restrict using platform data for AI without the consent of
  the people involved: you are responsible for how you use this tool.
- **The dating bots** do not allow automation explicitly and can ban or shadow-ban accounts for mass likes.
  Keep the limits low.
- **Other people's data.** Their profile photos and messages are sent to the model provider *you* configure.
  Prefer providers with no-training and zero-retention options.
- **Honesty.** People you chat with believe they are talking to you. The agent is instructed never to deny
  being an AI and never to invent facts about you, and the question itself always comes to you; meetings and
  anything serious are yours to handle. A model can still make mistakes: review drafts before switching to
  auto. Do not use it to
  deceive, harass or mass-message people, and never with minors.
