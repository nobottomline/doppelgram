# Security policy

Doppelgram holds a Telegram session (full access to an account) and API keys, so please report security
problems privately.

## Reporting

Use GitHub's *Report a vulnerability* (Security → Advisories) on this repository. Do not open a public issue
for anything that could expose a session, keys, owner commands or other people's data. Expect an answer within
a week.

In scope: anything that lets someone other than the configured owner control the agent, leaks the session,
keys or stored data, makes the agent pay, share contacts or personal data, write to someone it should not
(a friend, an existing chat, a minor), or bypasses the owner's approval. Prompt injection through profiles,
bios, messages, image captions or voice messages that changes what the agent does is in scope too.

## Keeping your installation safe

* The `*.session` file is a login to your account. Keep it and `.env` readable only by you (`chmod 600`)
  and the `data/` folder accessible only to you (`chmod 700`); the Linux installer does both. Never commit or
  share them.
* Restrict `owner.user_ids` to your own other account. Commands from anyone else are ignored.
* If a session may have leaked, end it in Telegram → Settings → Devices and log in again.
* Use a model provider you trust with other people's photos and messages, ideally with no-training and
  zero-retention options; everything is sent under your own key.
