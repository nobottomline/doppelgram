# Security policy

Doppelgram holds a Telegram session (full access to an account) and API keys, so please report security
problems privately.

## Reporting

Use GitHub's *Report a vulnerability* (Security → Advisories) on this repository. Do not open a public issue
for anything that could expose a session, keys, owner commands or other people's data. Expect an answer within
a week.

In scope: anything that lets someone other than the configured owner control the agent, leaks the session,
keys or stored data, makes the agent pay, send contacts or personal data, or bypasses the owner's approval.

## Keeping your installation safe

* The `*.session` file is a login to your account. Keep it and `.env` readable only by you (`chmod 600`)
  and the `data/` folder accessible only to you (`chmod 700`); the Linux installer does both. Never commit or
  share them.
* Restrict `owner.user_ids` to your own other account. Commands from anyone else are ignored.
* If a session may have leaked, end it in Telegram → Settings → Devices and log in again.
