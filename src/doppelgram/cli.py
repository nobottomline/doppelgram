"""Command line: login, run, resolve, test-llm."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import sys
from pathlib import Path

from .config import ConfigError, load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("telethon").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _session_files(session: Path) -> list[Path]:
    """Telethon adds ".session" unless the name already ends with it; the journal sits next to it."""
    base = str(session)
    main = base if base.endswith(".session") else base + ".session"
    return [Path(main), Path(main + "-journal")]


def _secure_session(cfg) -> None:
    for p in _session_files(cfg.telegram.session):
        if p.exists():
            os.chmod(p, 0o600)


MAX_PASSWORD_ATTEMPTS = 5


async def _password_login(client) -> None:
    """Ask for the cloud (2FA) password until it is accepted."""
    from telethon import errors

    for attempt in range(1, MAX_PASSWORD_ATTEMPTS + 1):
        try:
            await client.sign_in(password=getpass.getpass("2FA password: "))
            return
        except errors.PasswordHashInvalidError:
            left = MAX_PASSWORD_ATTEMPTS - attempt
            print(f"Wrong password. Attempts left: {left}." if left else "Wrong password.")
        except errors.FloodWaitError as exc:
            raise SystemExit(f"Telegram asks to wait {exc.seconds} s before the next password attempt. "
                             f"Run `login` again after that; the QR step is remembered.") from exc
    raise SystemExit("Too many wrong passwords. Run `login` again later; the QR step is remembered.")


async def _qr_login(client) -> None:
    import qrcode
    from telethon import errors

    try:
        qr_login = await client.qr_login()
    except errors.SessionPasswordNeededError:
        # The QR was already accepted in a previous run; only the password is missing.
        await _password_login(client)
        return
    while True:
        qr = qrcode.QRCode(border=1)
        qr.add_data(qr_login.url)
        print("\nScan with Telegram on your phone: Settings → Devices → Link Desktop Device\n")
        qr.print_ascii(invert=True)
        try:
            await qr_login.wait(timeout=60)
            return
        except TimeoutError:
            try:
                await qr_login.recreate()
            except errors.SessionPasswordNeededError:
                await _password_login(client)
                return
        except errors.SessionPasswordNeededError:
            await _password_login(client)
            return


async def cmd_login(cfg, args) -> int:
    from telethon import errors

    from .telegram import make_client

    client = make_client(cfg)
    await client.connect()
    try:
        if await client.is_user_authorized():
            me = await client.get_me()
            print(f"Already logged in as {me.first_name} (@{me.username}, id {me.id}).")
            return 0
        try:
            if args.phone:
                await client.start(phone=lambda: input("Phone (+7...): "),
                                   password=lambda: getpass.getpass("2FA password: "), max_attempts=5)
            else:
                await _qr_login(client)
        except errors.FloodWaitError as exc:
            raise SystemExit(f"Telegram asks to wait {exc.seconds} s before trying again. "
                             f"Run `login` again after that.") from exc
        me = await client.get_me()
        print(f"Logged in as {me.first_name} (@{me.username}, id {me.id}).")
        return 0
    finally:
        await client.disconnect()
        _secure_session(cfg)


async def cmd_resolve(cfg, args) -> int:
    from .telegram import make_client

    client = make_client(cfg)
    await client.connect()
    try:
        if not await client.is_user_authorized():
            print("Not logged in; run `doppelgram login` first.", file=sys.stderr)
            return 1
        target = args.who
        entity = await client.get_me() if target == "me" else await client.get_entity(target)
        print(json.dumps({"id": entity.id, "username": getattr(entity, "username", None),
                          "name": getattr(entity, "first_name", None) or getattr(entity, "title", None),
                          "bot": bool(getattr(entity, "bot", False))}, ensure_ascii=False))
        return 0
    finally:
        await client.disconnect()


async def cmd_run(cfg, args) -> int:
    from .agent import Agent
    from .llm import LLMClient
    from .storage import Store
    from .supervisor import Supervisor
    from .taste import TasteProfile
    from .telegram import TelegramAccount

    missing = [t.name for t in cfg.targets if not t.bot]
    if not cfg.targets or missing:
        print("config error: every [[bots]] entry needs a username" + (f" (missing: {', '.join(missing)})"
              if missing else ""), file=sys.stderr)
        return 2
    if not cfg.owner_ids:
        logging.warning("owner.user_ids is empty: commands only via Saved Messages")
    cfg.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    account = TelegramAccount(cfg)
    llm = LLMClient(cfg.llm)
    try:
        channels = await account.start()
        _secure_session(cfg)
        multi = len(cfg.targets) > 1
        profiles: dict[str, TasteProfile] = {}
        for t in cfg.targets:  # the first bot using a profile names its taste file
            name = t.behavior.taste_profile
            profiles.setdefault(name, TasteProfile(name, cfg.data_dir, t.behavior.taste_file))
        agents = {t.name: Agent(cfg, channels[t.name], llm, Store(t.db_path), behavior=t.behavior,
                                label=t.name if multi else "", taste=profiles[t.behavior.taste_profile],
                                name=t.name)
                  for t in cfg.targets}
        from .conversation import ConversationManager, ConversationStore
        from .telegram import PeopleChannel

        conversations = None
        if cfg.conversations is not None:
            conversations = ConversationManager(cfg.conversations, llm, PeopleChannel(account),
                                                ConversationStore(cfg.data_dir / "conversations.sqlite3"),
                                                account.notify_owner)
        supervisor = Supervisor(agents, account.notify_owner, profiles, conversations,
                                control=Store(cfg.data_dir / "control.sqlite3"), llm=llm,
                                allowed_models=[cfg.llm.model, *cfg.llm.allowed_models])
        account.router = supervisor
        loop_task = asyncio.create_task(supervisor.run_forever())
        conn_task = asyncio.create_task(account.client.run_until_disconnected())
        done, pending = await asyncio.wait({loop_task, conn_task}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            t.result()  # surface the exception, systemd restarts us
        logging.error("Telegram connection closed")
        return 1
    finally:
        await llm.aclose()
        await account.client.disconnect()


async def cmd_test_llm(cfg, args) -> int:
    """Rate a local image with the configured model and buttons, without Telegram."""
    from . import brain
    from .llm import LLMClient
    from .media import image_data_url
    from .models import BotMessage, Button
    from .storage import Store

    labels = [s.strip() for s in args.buttons.split(",") if s.strip()]
    buttons = tuple(Button(id=f"r0c{i}", row=0, col=i, text=t, kind="callback", data=str(i).encode())
                    for i, t in enumerate(labels))
    msg = BotMessage(id=1, text=args.text, buttons=buttons, media_kind="photo", media_key="local")
    target = cfg.target(args.bot) if cfg.targets else None
    behavior = target.behavior if target else cfg.behavior
    examples = []
    if target and target.db_path.exists():
        examples = Store(target.db_path).recent_examples(behavior.examples_in_prompt)
    from .storage import read_user_text  # the same reading as the agent: template comments dropped
    from .taste import TasteProfile

    references = []
    taste_text = read_user_text(behavior.taste_file)
    if cfg.data_dir.is_dir():
        profile = TasteProfile(behavior.taste_profile, cfg.data_dir, behavior.taste_file)
        taste_text = profile.text()
        references = [(r.liked, r.note, image_data_url(r.path.read_bytes(), 768))
                      for r in profile.prompt_references(behavior.max_reference_images)]
    def sanitize(text: str) -> str:  # the same privacy settings as the running agent
        return brain.sanitize_text(text, behavior.age_in_prompt, behavior.age_pattern, behavior.hide_numbers_10_18)

    messages = brain.build_messages(
        task=read_user_text(behavior.task_file), taste=taste_text, examples=examples, references=references,
        history=[], recent_actions=[], current=msg,
        sanitize=sanitize if (behavior.age_in_prompt != "keep" or behavior.hide_numbers_10_18) else None,
        image_url=image_data_url(Path(args.image).read_bytes(), cfg.llm.image_max_side),
        image_from_previous=False, notes=[])
    llm = LLMClient(cfg.llm)
    try:
        result = await llm.complete_json(messages)
    finally:
        await llm.aclose()
    d = brain.validate_decision(result.data, msg, max_wait_s=behavior.max_wait_s,
                                max_send_chars=behavior.max_send_chars, deny=tuple(behavior.deny_buttons),
                                allowed_commands=tuple(dict.fromkeys([*behavior.start_command.split()[:1],
                                                                      *behavior.allowed_commands])))
    chosen = msg.button(d.button_id or "")
    print(json.dumps({
        "model": result.model, "action": d.action, "button": chosen.text if chosen else None,
        "confidence": d.confidence, "reason": d.reason, "description": d.description,
        "tokens": [result.prompt_tokens, result.completion_tokens], "cost_usd": round(result.cost_usd, 6),
    }, ensure_ascii=False, indent=2))
    return 0


async def cmd_test_chat(cfg, args) -> int:
    from .llm import LLMClient
    from .simulate import DEFAULT_PERSONA, simulate

    if cfg.conversations is None:
        print("no [conversations] section in the config", file=sys.stderr)
        return 2
    llm = LLMClient(cfg.llm)
    try:
        print(f"Симуляция: {args.card}\n")
        await simulate(cfg.conversations, llm, card=args.card, persona=args.persona or DEFAULT_PERSONA,
                       turns=args.turns, agent_first=args.agent_first)
    finally:
        await llm.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="doppelgram", description=__doc__)
    p.add_argument("-c", "--config", default="config.toml", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("login", help="authorize the Telegram account (QR by default)")
    lg.add_argument("--phone", action="store_true", help="log in with phone number + code instead of QR")
    sub.add_parser("run", help="run the agent")
    rs = sub.add_parser("resolve", help="print the numeric id of a user/bot (or 'me')")
    rs.add_argument("who")
    tl = sub.add_parser("test-llm", help="rate a local image with the configured model")
    tl.add_argument("image")
    tl.add_argument("--buttons", default="1,2,3,4,5,6,7,8,9,10")
    tl.add_argument("--text", default="Оцените рисунок")
    tl.add_argument("--bot", default=None, help="use this bot's task/taste files (default: the first bot)")
    tc = sub.add_parser("test-chat", help="let the conversation agent talk to a simulated match (no Telegram)")
    tc.add_argument("--card", default="Аня, 24, Москва – дизайнер, люблю горы, кофе и долгие прогулки",
                    help="the match's dating profile")
    tc.add_argument("--persona", default=None, help="who the simulated match is and how they behave")
    tc.add_argument("--turns", type=int, default=8)
    tc.add_argument("--agent-first", action="store_true", help="the agent writes the first message")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)
    needs_tg = args.cmd in ("login", "run", "resolve")
    needs_llm = args.cmd in ("run", "test-llm")
    try:
        cfg = load_config(args.config, require_llm=needs_llm, require_telegram=needs_tg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    handler = {"login": cmd_login, "run": cmd_run, "resolve": cmd_resolve, "test-llm": cmd_test_llm,
               "test-chat": cmd_test_chat}[args.cmd]
    try:
        return asyncio.run(handler(cfg, args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
