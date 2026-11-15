"""Configuration: non-secret settings from TOML, secrets from the environment."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


class ConfigError(RuntimeError):
    pass


@dataclass
class TelegramConfig:
    api_id: int
    api_hash: str
    session: Path
    device_model: str = "Server"
    system_version: str = "Linux"
    app_version: str = "doppelgram 0.1"
    lang_code: str = "ru"


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    fallback_models: list[str] = field(default_factory=list)
    # Models the owner may switch to from Telegram (/set модель ...); the configured model is always allowed.
    allowed_models: list[str] = field(default_factory=list)
    max_tokens: int = 1500
    temperature: float | None = 0.2
    json_mode: bool = True
    timeout_s: float = 90.0
    image_max_side: int = 1536
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    extra_body: dict = field(default_factory=dict)


@dataclass
class BehaviorConfig:
    task_file: Path
    taste_file: Path
    mode: str = "supervised"  # supervised (every card to the owner) | smart | auto
    autostart: bool = False
    examples_in_prompt: int = 40
    history_in_prompt: int = 6
    settle_s: float = 2.0
    poll_interval_s: float = 60.0
    action_delay_min_s: float = 3.0
    action_delay_max_s: float = 8.0
    stuck_timeout_s: float = 300.0
    max_wait_s: float = 6 * 3600.0
    max_actions_per_hour: int = 120
    daily_budget_usd: float = 1.0
    report_every_h: float = 6.0
    supervise_images_only: bool = True
    # smart mode: picture cards below this model confidence go to the owner
    # (config alias: auto_min_confidence, the former name).
    confidence_threshold: float = 0.7
    # smart mode: if the owner does not answer within this many minutes, the agent applies its own choice
    # (0 = wait for the owner).
    ask_timeout_min: float = 0.0
    # smart/auto: share of autonomous picture decisions sent to the owner afterwards as a non-blocking check.
    spot_check_rate: float = 0.05
    spot_checks_per_hour: int = 3
    # Paid offers (VIP, stars, boosts): on a non-picture message matching this regex the agent presses the
    # first button containing one of `decline_buttons` and goes on, without asking the owner.
    premium_pattern: str = ""
    decline_buttons: list[str] = field(default_factory=list)
    # Which buttons are ratings (only these count as rating a profile: spot checks, owner cards, examples).
    rating_buttons: str = r"^\s*(?:10|[1-9])\s*$|❤|👍|👎"
    # "Someone rated you N": a message matching this regex (group 1 = N) is opened with the first button
    # containing one of `incoming_open_buttons`; the next profile's rating is raised to N if the agent's own
    # rating is at least `reciprocal_min` (it fits the owner's taste).
    incoming_rating_pattern: str = ""
    incoming_open_buttons: list[str] = field(default_factory=list)
    reciprocal_min: int = 6
    # Keep going without the owner (smart/auto): buttons that lead back to the profiles. If the model wants
    # to stop, or keeps waiting on the same short-lived screen, the agent presses one of these (or sends
    # start_command) instead, at most a dozen times an hour. A screen matching `owner_required_pattern`
    # (e.g. a video verification) still goes to the owner.
    continue_buttons: list[str] = field(default_factory=list)
    owner_required_pattern: str = ""
    # The bot's "no more profiles for now" notice (a daily like limit, a full koala). On such a text message
    # the agent pauses without asking the model and, if `limit_resume` is on, every `limit_retry_h` hours
    # presses one of `resume_buttons` (or a continue button, or sends start_command) to try again.
    # With `limit_resume` off it stops and the owner starts it again with /start.
    limit_pattern: str = ""
    limit_resume: bool = True
    limit_retry_h: float = 4.0
    resume_buttons: list[str] = field(default_factory=list)
    # Working hours in `timezone` ("10:00-23:00", may cross midnight): outside them the bot does nothing.
    active_hours_enabled: bool = True
    active_hours: str = "10:00-23:00"
    timezone: str = "Europe/Moscow"  # IANA name or a UTC offset such as "+3"
    # Daily norm: after this many rated profiles in a local day the bot waits for the next day. Off by default.
    daily_ratings_enabled: bool = False
    daily_ratings: int = 150
    # A bot's warning ("слишком быстро", "подозрительная активность") never stops the agent: on such a text
    # message it waits `warning_pause_s` seconds, then goes on (a continue button or start_command).
    warning_pattern: str = (r"слишком быстро|не так быстро|помедленн|притормози|подозрительн\w*\s+активн"
                            r"|слишком (?:много|часто)\s+(?:лайк|оцен|действ)|too fast|slow down|suspicious activity")
    warning_pause_s: float = 8.0
    # Buttons whose text contains any of these substrings are never pressed (paid features, reports, traps).
    deny_buttons: list[str] = field(default_factory=list)
    # Regex that marks a "mutual like" message; links/mentions in it are recorded as a match.
    match_pattern: str = ""
    age_pattern: str = ""  # card age field (first group), used only to sanitize provider-facing text
    # What the LLM provider sees of ages in card texts: "keep" (as is), "18" (the age is replaced by 18),
    # "hide" (all digits are removed). Applied to every bot text sent to the model; local data is untouched.
    age_in_prompt: str = "keep"
    # Remove standalone numbers 10-18 (possible ages) from profile texts sent to the model.
    hide_numbers_10_18: bool = True
    # Shared taste: rules and reference photos live in a named profile (data/taste-<name>.sqlite3).
    taste_profile: str = "default"
    max_reference_images: int = 6
    max_send_chars: int = 200
    start_command: str = "/start"
    # Commands the model may type besides start_command (anything else it wants to send goes to the owner).
    allowed_commands: list[str] = field(default_factory=list)


@dataclass
class ConversationConfig:
    """Chats with matched people. Disabled by default; every message needs approval unless mode = auto."""

    about_file: Path
    goal_file: Path
    style_file: Path
    enabled: bool = False
    mode: str = "approve"  # approve | auto
    # Writing first: off by default (the agent answers only people who wrote). When on, only matches the agent
    # rated at least write_first_min_score (1-10 taste score), at most first_messages_per_day a day.
    # Intimate topics (adults only): flirting, talking about sex, agreeing on an intimate meeting when the
    # person clearly wants it. Off: such topics go to the owner. Graphic sexting always does.
    intimacy: bool = True
    # People who write first without a match notice (e.g. Бибинто shares the username after mutual ratings):
    # "ask" — a card to the owner, who answers «веди» or «игнор»; "lead" — the agent takes up acquaintances
    # by itself and asks the owner about the rest; "ignore". Contacts, bots and existing chats are never touched.
    strangers: str = "ask"
    # What the person sends besides text: pictures are shown to the model; stickers are read as their emoji;
    # voice messages and video notes are transcribed with speech_model on the [llm] endpoint (OpenRouter).
    see_photos: bool = True
    speech: bool = True
    speech_model: str = "google/gemini-3.8-flash"
    speech_fallback_models: list[str] = field(default_factory=lambda: ["google/gemini-2.5-flash"])
    speech_max_seconds: float = 180.0
    write_first: bool = False
    write_first_min_score: int = 9
    first_messages_per_day: int = 3
    max_sends_per_hour: int = 20
    settle_s: float = 25.0  # wait for a burst of incoming messages before answering
    # A busy person's pace, never instant: the delay follows how fast the person answers (mirror_pace),
    # within reply_delay_min_s..reply_delay_max_s, and now and then a long pause (long_delay_chance).
    reply_delay_min_s: float = 60.0
    reply_delay_max_s: float = 5400.0
    mirror_pace: bool = True
    long_delay_chance: float = 0.15
    long_delay_min_s: float = 1800.0
    long_delay_max_s: float = 9000.0
    read_receipts: bool = True  # their messages are marked as read when the agent answers (not before)
    backlog_days: int = 30  # at start: unanswered chats people started in the last N days are triaged (0: off)
    quiet_hours: str = ""  # "01:00-09:00" in `timezone`: no messages are sent then
    timezone: str = "UTC"
    lowercase: bool = True
    history_messages: int = 30
    max_message_chars: int = 400
    daily_budget_usd: float = 1.0



@dataclass
class Target:
    """One bot the account interacts with, each with its own task, mode, limits and state database."""

    name: str  # short name used in owner commands, e.g. "арт" or "likes"
    bot: str  # @username
    behavior: BehaviorConfig
    db_path: Path


@dataclass
class Config:
    telegram: TelegramConfig
    targets: list[Target]
    owner_ids: list[int]
    allow_saved_messages: bool
    llm: LLMConfig
    behavior: BehaviorConfig  # defaults shared by all targets
    data_dir: Path
    conversations: ConversationConfig | None = None

    def target(self, name: str | None) -> Target:
        if not self.targets:
            raise ConfigError("no bots configured")
        if name is None:
            return self.targets[0]
        for t in self.targets:
            if t.name == name:
                return t
        raise ConfigError(f"unknown bot name {name!r}; configured: {', '.join(t.name for t in self.targets)}")


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader; existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def _env(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "").strip()
    if required and not value:
        raise ConfigError(f"environment variable {name} is not set")
    return value


def _path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: Path, *, require_llm: bool = True, require_telegram: bool = True) -> Config:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    base = path.parent.resolve()
    load_dotenv(base / ".env")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    data_dir = _path(base, raw.get("data_dir", "data"))

    tg = raw.get("telegram", {})
    api_id_raw = os.environ.get("TG_API_ID") or tg.get("api_id")
    if require_telegram and not api_id_raw:
        raise ConfigError("telegram.api_id (or TG_API_ID) is required")
    telegram = TelegramConfig(
        api_id=int(api_id_raw or 0),
        api_hash=_env(tg.get("api_hash_env", "TG_API_HASH"), required=require_telegram),
        session=_path(base, tg.get("session", str(data_dir / "account"))),
        device_model=tg.get("device_model", "Server"),
        system_version=tg.get("system_version", "Linux"),
        app_version=tg.get("app_version", "doppelgram 0.1"),
        lang_code=tg.get("lang_code", "ru"),
    )

    owner = raw.get("owner", {})
    owner_ids = [int(x) for x in owner.get("user_ids", [])]

    llm_raw = raw.get("llm", {})
    # api_key_env = "" means the endpoint needs no key (local Ollama / LM Studio / vLLM).
    key_env = llm_raw.get("api_key_env", "OPENROUTER_API_KEY")
    llm = LLMConfig(
        base_url=llm_raw.get("base_url", "https://openrouter.ai/api/v1").rstrip("/"),
        api_key=_env(key_env, required=require_llm) if key_env else "",
        model=llm_raw.get("model", "openai/gpt-6-luna"),
        fallback_models=list(llm_raw.get("fallback_models", [])),
        allowed_models=list(llm_raw.get("allowed_models", [])),
        max_tokens=int(llm_raw.get("max_tokens", 1500)),
        temperature=llm_raw.get("temperature", 0.2),
        json_mode=bool(llm_raw.get("json_mode", True)),
        timeout_s=float(llm_raw.get("timeout_s", 90)),
        image_max_side=int(llm_raw.get("image_max_side", 1536)),
        price_in_per_mtok=float(llm_raw.get("price_in_per_mtok", 0.0)),
        price_out_per_mtok=float(llm_raw.get("price_out_per_mtok", 0.0)),
        extra_body=dict(llm_raw.get("extra_body", {})),
    )

    behavior = _behavior(base, raw.get("behavior", {}), None)
    targets = _targets(base, raw, behavior, data_dir)
    conversations = _conversations(base, raw.get("conversations", {}))

    return Config(
        telegram=telegram,
        targets=targets,
        owner_ids=owner_ids,
        allow_saved_messages=bool(owner.get("allow_saved_messages", True)),
        llm=llm,
        behavior=behavior,
        data_dir=data_dir,
        conversations=conversations,
    )


_CONV_KEYS = {f.name for f in fields(ConversationConfig)}


def _conversations(base: Path, table: dict) -> ConversationConfig:
    values = dict(table)
    for removed in ("owner_age", "min_match_age"):  # removed options: ignored in older configs
        values.pop(removed, None)
    unknown = set(values) - _CONV_KEYS
    if unknown:
        raise ConfigError(f"unknown conversations option(s): {', '.join(sorted(unknown))}")
    if "write_first" not in values and "first_messages_per_day" in values:
        # Before write_first existed, a positive daily number meant "write first".
        values["write_first"] = int(values["first_messages_per_day"]) > 0
    for key, default in (("about_file", "about-me.md"), ("goal_file", "conversation.md"),
                         ("style_file", "style.md")):
        values[key] = _path(base, values.get(key, default))
    cfg = ConversationConfig(**values)
    if cfg.mode not in ("approve", "auto"):
        raise ConfigError("conversations.mode must be 'approve' or 'auto'")
    if cfg.quiet_hours and not re.fullmatch(r"\d{1,2}:\d{2}-\d{1,2}:\d{2}", cfg.quiet_hours):
        raise ConfigError("conversations.quiet_hours must look like '01:00-09:00'")
    if cfg.strangers not in ("ask", "lead", "ignore"):
        raise ConfigError("conversations.strangers must be 'ask', 'lead' or 'ignore'")
    if not 1 <= cfg.write_first_min_score <= 10:
        raise ConfigError("conversations.write_first_min_score must be 1-10")
    if cfg.write_first and cfg.first_messages_per_day < 1:
        raise ConfigError("conversations.write_first needs first_messages_per_day of at least 1")
    if not 0 <= cfg.long_delay_chance <= 1 or cfg.long_delay_max_s < cfg.long_delay_min_s:
        raise ConfigError("long_delay_chance must be 0-1 and long_delay_max_s >= long_delay_min_s")
    if cfg.reply_delay_max_s < cfg.reply_delay_min_s:
        raise ConfigError("reply_delay_max_s must be >= reply_delay_min_s")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(cfg.timezone)
    except Exception as exc:
        raise ConfigError(f"unknown timezone {cfg.timezone!r}") from exc
    return cfg


_BEHAVIOR_KEYS = {f.name for f in fields(BehaviorConfig)}
_NAME_RE = re.compile(r"^[\w-]{1,32}$")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(10)), *(f"LPT{n}" for n in range(10))}


def _behavior(base: Path, table: dict, parent: BehaviorConfig | None) -> BehaviorConfig:
    values = dict(table)
    for removed in ("min_age", "skip_buttons"):  # accept older configs without enforcing the retired gate
        values.pop(removed, None)
    if "hide_numbers_10_20" in values:  # former name
        values.setdefault("hide_numbers_10_18", values.pop("hide_numbers_10_20"))
    if "auto_min_confidence" in values:  # former name
        legacy = values.pop("auto_min_confidence")
        values.setdefault("confidence_threshold", legacy if legacy > 0 else 0.7)
    unknown = set(values) - _BEHAVIOR_KEYS
    if unknown:
        raise ConfigError(f"unknown behavior option(s): {', '.join(sorted(unknown))}")
    if "mode" in values:
        from .settings import MODES
        values["mode"] = MODES.get(str(values["mode"]).lower(), values["mode"])
    for key in ("task_file", "taste_file"):
        if key in values:
            values[key] = _path(base, values[key])
    if parent is None:
        values.setdefault("task_file", _path(base, "task.md"))
        values.setdefault("taste_file", _path(base, "taste.md"))
        result = BehaviorConfig(**values)
    else:
        result = replace(parent, **values)
    if result.mode not in ("supervised", "smart", "auto"):
        raise ConfigError("mode must be 'supervised' (ручной), 'smart' (умный) or 'auto' (авто)")
    if result.age_in_prompt not in ("keep", "18", "hide"):
        raise ConfigError("age_in_prompt must be 'keep', '18' or 'hide'")
    if result.action_delay_max_s < result.action_delay_min_s:
        raise ConfigError("action_delay_max_s must be >= action_delay_min_s")
    from .schedule import parse_tz, parse_window
    try:
        parse_window(result.active_hours)
        parse_tz(result.timezone)
    except ValueError as exc:
        raise ConfigError(f"active_hours/timezone: {exc}") from exc
    if result.daily_ratings < 1:
        raise ConfigError("daily_ratings must be at least 1")
    return result


def _targets(base: Path, raw: dict, behavior: BehaviorConfig, data_dir: Path) -> list[Target]:
    targets: list[Target] = []
    for i, table in enumerate(raw.get("bots", [])):
        table = dict(table)
        name = str(table.pop("name", "")).strip().lower()
        bot = str(table.pop("username", "")).strip()
        if not _NAME_RE.match(name):
            raise ConfigError(f"bots[{i}].name must be a short word (letters, digits, _ or -)")
        if name.split(".")[0].upper() in _WINDOWS_RESERVED:
            raise ConfigError(f"bots[{i}].name {name!r} is a reserved file name on Windows")
        targets.append(Target(name, bot, _behavior(base, table, behavior), data_dir / f"{name}.sqlite3"))
    legacy = raw.get("target", {})
    if legacy.get("bot"):
        if targets:
            raise ConfigError("use either [target] or [[bots]], not both")
        targets.append(Target("main", str(legacy["bot"]).strip(), behavior, data_dir / "state.sqlite3"))
    names = [t.name for t in targets]
    if len(set(names)) != len(names):
        raise ConfigError("bot names must be unique")
    bots = [t.bot.lstrip("@").lower() for t in targets if t.bot]
    if len(set(bots)) != len(bots):
        raise ConfigError("each bot may be configured only once")
    return targets
