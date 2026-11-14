#!/usr/bin/env bash
# Install or update doppelgram on Ubuntu. Run as root from the project checkout:
#   sudo ./deploy/install.sh
# Idempotent: never overwrites .env, config.toml, the local task*/taste* .md files or data/.
set -euo pipefail

APP_DIR=/opt/doppelgram
APP_USER=doppelgram
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo)" >&2
  exit 1
fi

apt-get update -qq
apt-get install -y -qq curl ca-certificates rsync >/dev/null

# The service user needs uv too: a copy in root's home (~/.local/bin) is not enough.
UV=/usr/local/bin/uv
if [[ ! -x $UV ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi

id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"

WAS_ACTIVE=0
if systemctl is-active --quiet doppelgram 2>/dev/null; then
  WAS_ACTIVE=1
  systemctl stop doppelgram
fi

mkdir -p "$APP_DIR"
umask 077  # copies of config and personal .md files must not be readable by other users
rsync -a --delete \
  --exclude .git --exclude .venv --exclude .python --exclude .cache --exclude data --exclude .env \
  --exclude config.toml --include '*.example.md' --include '/README.md' --exclude '*.md' \
  "$SRC_DIR"/ "$APP_DIR"/

cd "$APP_DIR"
[[ -f config.toml ]] || cp config.example.toml config.toml
for example in *.example.md; do
  [[ -f "${example%.example.md}.md" ]] || cp "$example" "${example%.example.md}.md"
done
[[ -f .env ]] || cp .env.example .env
mkdir -p data

chown -R "$APP_USER:$APP_USER" "$APP_DIR"
# Only the service user may read anything here: the session, keys, your taste and facts about you.
chmod 700 "$APP_DIR" "$APP_DIR/data"
chmod 600 "$APP_DIR/.env" "$APP_DIR/config.toml" "$APP_DIR"/*.md

sudo -u "$APP_USER" env UV_PYTHON_INSTALL_DIR="$APP_DIR/.python" UV_CACHE_DIR="$APP_DIR/.cache/uv" \
  "$UV" sync --frozen --no-dev --project "$APP_DIR"

install -m 644 "$APP_DIR/deploy/doppelgram.service" /etc/systemd/system/doppelgram.service
systemctl daemon-reload
if [[ $WAS_ACTIVE -eq 1 ]]; then
  systemctl start doppelgram
  echo "doppelgram restarted with the new version."
fi

cat <<EOF

Installed to $APP_DIR. Next steps:
  1. Edit $APP_DIR/.env          (TG_API_ID, TG_API_HASH, OPENROUTER_API_KEY)
  2. Edit $APP_DIR/config.toml   (owner.user_ids; the [[bots]] blocks are ready for Leomatch and Bibinto)
  3. Edit $APP_DIR/taste.md      (your taste; task-*.md describe each bot and work as they are)
  4. Log in:  sudo -u $APP_USER $APP_DIR/.venv/bin/doppelgram -c $APP_DIR/config.toml login
  5. Start:   systemctl enable --now doppelgram && journalctl -u doppelgram -f
  6. From your other account send /start to this account (/help lists the commands).
EOF
