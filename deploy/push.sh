#!/usr/bin/env bash
# Push your local files (and optionally the code) to a server installed with deploy/install.sh.
#
#   ./deploy/push.sh HOST              your .md files: about-me, style, conversation, slang, taste, task-*
#   ./deploy/push.sh HOST --config     also config.toml (the service restarts to read it)
#   ./deploy/push.sh HOST --code       also the committed code (git HEAD), installed with install.sh
#
# HOST is anything ssh accepts (user@host or a Host from ~/.ssh/config); the SSH user must be root or have
# passwordless sudo. The .md files are read before every reply, so they need no restart. Every replaced file is
# first copied to data/backup-<time>/ on the server. .env is never sent: the server keeps its own.
set -euo pipefail

if [[ $# -lt 1 || $1 == -* ]]; then
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  exit 2
fi
HOST=$1
shift
CONFIG=0
CODE=0
for arg in "$@"; do
  case $arg in
    --config) CONFIG=1 ;;
    --code) CODE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

APP_DIR=${APP_DIR:-/opt/doppelgram}
APP_USER=${APP_USER:-doppelgram}
cd "$(dirname "$0")/.."

files=()
for f in about-me.md style.md conversation.md slang.md taste.md task-*.md; do
  [[ -f $f && $f != *.example.md ]] || continue
  # The names go into a remote shell: only plain ones (letters, digits, ".", "_", "-").
  if [[ ! $f =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "skipped (unusual file name): $f" >&2
    continue
  fi
  files+=("$f")
done
((CONFIG)) && files+=(config.toml)
if ((${#files[@]} == 0 && CODE == 0)); then
  echo "nothing to push" >&2
  exit 1
fi

# Root on the server, or sudo without a password.
remote() { ssh "$HOST" "if [ \"\$(id -u)\" -eq 0 ]; then bash -s; else sudo -n bash -s; fi"; }

stamp=$(date +%Y%m%d-%H%M%S)
# macOS tar adds extended attributes the server's tar warns about; leave them out.
tar_opts=()
if tar --version 2>/dev/null | grep -q bsdtar; then
  tar_opts=(--no-mac-metadata --no-xattrs)
fi
if ((${#files[@]})); then
  echo "files: ${files[*]}"
  # One archive over one connection; unpacked into a private temp dir, then installed with the right owner.
  tar "${tar_opts[@]}" -cf - "${files[@]}" | ssh "$HOST" "umask 077; t=\$(mktemp -d); cat > \$t/files.tar; echo \$t" > /tmp/doppelgram-push.$$
  tmp=$(cat /tmp/doppelgram-push.$$)
  rm -f /tmp/doppelgram-push.$$
  remote <<EOF
set -euo pipefail
cd "$APP_DIR"
mkdir -p "data/backup-$stamp"
tar -xf "$tmp/files.tar" -C "$tmp"
for f in ${files[*]}; do
  [ -f "\$f" ] && cp -p "\$f" "data/backup-$stamp/"
  install -o "$APP_USER" -g "$APP_USER" -m 600 "$tmp/\$f" "$APP_DIR/\$f"
done
chown -R "$APP_USER:$APP_USER" "data/backup-$stamp"
chmod 700 "data/backup-$stamp"
rm -rf "$tmp"
echo "installed; previous versions in $APP_DIR/data/backup-$stamp"
EOF
fi

if ((CODE)); then
  # The service is stopped while install.sh replaces the code: on a failure the previous code comes back.
  if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "note: uncommitted changes are not sent; the code comes from git HEAD ($(git rev-parse --short HEAD))"
  fi
  src=/root/doppelgram-src-$stamp
  git archive --format=tar HEAD | ssh "$HOST" "umask 077; t=\$(mktemp -d); cat > \$t/code.tar; echo \$t" \
    > /tmp/doppelgram-push.$$
  tmp=$(cat /tmp/doppelgram-push.$$)
  rm -f /tmp/doppelgram-push.$$
  remote <<EOF
set -euo pipefail
mkdir -p "$src"
tar -xf "$tmp/code.tar" -C "$src"
rm -rf "$tmp"
backup="$APP_DIR/data/backup-$stamp/code"
mkdir -p "\$backup"
rsync -a --exclude data --exclude .venv --exclude .python --exclude .cache "$APP_DIR"/ "\$backup"/
cd "$src"
if ! ./deploy/install.sh > /tmp/doppelgram-install.log 2>&1; then
  tail -20 /tmp/doppelgram-install.log
  echo "install failed: restoring the previous code" >&2
  rsync -a "\$backup"/ "$APP_DIR"/
  chown -R "$APP_USER:$APP_USER" "$APP_DIR"
  systemctl start doppelgram || true
  echo "service: \$(systemctl is-active doppelgram || true)" >&2
  exit 1
fi
rm -rf "$src"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/data/backup-$stamp"
chmod -R go-rwx "$APP_DIR/data/backup-$stamp"
echo "code installed (log: /tmp/doppelgram-install.log)"
EOF
fi

if ((CONFIG || CODE)); then
  remote <<'EOF'
set -uo pipefail
systemctl restart doppelgram
sleep 10
state=$(systemctl is-active doppelgram || true)
echo "service: $state"
journalctl -u doppelgram --since "-15s" --no-pager -o cat | grep -E "WARN|ERROR|Traceback" | tail -5 || true
[ "$state" = active ] || { echo "the service is not running" >&2; exit 1; }
EOF
fi
