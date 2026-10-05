#!/usr/bin/env bash
# Бэктест на этом сервере вместо GitHub Actions (без очереди за машинами).
#
#   bash scalper/deploy/research.sh tline     крипта: правило бота, ТФ, масштабы линий, ликвидность, лонги, статьи
#   bash scalper/deploy/research.sh tradfi    золото, серебро, нефть
#
# Считает в отдельном контейнере python:3.12-slim на всех ядрах, кроме одного (оно остаётся боту), с пониженным
# приоритетом. Данные Binance кешируются в $RESEARCH_DATA (по умолчанию ~/scalper-research): первый прогон качает
# историю (десятки минут), следующие — только новые месяцы.
# Результат — $RESEARCH_DATA/out/<что>-<время>/log.txt. Если в ~/.scalper-research-token лежит GitHub-токен с правом
# Contents: write на этот репозиторий, лог ещё и публикуется в ветку research-results (runs/vps-<что>-<время>).
set -euo pipefail

WHAT=${1:-tline}
case "$WHAT" in tline|tradfi) ;; *) echo "usage: research.sh tline|tradfi"; exit 2 ;; esac

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${RESEARCH_DATA:-$HOME/scalper-research}
CORES=$(nproc)
JOBS=${JOBS:-$(( CORES > 1 ? CORES - 1 : 1 ))}
STAMP=$(date -u +%Y%m%d-%H%M)
OUT="$DATA/out/$WHAT-$STAMP"
mkdir -p "$DATA/bn" "$DATA/pip" "$OUT"

echo "== $WHAT: ядер $CORES, параллельно $JOBS, данные $DATA, результат $OUT"
docker run --rm --name "scalper-research-$WHAT" --cpus="$JOBS" \
  -e WHAT="$WHAT" -e JOBS="$JOBS" \
  -e PYTHONDONTWRITEBYTECODE=1 -e NUMBA_CACHE_DIR=/tmp/numba \
  -v "$REPO":/repo:ro -v "$DATA/bn":/root/bn -v "$DATA/pip":/root/.cache/pip -v "$OUT":/out \
  -w /repo/scalper python:3.12-slim nice -n 10 bash /repo/scalper/deploy/research_inner.sh \
  2>&1 | tee "$OUT/run.txt"

TOKEN_FILE="$HOME/.scalper-research-token"
if [ -s "$TOKEN_FILE" ] && [ -s "$OUT/log.txt" ]; then
  RR="$DATA/rr"
  URL="https://x-access-token:$(cat "$TOKEN_FILE")@github.com/Roman-Petukhov/scalper.git"
  rm -rf "$RR"
  git clone -q --depth 1 -b research-results "$URL" "$RR"
  DEST="runs/vps-$WHAT-$STAMP"
  mkdir -p "$RR/$DEST"
  cp "$OUT"/*.txt "$RR/$DEST/"
  git -C "$REPO" rev-parse HEAD > "$RR/$DEST/commit.txt"
  git -C "$RR" -c user.name="scalper-vps" -c user.email="scalper-vps@users.noreply.github.com" add runs
  git -C "$RR" -c user.name="scalper-vps" -c user.email="scalper-vps@users.noreply.github.com" \
    commit -qm "vps $WHAT run $STAMP ($(git -C "$REPO" rev-parse --short HEAD))"
  for i in 1 2 3 4; do
    git -C "$RR" push -q origin research-results && break
    git -C "$RR" pull -q --rebase origin research-results
  done
  rm -rf "$RR"
  echo "== опубликовано: $DEST"
else
  echo "== готово: $OUT/log.txt"
fi
