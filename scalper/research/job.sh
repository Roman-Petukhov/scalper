#!/usr/bin/env bash
# Задание для .github/workflows/research.yml; запускается из папки scalper/.
#   bash research/job.sh collect   — своя часть (SHARD из NSHARDS, см. research/shard.py), результаты в $PARTS
#   bash research/job.sh report    — сборка итогов из всех частей
# TASKS — какие исследования гонять в этом прогоне (по порядку).
set -euo pipefail
TASKS=${TASKS:-"prepw exits15"}
export TFUP_TFS=${TFUP_TFS:-4h,2h}
export TL_TFS=${TL_TFS:-15m,2h,4h,6h,12h,1d} TL_CHARTS=${TL_CHARTS:-0}
MODE=${1:-collect}
FILTER="Pandas4Warning\|pd.concat\|файлов$\|мес. свечей"

hl_collect() { python -m research.hl_whales collect --wallets 500; }
hl_report() {
  echo "===== HL WHALES: копирование кошельков Hyperliquid (отбор до T, проверка после T) ====="
  python -m research.hl_whales report
}
lux_collect() {
  python -m research.prepare 1h-qualified 15m-core70
  python -m research.lux_rr collect --symbols15 "$(cat /tmp/syms_15m-core70.txt)" --symbols1h "$(cat /tmp/syms_1h-qualified.txt)"
}
lux_report() {
  echo "===== LUX R:R: LuxAlgo Trendlines with Breaks, выход 1:3 / 1:5, вход по пробою и на ретесте ====="
  python -m research.lux_rr report
}

prep_collect() { python -m research.prepare 1h-qualified 1h-all 1h-spot; }
prep_report() { :; }
listings2_collect() { python -m research.listings2 collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
listings2_report() {
  echo "===== LISTINGS2: шорты по анонсам Binance — funding, цены Bybit, шорт после листинга ====="
  python -m research.listings2 report
}
trail_collect() {
  python -m research.trend_trail collect --symbols15 "$(cat /tmp/syms_15m-core70.txt)" --symbols1h "$(cat /tmp/syms_1h-qualified.txt)"
}
trail_report() {
  echo "===== TREND TRAIL: пробои LuxAlgo и Дончиана с подтягиванием стопа (15m / 1h / 4h) ====="
  python -m research.trend_trail report
}
mom_collect() { python -m research.momentum collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
mom_report() {
  echo "===== MOMENTUM: межмонетный (неделя) и по времени (CTA, день) ====="
  python -m research.momentum report
}

export_collect() { python -m research.export_more collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
export_report() { python -m research.export_more report; }

unlocks_collect() { python -m research.unlocks collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
unlocks_report() {
  echo "===== UNLOCKS: разлоки токенов (DefiLlama) и цена перпетуала до и после ====="
  python -m research.unlocks report
}
fundarb_collect() { python -m research.fundarb collect --symbols "$(cat /tmp/syms_1h-spot.txt)"; }
fundarb_report() {
  echo "===== FUND ARB: спот + шорт перпа при высоком funding ====="
  python -m research.fundarb report
}

prepall_collect() { python -m research.prepare 1h-all; }
prepall_report() { :; }
upbit_collect() { python -m research.upbit collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
upbit_report() {
  echo "===== UPBIT: листинги на Upbit и перпетуал Binance ====="
  python -m research.upbit report
}

unlocks2_collect() { python -m research.unlocks2 collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
unlocks2_report() {
  echo "===== UNLOCKS2: устойчивость шорта перед разлоком ====="
  python -m research.unlocks2 report
}

precursors_collect() { python -m research.precursors collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
precursors_report() {
  echo "===== PRECURSORS: стакан и крупные сделки перед иксами ====="
  python -m research.precursors report
}
prepq_collect() { python -m research.prepare 1h-qualified; }
prepq_report() { :; }
prept_collect() { python -m research.prepare 1h-qualified 1h-spot 15m-core70; }
prept_report() { :; }
tline_collect() {
  python -m research.tline collect --symbols "$(cat /tmp/syms_1h-qualified.txt)" --symbols15 "$(cat /tmp/syms_15m-core70.txt)"
}
tline_report() {
  echo "===== TLINE: линия тренда по закрытиям, пробой с закреплением, ретест, лесенка 3R / 5R ====="
  pip install -q lightgbm && python -m research.tline report
}
prepo_collect() { RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified; }
prepo_report() { :; }
oos_collect() { python -m research.oos collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
oos_report() { python -m research.oos report; }
failbo_collect() { python -m research.failbo collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
failbo_report() { python -m research.failbo report; }
targets4_collect() { python -m research.targets4 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
targets4_report() { python -m research.targets4 report; }
targets15_collect() { python -m research.targets15 collect --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
targets15_report() { python -m research.targets15 report; }
hidden_collect() { python -m research.hidden collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
hidden_report() { python -m research.hidden report; }
dtrend_collect() { python -m research.dtrend collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dtrend_report() { python -m research.dtrend report; }
patterns_collect() {
  python -m research.patterns collect --symbols4 "$(cat /tmp/syms_1h-qualified.txt)" --symbols15 "$(cat /tmp/syms_15m-wide.txt)"
}
patterns_report() { python -m research.patterns report; }
dtri_collect() { python -m research.dtri collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dtri_report() { python -m research.dtri report; }
prepi_collect() { RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 1h-all; }
prepi_report() { :; }
improve_collect() {
  python -m research.improve collect --symbols "$(cat /tmp/syms_1h-all.txt)" --qualified "$(cat /tmp/syms_1h-qualified.txt)"
}
improve_report() { python -m research.improve report; }
extension_collect() { python -m research.extension collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
extension_report() { python -m research.extension report; }
extension15_collect() { python -m research.extension15 collect --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
extension15_report() { python -m research.extension15 report; }
regime_collect() { python -m research.regime collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
regime_report() { python -m research.regime report; }
funnel_collect() { python -m research.funnel collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
funnel_report() { python -m research.funnel report; }
pullback_collect() {
  python -m research.pullback collect --symbols "$(cat /tmp/syms_1h-qualified.txt)" --symbols15 "$(cat /tmp/syms_15m-wide.txt)"
}
pullback_report() { python -m research.pullback report; }
trenddef_collect() { python -m research.trenddef collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
trenddef_report() { python -m research.trenddef report; }
noline_collect() { python -m research.noline collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
noline_report() { python -m research.noline report; }
hedged_collect() { python -m research.hedged collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
hedged_report() { python -m research.hedged report; }
ddrisk_collect() { python -m research.ddrisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
ddrisk_report() { python -m research.ddrisk report; }
margsim_collect() { python -m research.margsim collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
margsim_report() { python -m research.margsim report; }
hedge15_collect() { python -m research.hedge15 collect --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
hedge15_report() { python -m research.hedge15 report; }
tfup_collect() { TFUP_TFS=${TFUP_TFS:-4h,12h,1d} python -m research.tfup collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
tfup_report() { python -m research.tfup report; }
prepl_collect() { RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-all; }
prepl_report() { :; }
newlist_collect() { python -m research.newlist collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
newlist_report() { python -m research.newlist report; }
pumpfade_collect() { python -m research.pumpfade collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
pumpfade_report() { python -m research.pumpfade report; }
preplow_collect() { RESEARCH_METRICS=0 python -m research.prepare lowtf-core70; }
preplow_report() { :; }
lowtf_collect() { python -m research.lowtf collect --symbols "$(cat /tmp/syms_lowtf-core70.txt)"; }
lowtf_report() { python -m research.lowtf report; }
deflate_collect() { python -m research.deflate collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
deflate_report() { python -m research.deflate report; }
fibvp_collect() { python -m research.fibvp collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
fibvp_report() { python -m research.fibvp report; }
pyramid_collect() { python -m research.pyramid collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
pyramid_report() { python -m research.pyramid report; }
pyramid15_collect() { python -m research.pyramid collect15 --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
pyramid15_report() { python -m research.pyramid report15; }
dconf_collect() { python -m research.dconf collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dconf_report() { python -m research.dconf report; }
sidefunnel_collect() { python -m research.sidefunnel collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
sidefunnel_report() { python -m research.sidefunnel report; }
longfilt_collect() { python -m research.longfilt collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
longfilt_report() { python -m research.longfilt report; }
trendfam_collect() { python -m research.trendfam collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
trendfam_report() { python -m research.trendfam report; }
hotpull_collect() { python -m research.hotpull collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
hotpull_report() { python -m research.hotpull report; }
flatsell_collect() { python -m research.flatsell collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
flatsell_report() { python -m research.flatsell report; }
extreg_collect() { python -m research.extreg collect; }
extreg_report() { python -m research.extreg report; }
liqmap_collect() { python -m research.liqmap collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
liqmap_report() { python -m research.liqmap report; }
bullongs_collect() { python -m research.bullongs collect; }
bullongs_report() { python -m research.bullongs report; }
prepspot_collect() { RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 1h-spot; }
prepspot_report() { :; }
spotlong_collect() { python -m research.spotlong collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
spotlong_report() { python -m research.spotlong report; }
springbo_collect() { python -m research.springbo collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
springbo_report() { python -m research.springbo report; }
oilong_collect() { python -m research.oilong collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
oilong_report() { python -m research.oilong report; }
topcrowd_collect() { python -m research.topcrowd collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
topcrowd_report() { python -m research.topcrowd report; }
case_collect() { python -m research.case collect; }
case_report() { python -m research.case report; }
range4_collect() { python -m research.range4 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
range4_report() { python -m research.range4 report; }
exhaust_collect() { python -m research.exhaust collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
exhaust_report() { python -m research.exhaust report; }
exhaust15_collect() { python -m research.exhaust collect15 --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
exhaust15_report() { python -m research.exhaust report15; }
candles_collect() { python -m research.candles collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
candles_report() { python -m research.candles report; }
btcline_collect() { python -m research.btcline collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
btcline_report() { python -m research.btcline report; }
btcline15_collect() { python -m research.btcline collect15 --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
btcline15_report() { python -m research.btcline report15; }
entrycandle_collect() { python -m research.entrycandle collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
entrycandle_report() { python -m research.entrycandle report; }
entrycandle15_collect() { python -m research.entrycandle collect15 --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
entrycandle15_report() { python -m research.entrycandle report15; }
rebreak_collect() { python -m research.rebreak collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
rebreak_report() { python -m research.rebreak report; }
rebreak15_collect() { python -m research.rebreak collect15 --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
rebreak15_report() { python -m research.rebreak report15; }
lineexit15_collect() { python -m research.lineexit15 collect --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
lineexit15_report() { python -m research.lineexit15 report; }
bigflow_collect() { python -m research.bigflow collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
bigflow_report() { python -m research.bigflow report; }
joint_collect() { python -m research.joint collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
joint_report() { python -m research.joint report; }
classic_collect() { python -m research.classic collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
classic_report() { python -m research.classic report; }
prepb_collect() { RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-premium; }
prepb_report() { :; }
basis_collect() { python -m research.basis collect --symbols "$(cat /tmp/syms_1h-premium.txt)"; }
basis_report() { python -m research.basis report; }
dtregime_collect() { python -m research.dtrend collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dtregime_report() { python -m research.dtregime report; }
dtiers_collect() { python -m research.dtrend collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dtiers_report() { python -m research.dtiers report; }
dboost_collect() { python -m research.dtrend collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dboost_report() { python -m research.dboost report; }
dboost2_collect() { python -m research.dboost2 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dboost2_report() { python -m research.dboost2 report; }
dboost3_collect() { python -m research.dboost3 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
dboost3_report() { python -m research.dboost3 report; }
bybitflow_collect() { python -m research.bybitflow collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
bybitflow_report() { python -m research.bybitflow report; }
retest15_collect() { python -m research.retest15 collect --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
retest15_report() { python -m research.retest15 report; }
group_collect() { python -m research.group collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
group_report() { python -m research.group report; }
# Квартальная пересдача (research/quarterly.py): данные по прошлый месяц, итог «держится / стареет» в ../out/quarterly.md
quarterly_collect() {
  export RESEARCH_END=${RESEARCH_END:-$(date -u -d "$(date -u +%Y-%m-01) -1 day" +%Y-%m)}
  RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.extreg collect
  python -m research.quarterly collect --symbols "$(cat /tmp/syms_1h-qualified.txt)" --symbols15 "$(cat /tmp/syms_15m-wide.txt)"
}
quarterly_report() { python -m research.quarterly report; }
revgrid_collect() { python -m research.revgrid collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
revgrid_report() { python -m research.revgrid report; }
exits15_collect() { python -m research.exits15 collect --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
exits15_report() { python -m research.exits15 report; }
robust_collect() { python -m research.robust collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
robust_report() { python -m research.robust report; }
crowd_collect() { python -m research.crowd collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
crowd_report() { python -m research.crowd report; }
side_collect() { python -m research.sidecap collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
side_report() { python -m research.sidecap report; }
prepw_collect() { python -m research.prepare 1h-qualified 15m-wide; }
prepw_report() { :; }
wide15_collect() { python -m research.wide15 collect --symbols "$(cat /tmp/syms_15m-wide.txt)"; }
wide15_report() { python -m research.wide15 report; }
fade_collect() { python -m research.fade15 collect --symbols15 "$(cat /tmp/syms_15m-core70.txt)"; }
fade_report() { python -m research.fade15 report; }
donch_collect() { python -m research.donchian collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
donch_report() { python -m research.donchian report; }
smc2_collect() { python -m research.smc2 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
smc2_report() {
  echo "===== SMC2: Smart Money по учебнику (внешняя структура, BOS / CHoCH, discount / premium, цель — ликвидность) ====="
  python -m research.smc2 report
}
smc_collect() { python -m research.smc collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
smc_report() {
  echo "===== SMC: ордер-блоки, имбалансы и пробои на 1h, сетка на ретесте, ML на нескольких ТФ ====="
  pip install -q lightgbm && python -m research.smc report
}
moonshake_collect() { python -m research.moonshake collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
moonshake_report() {
  echo "===== MOONSHAKE: лонг после вытряхивания на сигналах модели ====="
  pip install -q lightgbm && python -m research.moonshake report
}
moonbracket_collect() { python -m research.moonbracket collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
moonbracket_report() {
  echo "===== MOONBRACKET: ловля иксов вилкой стоп-ордеров ====="
  pip install -q lightgbm && python -m research.moonbracket report
}
spikesfull_collect() { python -m research.spikes_full collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
spikesfull_report() {
  echo "===== SPIKES FULL: прострелы по всему рынку и счёт \$100 ====="
  python -m research.spikes_full report
}

case "$MODE" in
  collect|report) ;;
  *) echo "usage: job.sh collect|report"; exit 2 ;;
esac
for t in $TASKS; do
  echo "----- $t $MODE (часть ${SHARD:-0}/${NSHARDS:-1}) -----"
  "${t}_${MODE}" 2>&1 | { grep -v "$FILTER" || true; }
done
