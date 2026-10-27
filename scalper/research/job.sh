#!/usr/bin/env bash
# Задание для .github/workflows/research.yml; запускается из папки scalper/.
#   bash research/job.sh collect   — своя часть (SHARD из NSHARDS, см. research/shard.py), результаты в $PARTS
#   bash research/job.sh report    — сборка итогов из всех частей
# TASKS — какие исследования гонять в этом прогоне (по порядку).
set -euo pipefail
TASKS=${TASKS:-"quarterly"}
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
# Профиль агрессоров внутри свечи пробоя бота 4h по 15m («кластеры»)
footprint_collect() {
  RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.footprint collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
footprint_report() { python -m research.footprint report; }
# Объём за всю консолидацию до пробоя линии бота 4h (Эдвардс и Мэги)
linevol_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.linevol collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
linevol_report() { python -m research.linevol report; }
# Макрособытия США (FOMC, CPI, NFP) и сделки бота 4h; доля выпущенных токенов (CoinGecko, последний год)
macrocal_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.macrocal collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
macrocal_report() { python -m research.macrocal report; }
# Топ движения дня (по разбору Digahka): правило бота на 15m / 5m по всем монетам
hotcoins_collect() {
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  python -m research.hotcoins collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
hotcoins_report() { python -m research.hotcoins report; }
# Ретест от уровня / по трендовой / после обновления свинга (4h и 15m)
retest2_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.retest2 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
retest2_report() { python -m research.retest2 report; }
# Поздний безубыток у цели 3R (4h и 15m)
latebe_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.latebe collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
latebe_report() { python -m research.latebe report; }
# Форма фигуры до пробоя: ширина, кучность, длина свечей (4h и 15m)
shape_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.shape collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
shape_report() { python -m research.shape report; }
# Наклонный канал (отскок от линии, с подтверждением по агрессорам) и ложный пробой с агрессорами разворота
chan_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.chan collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  python -m research.failbo collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
chan_report() {
  python -m research.chan report
  python -m research.failbo report
}
# Линии против канона: касания, длина, глубина пробоя, линия по теням (4h и 15m)
tlcanon_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.tlcanon collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
tlcanon_report() { python -m research.tlcanon report; }
# Срок сделки 50–800 свечей на 15m и 5m (все монеты, как hotcoins)
holdlen_collect() {
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  python -m research.holdlen collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
holdlen_report() { python -m research.holdlen report; }
# Сопровождение на 5m: цель 2 / 2.5R, б/у или часть на +2R, выход по закрытию обратно за линией
exit5_collect() {
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  python -m research.exit5 collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
exit5_report() { python -m research.exit5 report; }
# Угадать в момент +2R, что 3R не будет: скорость, агрессоры, объём, отбой, уровни впереди, растянутость (5m, 15m, 4h)
at2r_collect() {
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  python -m research.at2r collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
at2r_report() { python -m research.at2r report; }
# Модель сопровождения (LightGBM по картине открытой сделки): обучение на IS, проверка на VAL / HO (5m, 15m, 4h)
trademl_collect() {
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  python -m research.trademl collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
trademl_report() {
  pip install -q lightgbm
  python -m research.trademl report
}
# Мета-разметка: LightGBM поверх правила бота — брать ли сигнал и каким объёмом (4h ретест, 15m по рынку, с 2020)
metalabel_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.metalabel collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
metalabel_report() {
  pip install -q lightgbm
  python -m research.metalabel report
}
# Размер риска по Келли для бота 4h: по сделке, счёт с одновременностью, «вперёд», бутстреп по месяцам
kelly_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.margsim collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
kelly_report() { python -m research.kelly report; }
# Мета-разметка со скользящим переобучением каждый квартал (окна 6 / 12 / 24 мес., признаки режима)
metaroll_collect() { metalabel_collect; }
metaroll_report() {
  pip install -q lightgbm
  python -m research.metaroll report
}
# Стоп и цель по ATR против стопа за свингом (4h ретест, 15m пологие шорты, с 2020)
volstop_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.volstop collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
volstop_report() { python -m research.volstop report; }
# Автовыбор стороны по погоде рынка (4h ретест, 15m по рынку, обе стороны, с 2020)
sidegate_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.sidegate collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
sidegate_report() { python -m research.sidegate report; }
# Почему на 15m шорты в плюсе, лонги в минусе: R по сторонам и периодам, вклад funding, стоп, дрейф рынка
sides15_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.sides15 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
sides15_report() { python -m research.sides15 report; }
# То же для 5m: стороны, funding, стоп, дрейф (монеты с 5m-файлами)
sides5_collect() {
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  SIDES15_TFS=5m python -m research.sides15 collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
sides5_report() { SIDES15_TFS=5m python -m research.sides15 report; }
# Порог наклона линии выше 0.025 (15m и 5m, вход по рынку, обе стороны, сигналы бота без ограничения наклона)
slope_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  SLOPE_TFS=15m python -m research.slope collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  SLOPE_TFS=5m python -m research.slope collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
slope_report() { python -m research.slope report; }
# Закрытие свечи пробоя за линией в ATR (min_break_atr): 4h ретест, 15m и 5m по рынку, пологие линии
breakatr_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  BRK_TFS=4h python -m research.breakatr collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  BRK_TFS=15m python -m research.breakatr collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  RESEARCH_METRICS=0 python -m research.prepare hot-wide
  BRK_TFS=5m python -m research.breakatr collect --symbols "$(cat /tmp/syms_hot-wide.txt)"
}
breakatr_report() { python -m research.breakatr report; }
# Нестандартные ТФ: 8h и 30m (с 2020), 3m (из 1m, только HO) — правило бота
oddtf_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  ODD_TFS=8h,30m python -m research.oddtf collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  RESEARCH_METRICS=0 python -m research.prepare lowtf-core70
  ODD_TFS=3m python -m research.oddtf collect --symbols "$(cat /tmp/syms_lowtf-core70.txt)"
}
oddtf_report() { python -m research.oddtf report; }
# Повтор только для 30m (в первом прогоне тренд 2h не поддерживался)
oddtf30_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  ODD_TFS=30m python -m research.oddtf collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
oddtf30_report() { ODD_TFS=30m python -m research.oddtf report; }
# Свечная формация PPR (Pivot Point Reversal): сама по себе, у уровня, ложный пробой; и как фильтр бота
ppr_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  PPR_TFS=1h,4h,15m python -m research.ppr collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
ppr_report() { python -m research.ppr report; }
# KER-фильтр, линия Тейла–Сена и ATR Гармана–Класса против сделок бота (4h ретест, 15m шорты по рынку)
spec_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  SPEC_TFS=4h python -m research.spec collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  SPEC_TFS=15m python -m research.spec collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
spec_report() { python -m research.spec report; }
# Относительная сила монеты к BTC на момент сигнала: фильтр и очередность при лимите позиций (4h ретест, 15m шорты по рынку)
rs_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  RS_TFS=4h python -m research.rs collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  RS_TFS=15m python -m research.rs collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
rs_report() { python -m research.rs report; }
# Документ «стратегии чемпионов»: риск-менеджмент раздела 10 на сделках бота и стратегии Вильямса, Келла, HTF, ORB
champs_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  CRISK_TFS=4h python -m research.crisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  CRISK_TFS=15m python -m research.crisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  python -m research.champs collect --symbols "$(cat /tmp/syms_1h-qualified.txt)" --spot
}
champs_report() { python -m research.crisk report; python -m research.champs report; }
# EMA200 в фильтре тренда бота и «чистая линия» (прокол линии закрытием до пробоя): 4h ретест, 15m по рынку
ema200_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  E200_TFS=4h python -m research.ema200 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  E200_TFS=15m python -m research.ema200 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
ema200_report() { python -m research.ema200 report; }
# EMA200 вместо фильтра агрессоров (лонг — пробой над EMA200, шорт — под): 4h ретест, 15m по рынку
aggr200_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  A200_TFS=4h python -m research.aggr200 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  A200_TFS=15m python -m research.aggr200 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
aggr200_report() { python -m research.aggr200 report; }
# Длинные выходы только на лонгах 4h / 2h (5R, 8R, трейлинг 3 ATR, дневная EMA20, минимум 10 дней) при трёх порогах агрессоров
longexit_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  LX_TFS=4h python -m research.longexit collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  LX_TFS=2h python -m research.longexit collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
longexit_report() { python -m research.longexit report; }
# Выбивают ли лонги по стопам: доля «стоп, потом цель», стоп шире / от линии, перезаход после выноса (4h)
longstop_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.longstop collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
longstop_report() { python -m research.longstop report; }
# Перезаход в шорт после выноса стопа — свежие данные: 15m (главный), 2h (второй), 4h (справка)
reentry_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  RE_TFS=15m python -m research.reentry collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  RE_TFS=2h python -m research.reentry collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  RE_TFS=4h python -m research.reentry collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
reentry_report() { python -m research.reentry report; }
# Правило бота на BTC, ETH и десятке крупных: 15m / 1h / 4h / 1d, лонги и шорты, без агрессоров и с ними
majors_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.majors collect
}
majors_report() { python -m research.majors report; }
# Горизонтальные уровни на лонгах (шорты зеркально): пробой, отбой, ложный пробой; 4h и 1d
longlevels_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  LL_TFS=4h python -m research.longlevels collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  LL_TFS=1d python -m research.longlevels collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
longlevels_report() { python -m research.longlevels report; }
# Вход на откате 30 / 50% после длинной свечи пробоя и лестница риска на 15m в масштабе бота
pullladder_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  PULL_TFS=4h python -m research.pullentry collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  PULL_TFS=15m python -m research.pullentry collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  CRISK_TFS=15m python -m research.crisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
pullladder_report() { python -m research.pullentry report; python -m research.crisk ladder; }
# Паспорт формации у пробоя линии: длина, ширина, касания, возраст, сжатие, стоп (4h ретест, 15m по рынку)
formation_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  FORM_TFS=4h python -m research.formation collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  FORM_TFS=15m python -m research.formation collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
formation_report() { python -m research.formation report; }
# Вид фигуры у пробоя линии: треугольник / клин / канал / коробка / расширение, ритм касаний, свечи внутри
figure_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  FORM_TFS=4h python -m research.figure collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  FORM_TFS=15m python -m research.figure collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
figure_report() { python -m research.figure report; }
# BTC против EMA200 (дневка и неделя, спот с 2017) как фильтр стороны для сделок бота: 4h ретест, 15m по рынку
btc200_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  B200_TFS=4h python -m research.btc200 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  B200_TFS=15m python -m research.btc200 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
btc200_report() { python -m research.btc200 report; }
# Резкость фигуры у пробоя (самая большая свеча, выносы, вынос перед пробоем) и сжатие ATR на 4h как фильтр
sharp_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  FORM_TFS=4h python -m research.sharp collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  FORM_TFS=15m python -m research.sharp collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
sharp_report() { python -m research.sharp report; }
# Повышенный риск (×1.5 / ×2) на сжатых 4h-пробоях против фиксированного риска того же среднего
squeezesize_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  CRISK_TFS=4h python -m research.crisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
squeezesize_report() { python -m research.squeezesize; }
# Живые настройки одним счётом (15m + 4h, маржа, лимиты, потолок риска), «+ уровень» на 15m, риск по волатильности BTC
livebook_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  CRISK_TFS=4h python -m research.crisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  CRISK_TFS=15m python -m research.crisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
livebook_report() { python -m research.livebook; python -m research.squeezesize; }
livebook2_collect() { livebook_collect; }
livebook2_report() { python -m research.livebook 2; }
livebook3_collect() { livebook_collect; }
livebook3_report() { python -m research.livebook 3; }
livebook4_collect() { livebook_collect; }
livebook4_report() { python -m research.livebook 4; }
livebook5_collect() { livebook_collect; }
livebook5_report() { python -m research.livebook 5; }
livebook6_collect() { livebook_collect; }
livebook6_report() { python -m research.livebook 6; }
livebook7_collect() { livebook_collect; }
livebook7_report() { python -m research.livebook 7; }
livebook8_collect() { livebook_collect; }
livebook8_report() { python -m research.livebook 8; }
livebook9_collect() { livebook_collect; }
livebook9_report() { python -m research.livebook 9; }
livebook10_collect() { livebook_collect; }
livebook10_report() { python -m research.livebook 10; }
livebook11_collect() { livebook_collect; }
livebook11_report() { python -m research.livebook 11; }
# Выход по отрыву от EMA (Climax Extension Келла) на сделках бота; EMA-системы чемпионов (Унгер, Гаджала) и чистые EMA
emaexit_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  EXIT_TFS=4h python -m research.emaexit collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
  EXIT_TFS=15m python -m research.emaexit collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
emaexit_report() { python -m research.emaexit report; }
emasys_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.emasys collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
emasys_report() { python -m research.emasys report; }
# Сетка вокруг EMA200 (стратегия владельца): вход лимит / стоп, выходы, мартингейл, шаг по ATR, фильтр пилы; 15m и 1h на 5m
emagrid_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare lowtf-core70
  python -m research.emagrid collect --symbols "$(cat /tmp/syms_lowtf-core70.txt)"
}
emagrid_report() { python -m research.emagrid report; }
# Сетка с отбором по variance ratio / автокорреляции (core70, 15m и 1h из 5m) и продажа недельной волатильности BTC / ETH
vrgrid_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare lowtf-core70
  python -m research.vrgrid collect --symbols "$(cat /tmp/syms_lowtf-core70.txt)"
}
vrgrid_report() { python -m research.vrgrid report; }
vrp_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.vrp collect
}
vrp_report() { python -m research.vrp report; }
# Коробка + пробой EMA10 / EMA20, выход по отрыву от EMA20, EMA200 — тренд и препятствие (стратегия владельца, AVAX 15m)
boxema_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-core70
  python -m research.boxema collect --symbols "$(cat /tmp/syms_15m-core70.txt)"
}
boxema_report() { python -m research.boxema report; }
# Малоликвидные альты: дневной разворот (Zaremba и др. 2021) и отставание от резкого хода BTC на 5m
dayrev_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-all
  python -m research.dayrev collect --symbols "$(cat /tmp/syms_1h-all.txt)"
}
dayrev_report() { python -m research.dayrev report; }
btclag_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-all
  python -m research.btclag collect --symbols "$(cat /tmp/syms_1h-all.txt)"
}
btclag_report() { python -m research.btclag report; }
# Круглые числа (Osler): пробой через круглый уровень, уровень впереди, стоп «за толпой» (4h и 15m)
roundnum_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-wide
  python -m research.roundnum collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
roundnum_report() { python -m research.roundnum report; }
supply_collect() { :; }
supply_report() { python -m research.supply; }
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
# Тренд бумажного счёта: спот вместо перпа, вход внутри дня по уровню канала, лимит сектора, цель волатильности
dboost4_collect() {
  python -m research.extreg collect
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.dboost4 collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
dboost4_report() { python -m research.dboost4 report; }
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
# Правило бота на дневках: спот Binance с 2017 (свечи качает сам) и перпетуалы с 2020, стопы, цели, перезаход
htftrend_collect() {
  python -m research.extreg collect
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.htftrend collect --perp "$(cat /tmp/syms_1h-qualified.txt)" \
    --end "${RESEARCH_END:-$(date -u -d "$(date -u +%Y-%m-01) -1 day" +%Y-%m)}"
}
htftrend_report() { python -m research.htftrend report; }
# Системы известных трейдеров на дневках: Qullamaggie, Connors RSI(2), Raschke Holy Grail, Turtle Soup, Crabel NR7
masters_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.masters collect --perp "$(cat /tmp/syms_1h-qualified.txt)" \
    --end "${RESEARCH_END:-$(date -u -d "$(date -u +%Y-%m-01) -1 day" +%Y-%m)}"
}
masters_report() { python -m research.masters report; }
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

# «Недооценённые альты на месяцы»: этап 1 — проба бесплатной истории (адреса, комиссии, капитализация),
# этап 2 — разворот 1–12 мес. на споте Binance против удержания BTC
onchainprobe_collect() { :; }
onchainprobe_report() { python -m research.onchainprobe report; }
spotrev_collect() { python -m research.spotrev collect; }
spotrev_report() { python -m research.spotrev report; }
# этап 3: ончейн-стоимость (CoinMetrics, DefiLlama) — цены из частей spotrev того же прогона
onchainval_collect() { :; }
onchainval_report() { python -m research.onchainval report; }
# Причинные связки: выплата funding, часы суток, экспирация Deribit, начало месяца, гэп CME
causal_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.causal collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
causal_report() { python -m research.causal report; }
# Цена на активный адрес как фильтр шортов бота 4h
nmfilter_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  CRISK_TFS=4h python -m research.crisk collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
nmfilter_report() { python -m research.nmfilter report; }
# Топливо сквиза: OI против спотового оборота и перекос толпы — анатомия пампов >= +50% за день и сделки
squeezefuel_collect() {
  RESEARCH_START=2022-01 python -m research.prepare 1h-qualified 1h-spot
  python -m research.squeezefuel collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
squeezefuel_report() { python -m research.squeezefuel report; }
# Сектора (ротация, заражение, опережение), фондовый рынок и спотовые ETF
sectors_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-all
  python -m research.sectors collect --symbols "$(cat /tmp/syms_1h-all.txt)"
}
sectors_report() { python -m research.sectors report; python -m research.markets report; }
markets_collect() { sectors_collect; }
markets_report() { python -m research.markets report; }

# Плацебо бота 4h: случайный вход на тех же монетах, месяце, стороне и стопе; фон «шорт любого альта»
placebo_collect() {
  RESEARCH_START=2020-01 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified
  python -m research.placebo collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"
}
placebo_report() { python -m research.placebo report; }

# Фундаментальный скринер на 1–2 года: выручка держателям, FDV, разлоки (DefiLlama + CoinGecko, только сегодня)
fundscreen_collect() { :; }
fundscreen_report() { python -m research.fundscreen; }

# BTC / ETH ниже недельной EMA200 как зона накопления; альты — цены из частей spotrev того же прогона
w200_collect() { python -m research.spotrev collect; }
w200_report() { python -m research.w200; }

case "$MODE" in
  collect|report) ;;
  *) echo "usage: job.sh collect|report"; exit 2 ;;
esac
for t in $TASKS; do
  echo "----- $t $MODE (часть ${SHARD:-0}/${NSHARDS:-1}) -----"
  "${t}_${MODE}" 2>&1 | { grep -v "$FILTER" || true; }
done
