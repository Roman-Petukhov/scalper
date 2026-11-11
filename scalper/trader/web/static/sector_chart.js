// График сектора во вкладке «Скринер» (Lightweight Charts, панели по метрикам): индекс сектора сверх рынка альтов
// с подложкой режима BTC, сводный балл, оборот к норме, доля рыночных покупок, рост открытого интереса.
// Каждая метрика — своя панель со своей шкалой (две шкалы на одной панели читаются неверно).
(() => {
  "use strict";
  const KEY_SECTOR = "sector-chart-sector";
  const KEY_METRICS = "sector-chart-metrics";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (v, d = 1) => (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%";
  const METRICS = {
    score:    { title: "балл", ref: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(2) },
    turnover: { title: "оборот", ref: 1, fmt: (v) => "×" + v.toFixed(2) },
    taker:    { title: "покупки", ref: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(2) },
    oi:       { title: "OI 7 дн.", ref: 0, fmt: (v) => pct(v) },
  };
  const DEFAULT_METRICS = ["score", "turnover"];
  let data = null;                                   // ответ /api/sectors/history, один на загрузку вкладки
  let loading = null;

  const store = {
    get(k, dflt) { try { const v = localStorage.getItem(k); return v === null ? dflt : JSON.parse(v); } catch (_) { return dflt; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* приватный режим */ } },
  };

  async function load() {
    if (data) return data;
    if (!loading) {
      loading = fetch("/api/sectors/history", { credentials: "same-origin", cache: "no-store" }).then(async (r) => {
        if (r.status === 401) { location.href = "/login"; throw new Error("401"); }
        if (!r.ok) throw new Error(String(r.status));
        data = await r.json();
        return data;
      }).finally(() => { loading = null; });
    }
    return loading;
  }

  const points = (days, values) => {
    const out = [];
    (values || []).forEach((v, i) => { if (v !== null && v !== undefined) out.push({ time: days[i], value: v }); });
    return out;
  };

  function theme() {
    return {
      autoSize: true,
      layout: { background: { color: css("--surface") }, textColor: css("--muted"), fontSize: 11,
                fontFamily: 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace',
                panes: { separatorColor: css("--line"), separatorHoverColor: css("--line"), enableResize: false } },
      grid: { vertLines: { visible: false }, horzLines: { color: css("--line") + "55" } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, rightOffset: 1 },
      localization: { locale: "ru-RU", dateFormat: "dd.MM.yy" },
      crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
      handleScroll: false, handleScale: false,
    };
  }

  function build(root) {
    const st = root._sfc || (root._sfc = {});
    if (st.ro) { st.ro.disconnect(); st.ro = null; }
    if (st.chart) { st.chart.remove(); st.chart = null; }
    const box = root.querySelector(".sfc-box");
    const legend = root.querySelector(".sfc-legend");
    const note = root.querySelector(".sfc-note");
    const sec = data && data.sectors[st.sector];
    if (!sec) {
      box.style.height = "0";
      legend.textContent = "";
      note.textContent = data && data.days.length ? "По этому сектору мало данных." : "История копится: график появится после первого пересчёта.";
      return;
    }
    note.textContent = "";
    const days = data.days;
    const metrics = st.metrics.filter((m) => (sec[m] || []).some((v) => v !== null));
    box.style.height = `${(window.innerWidth < 760 ? 190 : 230) + 100 * metrics.length}px`;
    const chart = LightweightCharts.createChart(box, theme());
    st.chart = chart;

    // подложка режима BTC: бычьи дни — зелёная полоса во всю высоту верхней панели
    const regime = chart.addSeries(LightweightCharts.HistogramSeries, {
      priceScaleId: "regime", priceLineVisible: false, lastValueVisible: false, base: 0,
      autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }) }, 0);
    chart.priceScale("regime", 0).applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
    const bullColor = css("--long") + "1f", bearColor = css("--short") + "14";
    regime.setData(days.map((d, i) => data.bull[i] === null
      ? { time: d } : { time: d, value: 1, color: data.bull[i] ? bullColor : bearColor }));

    const index = chart.addSeries(LightweightCharts.BaselineSeries, {
      baseValue: { type: "price", price: 0 }, lineWidth: 2, title: "к рынку",
      topLineColor: css("--long"), topFillColor1: css("--long") + "33", topFillColor2: css("--long") + "05",
      bottomLineColor: css("--short"), bottomFillColor1: css("--short") + "05", bottomFillColor2: css("--short") + "33",
      priceFormat: { type: "custom", formatter: (v) => pct(v), minMove: 0.0001 }, priceLineVisible: false }, 0);
    index.setData(points(days, sec.index));
    const series = [{ s: index, fmt: (v) => pct(v), title: "к рынку" }];

    metrics.forEach((m, k) => {
      const meta = METRICS[m];
      const line = chart.addSeries(LightweightCharts.LineSeries, {
        color: css("--accent"), lineWidth: 2, title: meta.title, priceLineVisible: false,
        priceFormat: { type: "custom", formatter: meta.fmt, minMove: 0.0001 } }, k + 1);
      line.setData(points(days, sec[m]));
      line.createPriceLine({ price: meta.ref, color: css("--muted"), lineWidth: 1, axisLabelVisible: false,
                             lineStyle: LightweightCharts.LineStyle.Dashed });
      series.push({ s: line, fmt: meta.fmt, title: meta.title });
    });
    const panes = chart.panes();                      // верхняя панель — вдвое выше панелей метрик
    panes.forEach((p, i) => p.setStretchFactor(i === 0 ? 2.2 : 1));
    chart.timeScale().fitContent();
    // ширина приходит позже (вкладка была скрыта, поворот экрана) — каждый раз вписываем все 90 дней заново
    if (st.ro) st.ro.disconnect();
    st.ro = new ResizeObserver(() => { if (st.chart === chart) chart.timeScale().fitContent(); });
    st.ro.observe(box);

    const paint = (param) => {
      const idx = param && param.time ? days.indexOf(param.time) : days.length - 1;
      const day = days[idx] || "";
      const parts = series.map(({ s, fmt, title }) => {
        const p = param && param.time ? param.seriesData.get(s) : s.data().at(-1);
        return p && p.value !== undefined ? `${title} ${fmt(p.value)}` : null;
      }).filter(Boolean);
      const reg = data.bull[idx] === true ? "BTC выше EMA50" : data.bull[idx] === false ? "BTC ниже EMA50" : "";
      legend.textContent = [day.split("-").reverse().join("."), ...parts, reg].filter(Boolean).join(" · ");
    };
    chart.subscribeCrosshairMove(paint);
    paint(null);
  }

  function sync(root) {
    const st = root._sfc;
    for (const b of root.querySelectorAll("[data-sector]")) b.setAttribute("aria-pressed", String(b.dataset.sector === st.sector));
    for (const b of root.querySelectorAll("[data-metric]")) b.setAttribute("aria-pressed", String(st.metrics.includes(b.dataset.metric)));
  }

  async function init(root) {
    if (root._sfc && root._sfc.ready) return;
    const st = root._sfc = { ready: true, chart: null,
      sector: store.get(KEY_SECTOR, null), metrics: store.get(KEY_METRICS, DEFAULT_METRICS) };
    if (!Array.isArray(st.metrics)) st.metrics = DEFAULT_METRICS.slice();
    const names = [...root.querySelectorAll("[data-sector]")].map((b) => b.dataset.sector);
    if (!names.includes(st.sector)) st.sector = root.dataset.default || names[0];
    root.addEventListener("click", (e) => {
      const s = e.target.closest("[data-sector]");
      const m = e.target.closest("[data-metric]");
      if (s) { st.sector = s.dataset.sector; store.set(KEY_SECTOR, st.sector); }
      else if (m) {
        const k = m.dataset.metric;
        st.metrics = st.metrics.includes(k) ? st.metrics.filter((x) => x !== k) : Object.keys(METRICS).filter((x) => x === k || st.metrics.includes(x));
        store.set(KEY_METRICS, st.metrics);
      } else return;
      sync(root);
      build(root);
    });
    sync(root);
    try {
      await load();
    } catch (err) {
      root.querySelector(".sfc-note").textContent = "График не загрузился, повторю при следующем открытии вкладки.";
      st.ready = false;
      return;
    }
    build(root);
  }

  function scan(scope) {
    if (!window.LightweightCharts) return;
    for (const el of (scope || document).querySelectorAll("[data-sector-chart]")) {
      if (scope && scope !== document) data = null;          // карточку перерисовали — берём свежие ряды
      init(el);
    }
  }

  document.addEventListener("DOMContentLoaded", () => scan(document));
  document.addEventListener("htmx:load", (e) => scan(e.detail.elt));
})();
