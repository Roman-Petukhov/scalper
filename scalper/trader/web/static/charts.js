// Живые графики сигналов (TradingView Lightweight Charts): свечи с текущей, трендовая линия, вход / стоп / цель.
// Опрашиваем только графики, которые видно на экране, и только пока вкладка активна.
(() => {
  "use strict";
  const POLL_MS = { "15m": 5000, "4h": 15000 };
  const live = new Map();                       // элемент → состояние графика
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // Время на оси и в перекрестье — в часовом поясе браузера (свечи приходят в секундах UTC)
  const MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
  const pad = (n) => String(n).padStart(2, "0");
  const stamp = (sec) => {
    const d = new Date(sec * 1000);
    return `${pad(d.getDate())}.${pad(d.getMonth() + 1)} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  };
  const tickLabel = (time, type) => {
    if (typeof time !== "number") return String(time);
    const d = new Date(time * 1000);
    if (type === 0) return String(d.getFullYear());                 // TickMarkType.Year
    if (type === 1) return MONTHS[d.getMonth()];                     // Month
    if (type === 2) return String(d.getDate());                      // DayOfMonth
    return `${pad(d.getHours())}:${pad(d.getMinutes())}`;            // Time
  };

  const theme = () => ({
    layout: { background: { color: css("--surface") }, textColor: css("--muted"), fontSize: 11,
              fontFamily: 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace' },
    grid: { vertLines: { color: css("--line") + "66" }, horzLines: { color: css("--line") + "66" } },
    rightPriceScale: { borderColor: css("--line") },
    timeScale: { borderColor: css("--line"), timeVisible: true, secondsVisible: false, rightOffset: 4,
                 tickMarkFormatter: (time, type) => tickLabel(time, type) },
    localization: { timeFormatter: (time) => typeof time === "number" ? stamp(time) : String(time) },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });

  const precision = (candles) => {
    const p = candles.length ? candles[candles.length - 1].close : 1;
    const d = p >= 1000 ? 2 : p >= 10 ? 3 : p >= 1 ? 4 : p >= 0.01 ? 6 : 8;
    return { type: "price", precision: d, minMove: Math.pow(10, -d) };
  };

  async function fetchData(id) {
    const r = await fetch(`/api/chart/${id}`, { credentials: "same-origin", cache: "no-store" });
    if (r.status === 401) { location.href = "/login"; throw new Error("401"); }
    if (!r.ok) throw new Error(String(r.status));
    return r.json();
  }

  function setLevels(st, d) {
    for (const pl of st.priceLines) st.candles.removePriceLine(pl);
    st.priceLines = [];
    st.levels = d.active ? d.levels : null;
    st.levelsKey = JSON.stringify(d.levels) + d.entry_label + d.active;
    if (!d.active) return;                                  // сигнал истёк или пропущен — вход и стоп не рисуем
    const lv = d.levels;
    const mk = (price, color, title, style) => st.candles.createPriceLine({
      price, color, title, lineWidth: 1, lineStyle: style, axisLabelVisible: true });
    st.priceLines = [
      mk(lv.entry, css("--text"), d.entry_label, LightweightCharts.LineStyle.Solid),
      mk(lv.stop, css("--short"), "стоп", LightweightCharts.LineStyle.Dashed),
      mk(lv.target, css("--long"), "цель", LightweightCharts.LineStyle.Dotted),
    ];

  }

  function create(el, d) {
    el.textContent = "";
    el.classList.add("is-live");
    const box = document.createElement("div");
    box.className = "lc-box";
    const tag = document.createElement("span");
    tag.className = "lc-tag num";
    el.append(box, tag);
    const chart = LightweightCharts.createChart(box, { autoSize: true, ...theme() });
    // линии строятся на логарифмической шкале — и показываем в ней же
    if (d.log) chart.priceScale("right").applyOptions({ mode: LightweightCharts.PriceScaleMode.Logarithmic });
    const st = { chart, tag, priceLines: [], levelsKey: "", levels: d.levels, lastTime: d.candles.at(-1)?.time,
                 timer: null, visible: false, tf: el.dataset.tf, digits: precision(d.candles).precision };
    const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: css("--long"), downColor: css("--short"), wickUpColor: css("--long"), wickDownColor: css("--short"),
      borderVisible: false, priceFormat: precision(d.candles),
      // шкала цены всегда вмещает стоп и цель, даже если цена до них ещё не доходила
      autoscaleInfoProvider: (base) => {
        const r = base();
        if (!r || !r.priceRange || !st.levels) return r;
        const lv = Object.values(st.levels);
        r.priceRange.minValue = Math.min(r.priceRange.minValue, ...lv);
        r.priceRange.maxValue = Math.max(r.priceRange.maxValue, ...lv);
        return r;
      } });
    const line = chart.addSeries(LightweightCharts.LineSeries, {
      color: d.side > 0 ? css("--accent") : css("--warn"), lineWidth: 2, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false });
    candles.setData(d.candles);
    line.setData(d.line);
    if (LightweightCharts.createSeriesMarkers) {
      LightweightCharts.createSeriesMarkers(candles, [{
        time: d.breakout, position: d.side > 0 ? "belowBar" : "aboveBar", color: d.side > 0 ? css("--long") : css("--short"),
        shape: d.side > 0 ? "arrowUp" : "arrowDown" }]);
    }
    const n = d.candles.length;
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 120), to: n + 8 });
    st.candles = candles;
    st.line = line;
    setLevels(st, d);
    paintTag(st, d);
    return st;
  }

  function paintTag(st, d) {
    const c = d.candles.at(-1);
    if (!c) return;
    const now = new Date();
    st.tag.textContent = `● ${c.close.toFixed(st.digits)} · ${LocalTime.format(now, "hms")} ${LocalTime.zone(now)}`;
    st.tag.classList.toggle("up", c.close >= c.open);
    st.tag.classList.toggle("stale", false);
  }

  function apply(st, d) {
    const fresh = d.candles.filter((c) => c.time >= (st.lastTime ?? 0));
    for (const c of fresh) st.candles.update(c);           // текущая свеча меняется, новые добавляются
    if (d.candles.length) st.lastTime = d.candles.at(-1).time;
    st.line.setData(d.line);
    if (JSON.stringify(d.levels) + d.entry_label + d.active !== st.levelsKey) setLevels(st, d);
    paintTag(st, d);
  }

  async function tick(el) {
    const st = live.get(el);
    if (!st || !st.visible || document.hidden || !el.isConnected) return;
    try { apply(st, await fetchData(el.dataset.signal)); }
    catch { st.tag.classList.add("stale"); }               // нет связи — попробуем на следующем шаге
  }

  function schedule(el) {
    const st = live.get(el);
    if (!st || st.timer) return;
    st.timer = setInterval(() => tick(el), POLL_MS[st.tf] || 10000);
  }

  function stop(el) {
    const st = live.get(el);
    if (st && st.timer) { clearInterval(st.timer); st.timer = null; }
  }

  async function start(el) {
    if (live.has(el) || el.dataset.loading) return;
    el.dataset.loading = "1";
    try {
      const d = await fetchData(el.dataset.signal);
      if (!el.isConnected) return;
      live.set(el, create(el, d));
    } catch { /* остаётся картинка-заглушка; повторим, когда график снова попадёт в экран */ }
    finally { delete el.dataset.loading; }
  }

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const el = e.target;
      if (e.isIntersecting) {
        start(el).then(() => {
          const st = live.get(el);
          if (st) { st.visible = true; schedule(el); }
        });
      } else {
        const st = live.get(el);
        if (st) st.visible = false;
        stop(el);
      }
    }
  }, { rootMargin: "200px 0px" });

  function scan(root) {
    const els = root.matches?.(".live-chart") ? [root] : root.querySelectorAll?.(".live-chart") || [];
    for (const el of els) io.observe(el);
  }

  function dispose(root) {
    const els = root.matches?.(".live-chart") ? [root] : root.querySelectorAll?.(".live-chart") || [];
    for (const el of els) {
      io.unobserve(el);
      stop(el);
      const st = live.get(el);
      if (st) { st.chart.remove(); live.delete(el); }
    }
  }

  window.addEventListener("load", () => {
    if (!window.LightweightCharts) return;                 // библиотека не загрузилась — остаются картинки
    scan(document.body);
    document.body.addEventListener("htmx:load", (e) => scan(e.detail.elt));
    document.body.addEventListener("htmx:beforeCleanupElement", (e) => dispose(e.detail.elt));
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) for (const el of live.keys()) tick(el);   // вернулись во вкладку — сразу обновить
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      for (const st of live.values()) st.chart.applyOptions(theme());
    });
  });
})();
