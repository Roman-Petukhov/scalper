// Вкладка «Скринер»: сортировка таблицы кандидатов кнопками и график выручки монеты по кнопке «График выручки»
// (комиссии, выручка протокола и выручка держателям по месяцам за всю историю, DefiLlama).
(() => {
  "use strict";
  const KEY_SORT = "screener-sort";
  // направление «лучше — выше» по умолчанию: true — по убыванию
  const DESC = { score: true, hr_year: true, rev_year: true, hr_12m: true, mcap: true, fdv_to_hr: false,
                 mcap_to_fees: false, hr_growth: true, fees_mom: true, buyback_cover: true, chg_1y: true, since_seen: true };
  const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь",
                  "ноябрь", "декабрь"];
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const money = (v) => {
    const a = Math.abs(v);
    for (const [k, s] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]]) if (a >= k) return "$" + (v / k).toFixed(1) + s;
    return "$" + v.toFixed(0);
  };
  const store = {
    get(k, dflt) { try { const v = localStorage.getItem(k); return v === null ? dflt : JSON.parse(v); } catch (_) { return dflt; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* приватный режим */ } },
  };
  const charts = new Map();        // id монеты → ответ сервера, на время жизни страницы

  // --- сортировка ---

  function values(row) {
    try { return JSON.parse(row.dataset.k || "{}"); } catch (_) { return {}; }
  }

  function sortTable(root, key, desc) {
    const table = root.querySelector(".scr-table[data-sortable]");
    if (!table) return;
    closeAll(table);
    const rows = [...table.querySelectorAll(":scope > .scr-row[data-k]")];
    rows.sort((a, b) => {
      const x = values(a)[key], y = values(b)[key];
      const nx = x === null || x === undefined, ny = y === null || y === undefined;
      if (nx !== ny) return nx ? 1 : -1;                      // без значения — в конце при любом направлении
      if (!nx && x !== y) return desc ? y - x : x - y;
      return Number(a.dataset.rank) - Number(b.dataset.rank);
    });
    for (const r of rows) table.appendChild(r);
    for (const b of root.querySelectorAll("[data-sort]")) {
      const on = b.dataset.sort === key;
      b.setAttribute("aria-pressed", String(on));
      const dir = b.querySelector(".scr-dir");
      if (dir) dir.textContent = on ? (desc ? " ↓" : " ↑") : "";
    }
  }

  function initSort(scope) {
    for (const bar of (scope || document).querySelectorAll("[data-scr-sort]")) {
      const root = bar.closest(".scr-list");
      const saved = store.get(KEY_SORT, null);
      if (saved && saved.key in DESC && root.querySelector(`[data-sort="${saved.key}"]`)) sortTable(root, saved.key, !!saved.desc);
      else sortTable(root, "score", true);
    }
  }

  // --- график выручки ---

  function theme() {
    return {
      autoSize: true,
      layout: { background: { color: css("--surface") }, textColor: css("--muted"), fontSize: 11,
                fontFamily: 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace' },
      grid: { vertLines: { visible: false }, horzLines: { color: css("--line") + "55" } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.1, bottom: 0 } },
      timeScale: { borderVisible: false, rightOffset: 1 },
      localization: { locale: "ru-RU", priceFormatter: money },
      crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
      handleScroll: false, handleScale: false,
    };
  }

  function periodTitle(iso, unit) {
    const [y, m, d] = iso.split("-");
    return unit === "month" ? `${MONTHS[Number(m) - 1]} ${y}` : `неделя с ${d}.${m}.${y.slice(2)}`;
  }

  function closeAll(table) {
    for (const p of table.querySelectorAll(".scr-chart")) close(p);
  }

  function close(panel) {
    if (panel._chart) panel._chart.remove();
    if (panel._ro) panel._ro.disconnect();
    const btn = document.querySelector(`[data-rev="${CSS.escape(panel.dataset.for)}"][aria-expanded="true"]`);
    if (btn) { btn.setAttribute("aria-expanded", "false"); btn.textContent = "График выручки"; }
    panel.remove();
  }

  function draw(panel, data) {
    const legend = panel.querySelector(".scr-chart-legend");
    const box = panel.querySelector(".scr-chart-box");
    const note = panel.querySelector(".scr-chart-note");
    if (!data.periods || !data.periods.length) {
      box.remove();
      legend.textContent = "";
      note.textContent = data.note ? `Графика пока нет: ${data.note}.` : "DefiLlama не отдаёт историю выручки этого протокола.";
      return;
    }
    const chart = LightweightCharts.createChart(box, theme());
    panel._chart = chart;
    const pts = (vals, color) => data.periods.flatMap((t, i) => vals[i] === null || vals[i] === undefined ? []
      : [color ? { time: t, value: vals[i], color: color(i) } : { time: t, value: vals[i] }]);
    const holdersColor = css("--long"), last = data.periods.length - 1;
    const fmt = { type: "custom", formatter: money, minMove: 0.01 };
    const holders = chart.addSeries(LightweightCharts.HistogramSeries, { priceFormat: fmt, priceLineVisible: false,
      lastValueVisible: false, title: "" });
    holders.setData(pts(data.holders, (i) => (data.partial_last && i === last ? holdersColor + "66" : holdersColor + "cc")));
    const revenue = chart.addSeries(LightweightCharts.LineSeries, { color: css("--accent"), lineWidth: 2, priceFormat: fmt,
      priceLineVisible: false, lastValueVisible: false });
    revenue.setData(pts(data.revenue));
    const fees = chart.addSeries(LightweightCharts.LineSeries, { color: css("--muted"), lineWidth: 1, priceFormat: fmt,
      lineStyle: LightweightCharts.LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false });
    fees.setData(pts(data.fees));
    chart.timeScale().fitContent();
    panel._ro = new ResizeObserver(() => { if (panel._chart === chart) chart.timeScale().fitContent(); });
    panel._ro.observe(box);

    const series = [["держателям", holders, data.holders], ["протокол", revenue, data.revenue], ["комиссии", fees, data.fees]];
    const paint = (param) => {
      const i = param && param.time ? data.periods.indexOf(param.time) : last;
      if (i < 0) return;
      const parts = series.map(([t, , vals]) => vals[i] === null || vals[i] === undefined ? null : `${t} ${money(vals[i])}`)
        .filter(Boolean);
      const partial = data.partial_last && i === last ? " (ещё идёт)" : "";
      legend.textContent = [periodTitle(data.periods[i], data.unit) + partial, ...parts].join(" · ");
    };
    chart.subscribeCrosshairMove(paint);
    paint(null);
    note.textContent = `${data.unit === "month" ? "Сумма за месяц" : "Сумма за неделю (история короче 9 месяцев)"}, DefiLlama. ` +
      "Столбцы — выручка держателям (выкуп, сжигание, раздача), линия — вся выручка протокола, пунктир — комиссии" +
      (data.partial_last ? "; бледный последний столбец — период ещё не закончился." : ".");
  }

  async function open(btn) {
    const id = btn.dataset.rev;
    const row = btn.closest(".scr-row");
    const table = row.parentElement;
    const was = table.querySelector(`.scr-chart[data-for="${CSS.escape(id)}"]`);
    closeAll(table);
    if (was) return;                                          // повторный клик — закрыть
    const panel = document.createElement("div");
    panel.className = "scr-chart";
    panel.dataset.for = id;
    panel.id = `rev-${id}`;
    panel.setAttribute("role", "region");
    panel.setAttribute("aria-label", "Выручка монеты по периодам");
    panel.innerHTML = '<div class="scr-chart-key num"><span class="k-h">держателям</span><span class="k-r">протокол</span>' +
      '<span class="k-f">комиссии</span></div><p class="scr-chart-legend num" aria-live="polite"></p>' +
      '<div class="scr-chart-box"></div><p class="hint scr-chart-note">Загружаю историю выручки…</p>';
    row.after(panel);
    btn.setAttribute("aria-expanded", "true");
    btn.setAttribute("aria-controls", panel.id);
    btn.textContent = "Скрыть график";
    try {
      let data = charts.get(id);
      if (!data) {
        const r = await fetch(`/api/screener/revenue/${encodeURIComponent(id)}`, { credentials: "same-origin" });
        if (r.status === 401) { location.href = "/login"; return; }
        data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || data.detail || `ошибка ${r.status}`);
        charts.set(id, data);
      }
      if (panel.isConnected) draw(panel, data);
    } catch (err) {
      if (!panel.isConnected) return;
      panel.querySelector(".scr-chart-box").remove();
      panel.querySelector(".scr-chart-note").textContent = `График не загрузился: ${err.message}.`;
    }
  }

  document.addEventListener("click", (e) => {
    const s = e.target.closest("[data-scr-sort] [data-sort]");
    if (s) {
      const root = s.closest(".scr-list");
      const key = s.dataset.sort;
      const on = s.getAttribute("aria-pressed") === "true";
      const cur = (store.get(KEY_SORT, {}) || {});
      const desc = on && cur.key === key ? !cur.desc : DESC[key];
      store.set(KEY_SORT, { key, desc });
      sortTable(root, key, desc);
      return;
    }
    const b = e.target.closest("[data-rev]");
    if (b && window.LightweightCharts) open(b);
  });
  document.addEventListener("DOMContentLoaded", () => initSort(document));
  document.addEventListener("htmx:load", (e) => initSort(e.detail.elt));
})();
