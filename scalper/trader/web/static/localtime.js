// Время в часовом поясе браузера. Сервер отдаёт <time class="lt" datetime="ISO UTC" data-f="…">…UTC</time>;
// здесь текст заменяется на местное время с подписью пояса (UTC+3 и т. п.). Без JS остаётся UTC.
(() => {
  "use strict";
  const pad = (n) => String(n).padStart(2, "0");
  const zone = (d) => {
    const m = -d.getTimezoneOffset();
    if (m === 0) return "UTC";
    const h = Math.trunc(Math.abs(m) / 60), r = Math.abs(m) % 60;
    return `UTC${m > 0 ? "+" : "−"}${h}${r ? ":" + pad(r) : ""}`;
  };
  const FORMATS = {
    dmhm: (d) => `${pad(d.getDate())}.${pad(d.getMonth() + 1)} ${pad(d.getHours())}:${pad(d.getMinutes())}`,
    dm: (d) => `${pad(d.getDate())}.${pad(d.getMonth() + 1)}`,
    hm: (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`,
    hms: (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`,
  };
  const format = (d, f) => (FORMATS[f] || FORMATS.dmhm)(d);

  function localize(root) {
    (root || document).querySelectorAll("time.lt[datetime]").forEach((el) => {
      const d = new Date(el.getAttribute("datetime"));
      if (Number.isNaN(d.getTime())) return;
      const f = el.dataset.f || "dmhm";
      el.textContent = f === "dm" ? format(d, f) : `${format(d, f)} ${zone(d)}`;
      el.title = `${d.toISOString().slice(0, 16).replace("T", " ")} UTC`;
    });
  }

  window.LocalTime = { format, zone, localize };
  document.addEventListener("DOMContentLoaded", () => localize(document));
  document.addEventListener("htmx:afterSettle", (e) => localize(e.target));
})();
