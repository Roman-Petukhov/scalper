// Разделы панели (Сигналы / Счёт / Настройки): один виден, выбор — в адресе (#account) и в памяти браузера.
// Без JS видны все разделы подряд.
(() => {
  const VIEWS = ["signals", "account", "settings"];
  const KEY = "panel-view";

  function read() {
    const h = location.hash.slice(1);
    if (VIEWS.includes(h)) return h;
    try { const v = localStorage.getItem(KEY); if (VIEWS.includes(v)) return v; } catch (_) { /* приватный режим */ }
    return "signals";
  }

  function show(view, scroll) {
    document.body.dataset.view = view;
    for (const a of document.querySelectorAll(".view-tab")) {
      const on = a.dataset.go === view;
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    }
    try { localStorage.setItem(KEY, view); } catch (_) { /* приватный режим */ }
    if (location.hash.slice(1) !== view) history.replaceState(null, "", "#" + view);
    if (scroll) window.scrollTo({ top: 0, behavior: "instant" });
    window.dispatchEvent(new Event("resize"));             // графики в только что показанном разделе
  }

  document.addEventListener("click", (e) => {
    const a = e.target.closest("[data-go]");
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey) return;
    e.preventDefault();
    show(a.dataset.go, true);
  });
  window.addEventListener("hashchange", () => show(read(), true));
  document.addEventListener("DOMContentLoaded", () => show(read(), true));
})();
