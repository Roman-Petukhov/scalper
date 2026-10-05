// Уведомления приложения: раз в 30 с спрашиваем у сервера новые сигналы и показываем системное уведомление
// (Windows / macOS / Android), звук и счётчик на иконке приложения. Работает, пока окно панели открыто
// (можно свёрнутым). Что уже показано — помним в localStorage этого устройства.
(() => {
  const KEY_LAST = "tt.lastSignalId", KEY_SOUND = "tt.sound";
  const POLL_MS = 30000;
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* приватный режим — живём без памяти */ } },
  };
  let unseen = 0;

  const supported = () => "Notification" in window && "serviceWorker" in navigator;
  const soundOn = () => store.get(KEY_SOUND, "1") === "1";

  function beep() {
    if (!soundOn()) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const t = ctx.currentTime;
      [[880, 0], [1320, 0.14]].forEach(([f, dt]) => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.type = "sine"; o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, t + dt);
        g.gain.exponentialRampToValueAtTime(0.18, t + dt + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t + dt + 0.22);
        o.connect(g).connect(ctx.destination); o.start(t + dt); o.stop(t + dt + 0.25);
      });
      setTimeout(() => ctx.close(), 800);
    } catch { /* без звука */ }
  }

  function setBadge(n) {
    if ("setAppBadge" in navigator) (n > 0 ? navigator.setAppBadge(n) : navigator.clearAppBadge()).catch(() => {});
    document.title = (n > 0 ? `(${n}) ` : "") + "Трендовые пробои";
  }

  async function show(sig) {
    const title = `${sig.symbol} · ${sig.timeframe} · ${sig.side}`;
    const fmt = (x) => Number(x).toPrecision(6).replace(/\.?0+$/, "");
    const body = `Вход (${sig.entry_kind}) ${fmt(sig.entry)} · стоп ${fmt(sig.stop)} · цель ${fmt(sig.target)}\n` +
                 `Агрессоры ${Math.round(sig.aggr * 100)}%`;
    const reg = await navigator.serviceWorker.ready;
    await reg.showNotification(title, {
      body, tag: `signal-${sig.id}`, icon: "/static/icons/icon-192.png", badge: "/static/icons/maskable-192.png",
      data: { url: `/#signal-${sig.id}` }, requireInteraction: true,
    });
  }

  async function poll() {
    const last = Number(store.get(KEY_LAST, "0"));
    let res;
    try {
      const r = await fetch(`/api/signals/new?after=${last}`, { credentials: "same-origin", cache: "no-store" });
      if (r.status === 401) { location.href = "/login"; return; }
      if (!r.ok) return;
      res = await r.json();
    } catch { return; }                                    // сервер не отвечает — попробуем позже
    if (last === 0) { store.set(KEY_LAST, String(res.last_id)); return; }   // первый запуск: старое не показываем
    if (!res.signals.length) return;
    store.set(KEY_LAST, String(Math.max(last, ...res.signals.map((x) => x.id))));
    if (window.htmx) htmx.trigger(document.body, "feed-refresh");
    beep();
    if (document.hidden) { unseen += res.signals.length; setBadge(unseen); }
    if (supported() && Notification.permission === "granted") for (const sig of res.signals) await show(sig);
  }

  function renderState() {
    const el = document.getElementById("notify-state");
    const btn = document.getElementById("notify-enable");
    const snd = document.getElementById("notify-sound");
    if (snd) snd.setAttribute("aria-pressed", soundOn() ? "true" : "false");
    if (!el || !btn) return;
    if (!supported()) { el.textContent = "Браузер не поддерживает уведомления."; btn.hidden = true; return; }
    const p = Notification.permission;
    el.textContent = p === "granted" ? "Включены: придут, пока окно панели открыто (можно свёрнутым)."
      : p === "denied" ? "Заблокированы в браузере: разрешите их в настройках сайта (значок замка в адресной строке)."
      : "Выключены.";
    btn.hidden = p !== "default";
  }

  document.addEventListener("click", async (e) => {
    const t = e.target.closest("#notify-enable, #notify-sound, #notify-test");
    if (!t) return;
    if (t.id === "notify-enable") { await Notification.requestPermission(); renderState(); }
    if (t.id === "notify-sound") { store.set(KEY_SOUND, soundOn() ? "0" : "1"); renderState(); }
    if (t.id === "notify-test") {
      beep();
      if (supported() && Notification.permission === "granted") {
        const reg = await navigator.serviceWorker.ready;
        reg.showNotification("Трендовые пробои", { body: "Так будет выглядеть уведомление о сигнале.",
          icon: "/static/icons/icon-192.png", tag: "test" });
      }
    }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { unseen = 0; setBadge(0); } });
  document.addEventListener("htmx:afterSwap", renderState);
  window.addEventListener("load", () => { renderState(); poll(); setInterval(poll, POLL_MS); });
})();
