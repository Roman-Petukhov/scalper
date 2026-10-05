// Сервис-воркер нужен, чтобы браузер предлагал «Установить приложение». Ничего не кеширует: панель всегда
// показывает свежие сигналы, а страницы с данными не остаются на диске.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  if (e.request.mode !== "navigate") return;
  e.respondWith(fetch(e.request).catch(() => new Response(
    "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>" +
    "<body style='font:16px system-ui;background:#0d1014;color:#e6ebf2;display:grid;place-items:center;height:100vh;margin:0'>" +
    "<p style='max-width:320px;text-align:center'>Панель не отвечает. Запустите run_panel на компьютере " +
    "и обновите страницу.</p>", {headers: {"Content-Type": "text/html; charset=utf-8"}})));
});
