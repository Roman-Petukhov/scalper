"""Веб-панель: вход по паролю, кнопки таймфреймов, режим «ручная / авто», параметры риска, лента сигналов
с графиками и решениями. HTMX: каждое действие возвращает обновлённый фрагмент страницы."""
from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .. import scheduler
from ..application.ports import MarketData, Notifier
from ..application.services import Scanner, SettingsService, SignalDecisions
from ..config import AppConfig
from ..domain.models import EntryPolicy, Mode, SignalStatus, Timeframe
from ..infrastructure.binance_data import BinanceMarketData
from ..infrastructure.charts import MatplotlibCharts
from ..infrastructure.sqlite_repo import SqliteStore
from ..infrastructure.telegram import TelegramNotifier
from ..infrastructure.webpush import MultiNotifier, WebPushNotifier

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")
STATUS_LABEL = {SignalStatus.NEW: "новый", SignalStatus.TAKEN: "в работе", SignalStatus.SKIPPED: "пропущен",
                SignalStatus.EXPIRED: "истёк"}
templates.env.globals.update(STATUS_LABEL=STATUS_LABEL, Timeframe=Timeframe, Mode=Mode, EntryPolicy=EntryPolicy,
                             SignalStatus=SignalStatus)


def create_app(cfg: AppConfig, market: MarketData | None = None, notifier: Notifier | None = None) -> FastAPI:
    store = SqliteStore(cfg.data_dir / "trader.db")
    charts = MatplotlibCharts(cfg.data_dir / "charts")
    market = market or BinanceMarketData()
    push = WebPushNotifier(store, cfg.data_dir / "vapid_private.pem")
    telegram = (TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)
                if cfg.telegram_token and cfg.telegram_chat_id else None)
    scanner = Scanner(market, store, store, charts, MultiNotifier(push, telegram, notifier), cfg.panel_url)
    settings_svc = SettingsService(store)
    decisions = SignalDecisions(store, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = asyncio.create_task(scheduler.run(scanner, cfg.scan_delay_s, stop)) if cfg.scheduler else None
        yield
        stop.set()
        if task is not None:
            await task

    app = FastAPI(title="Trendline Trader", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SessionMiddleware, secret_key=cfg.session_secret, session_cookie="tt_session",
                       max_age=14 * 24 * 3600, same_site="strict", https_only=cfg.panel_url.startswith("https"))
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.state.scanner = scanner

    def authed(request: Request) -> bool:
        return request.session.get("ok") is True

    def guard(request: Request, mutate: bool = False) -> None:
        if not authed(request):
            raise HTTPException(401, "нужен вход")
        if mutate and request.headers.get("HX-Request") != "true":
            raise HTTPException(403, "запрос не из панели")      # защита от подделки межсайтовых форм

    def page_context(request: Request) -> dict:
        s = settings_svc.get()
        return {"s": s, "signals": store.recent(80), "last": scanner.last}

    @app.get("/manifest.webmanifest")
    async def manifest():
        return FileResponse(HERE / "static" / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/sw.js")
    async def service_worker():
        return FileResponse(HERE / "static" / "sw.js", media_type="text/javascript",
                            headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})

    @app.get("/favicon.ico")
    async def favicon():
        return FileResponse(HERE / "static" / "icons" / "favicon-32.png", media_type="image/png")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request, password: str = Form(...)):
        if hmac.compare_digest(password.encode(), cfg.panel_password.encode()):
            request.session.clear()
            request.session["ok"] = True
            return RedirectResponse("/", status_code=303)
        await asyncio.sleep(1.0)                                   # замедление перебора
        return templates.TemplateResponse(request, "login.html", {"error": "Неверный пароль"}, status_code=401)

    @app.post("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if not authed(request):
            return RedirectResponse("/login", status_code=303)
        return templates.TemplateResponse(request, "index.html", page_context(request))

    @app.get("/feed", response_class=HTMLResponse)
    async def feed(request: Request):
        guard(request)
        return templates.TemplateResponse(request, "_feed.html", page_context(request))

    @app.get("/api/signals/new")
    async def new_signals(request: Request, after: int = 0) -> dict:
        """Сигналы новее `after` (по id) для включённых таймфреймов — для уведомлений приложения."""
        guard(request)
        s = settings_svc.get()
        items = [x for x in store.recent(50, set(s.timeframes)) if x.id is not None and x.id > after]
        last = max((x.id for x in store.recent(1) if x.id is not None), default=0)
        return {"last_id": last, "signals": [
            {"id": x.id, "symbol": x.symbol, "timeframe": x.timeframe.value, "side": x.side.label,
             "entry": x.plan.entry, "stop": x.plan.stop, "target": x.plan.target,
             "entry_kind": "ретест" if x.plan.entry_kind.value == "retest" else "рынок",
             "aggr": round(x.aggr, 3), "status": x.status.value} for x in sorted(items, key=lambda z: z.id)]}

    def guard_api(request: Request) -> None:
        guard(request)
        if request.headers.get("X-Panel") != "1":
            raise HTTPException(403, "запрос не из панели")

    @app.get("/api/push/key")
    async def push_key(request: Request) -> dict:
        guard(request)
        return {"key": push.public_key}

    @app.post("/api/push/subscribe")
    async def push_subscribe(request: Request, sub: dict = Body(...)) -> dict:
        guard_api(request)
        endpoint, keys = sub.get("endpoint"), sub.get("keys") or {}
        if not (isinstance(endpoint, str) and endpoint.startswith("https://") and keys.get("p256dh") and keys.get("auth")):
            raise HTTPException(422, "неверная подписка")
        store.add_subscription({"endpoint": endpoint, "keys": {"p256dh": keys["p256dh"], "auth": keys["auth"]}})
        return {"ok": True, "devices": len(store.subscriptions())}

    @app.post("/api/push/unsubscribe")
    async def push_unsubscribe(request: Request, sub: dict = Body(...)) -> dict:
        guard_api(request)
        if isinstance(sub.get("endpoint"), str):
            store.remove_subscription(sub["endpoint"])
        return {"ok": True}

    @app.post("/api/push/test")
    async def push_test(request: Request) -> dict:
        guard_api(request)
        sent = await push.send({"title": "Трендовые пробои", "body": "Push работает: так придёт сигнал.",
                                "tag": "test", "url": "/"})
        return {"sent": sent, "devices": len(store.subscriptions())}

    @app.post("/settings/tf/{tf}", response_class=HTMLResponse)
    async def toggle_tf(request: Request, tf: Timeframe):
        guard(request, mutate=True)
        settings_svc.toggle_timeframe(tf)
        return templates.TemplateResponse(request, "_controls.html", page_context(request))

    @app.post("/settings/mode/{mode}", response_class=HTMLResponse)
    async def set_mode(request: Request, mode: Mode):
        guard(request, mutate=True)
        settings_svc.set_mode(mode)
        return templates.TemplateResponse(request, "_controls.html", page_context(request))

    @app.post("/settings/entry/{policy}", response_class=HTMLResponse)
    async def set_entry(request: Request, policy: EntryPolicy):
        guard(request, mutate=True)
        settings_svc.set_entry_policy(policy)
        return templates.TemplateResponse(request, "_controls.html", page_context(request))

    @app.post("/settings/params", response_class=HTMLResponse)
    async def set_params(request: Request, risk_pct: float = Form(...), max_positions: int = Form(...),
                         daily_loss_pct: float = Form(...), min_aggr_pct: float = Form(...),
                         target_r: float = Form(...), hybrid_range_atr: float = Form(...)):
        guard(request, mutate=True)
        ctx_error = None
        try:
            settings_svc.update(risk_pct=risk_pct, max_positions=max_positions, daily_loss_pct=daily_loss_pct,
                                min_aggr=min_aggr_pct / 100, target_r=target_r, hybrid_range_atr=hybrid_range_atr)
        except ValueError as e:
            ctx_error = str(e)
        ctx = page_context(request) | {"params_error": ctx_error, "params_saved": ctx_error is None}
        return templates.TemplateResponse(request, "_controls.html", ctx)

    @app.post("/signals/{signal_id}/{action}", response_class=HTMLResponse)
    async def decide(request: Request, signal_id: int, action: str):
        guard(request, mutate=True)
        if action not in ("take", "skip"):
            raise HTTPException(404)
        try:
            sig = decisions.take(signal_id) if action == "take" else decisions.skip(signal_id)
            err = None
        except LookupError:
            raise HTTPException(404, "сигнал не найден")
        except ValueError as e:
            sig, err = store.get(signal_id), str(e)
        return templates.TemplateResponse(request, "_signal.html", {"sig": sig, "s": settings_svc.get(), "error": err})

    @app.post("/scan/{tf}", response_class=HTMLResponse)
    async def scan_now(request: Request, tf: Timeframe):
        guard(request, mutate=True)
        rep = await scanner.scan(tf)
        msg = (f"{tf.value}: выключен — включите кнопку" if tf not in settings_svc.get().timeframes else
               f"{tf.value}: монет {rep.symbols}, новых сигналов {len(rep.signals)}"
               + (f", ошибок {rep.errors}" if rep.errors else ""))
        return HTMLResponse(f'<span class="toast" role="status">{msg}</span>',
                            headers={"HX-Trigger": "feed-refresh"})

    @app.get("/charts/{name}")
    async def chart(request: Request, name: str):
        guard(request)
        base = (cfg.data_dir / "charts").resolve()
        path = (base / name).resolve()
        if path.parent != base or path.suffix != ".png" or not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.exception_handler(401)
    async def unauthorized(request: Request, exc: HTTPException) -> Response:
        if request.headers.get("HX-Request") == "true":
            return Response(status_code=401, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)

    return app


def main() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app(AppConfig.from_env())
