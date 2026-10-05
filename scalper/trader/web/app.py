"""Веб-панель: вход по паролю, кнопки таймфреймов, режим «ручная / авто», параметры риска, лента сигналов
с графиками и решениями. HTMX: каждое действие возвращает обновлённый фрагмент страницы."""
from __future__ import annotations

import asyncio
import hmac
import logging
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .. import scheduler
from ..application.execution import Executor
from ..application.live_chart import chart_payload
from ..application.ports import Broker, MarketData, Notifier
from ..application.services import Scanner, SettingsService, SignalDecisions
from ..config import AppConfig
from ..domain.execution import TradeStatus
from ..domain.models import EntryPolicy, Mode, SignalStatus, Timeframe
from ..infrastructure.binance_data import BinanceMarketData
from ..infrastructure.bybit import BrokerHolder, BybitBroker, BybitCredentials
from ..infrastructure.charts import MatplotlibCharts
from ..infrastructure.sqlite_repo import SqliteStore
from ..infrastructure.telegram import TelegramNotifier
from ..infrastructure.webpush import MultiNotifier, WebPushNotifier

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")
REMEMBER_S = 365 * 24 * 3600          # «запомнить на этом устройстве»
SHORT_LOGIN_S = 12 * 3600
CHART_BARS = 300                      # свечей на живом графике
STATUS_LABEL = {SignalStatus.NEW: "новый", SignalStatus.TAKEN: "в работе", SignalStatus.SKIPPED: "пропущен",
                SignalStatus.EXPIRED: "истёк"}


def chart_name(path: str) -> str:
    """Имя файла графика из сохранённого пути: на Windows путь с «\\», поэтому режем по обоим разделителям."""
    return re.split(r"[\\/]", path)[-1]


templates.env.filters["chart_name"] = chart_name
templates.env.globals.update(TRADE_LABEL={TradeStatus.PLACED: "лимитка ждёт", TradeStatus.FILLED: "на бирже",
                                          TradeStatus.EXPIRED: "лимитка снята", TradeStatus.CANCELLED: "снят / закрыт"},
                             FEED_TABS=[("all", "Все")] + [(t.value, t.value) for t in Timeframe],
                             STATUS_LABEL=STATUS_LABEL, Timeframe=Timeframe, Mode=Mode, EntryPolicy=EntryPolicy,
                             SignalStatus=SignalStatus)


def create_app(cfg: AppConfig, market: MarketData | None = None, notifier: Notifier | None = None,
               broker: Callable[[], Broker | None] | None = None) -> FastAPI:
    store = SqliteStore(cfg.data_dir / "trader.db")
    charts = MatplotlibCharts(cfg.data_dir / "charts")
    market = market or BinanceMarketData()
    push = WebPushNotifier(store, cfg.data_dir / "vapid_private.pem")
    telegram = (TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)
                if cfg.telegram_token and cfg.telegram_chat_id else None)
    notify = MultiNotifier(push, telegram, notifier)
    holder = BrokerHolder(store)
    executor = Executor(broker or holder, store, store, store, notify)
    scanner = Scanner(market, store, store, charts, notify, cfg.panel_url, executor=executor)
    settings_svc = SettingsService(store)
    decisions = SignalDecisions(store, store, executor)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = (asyncio.create_task(scheduler.run(scanner, cfg.scan_delay_s, stop, tick=executor.housekeep))
                if cfg.scheduler else None)
        yield
        stop.set()
        if task is not None:
            await task
        await holder.close()

    app = FastAPI(title="Trendline Trader", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SessionMiddleware, secret_key=cfg.session_secret, session_cookie="tt_session",
                       max_age=REMEMBER_S, same_site="strict", https_only=cfg.panel_url.startswith("https"))
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.state.scanner = scanner

    def authed(request: Request) -> bool:
        if request.session.get("ok") is not True:
            return False
        until = request.session.get("until")                       # вход без «запомнить» живёт 12 часов
        return until is None or time.time() < until

    def guard(request: Request, mutate: bool = False) -> None:
        if not authed(request):
            raise HTTPException(401, "нужен вход")
        if mutate and request.headers.get("HX-Request") != "true":
            raise HTTPException(403, "запрос не из панели")      # защита от подделки межсайтовых форм

    def feed_view(request: Request) -> Timeframe | None:
        """Какой таймфрейм смотрим в ленте (вкладка «Все» = None); запоминается в сессии."""
        v = request.session.get("view")
        return Timeframe(v) if v in {t.value for t in Timeframe} else None

    def page_context(request: Request) -> dict:
        s = settings_svc.get()
        view = feed_view(request)
        shown = set(s.timeframes) & ({view} if view else set(Timeframe))
        signals = store.recent(80, shown)
        return {"s": s, "signals": signals, "view": view, "last": scanner.last,
                "trades": store.trades_for([x.id for x in signals if x.id is not None]),
                "exchange": exchange_label(), "creds": holder.credentials if broker is None else None}

    def exchange_label() -> str | None:
        b = executor.broker()
        return None if b is None else ("Bybit демо" if b.network == "demo" else "Bybit реальный счёт")

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
    async def login(request: Request, password: str = Form(...), remember: str | None = Form(None)):
        if hmac.compare_digest(password.encode(), cfg.panel_password.encode()):
            request.session.clear()
            request.session["ok"] = True
            if not remember:
                request.session["until"] = time.time() + SHORT_LOGIN_S
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
    async def feed(request: Request, view: str | None = None):
        guard(request)
        if view is not None:
            request.session["view"] = view if view in {t.value for t in Timeframe} else "all"
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
        return templates.TemplateResponse(request, "_controls.html", page_context(request),
                                          headers={"HX-Trigger": "feed-refresh"})

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
    async def set_params(request: Request, risk_pct: float = Form(...), leverage: int = Form(5),
                         max_positions: int = Form(...),
                         daily_loss_pct: float = Form(...), min_aggr_pct: float = Form(...),
                         target_r: float = Form(...), hybrid_range_atr: float = Form(...),
                         min_break_atr: float = Form(0.0), min_close_loc_pct: float = Form(0.0)):
        guard(request, mutate=True)
        ctx_error = None
        try:
            settings_svc.update(risk_pct=risk_pct, leverage=leverage, max_positions=max_positions, daily_loss_pct=daily_loss_pct,
                                min_aggr=min_aggr_pct / 100, target_r=target_r, hybrid_range_atr=hybrid_range_atr,
                                min_break_atr=min_break_atr, min_close_loc=min_close_loc_pct / 100)
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
            sig = await decisions.take(signal_id) if action == "take" else decisions.skip(signal_id)
            err = None
        except LookupError:
            raise HTTPException(404, "сигнал не найден")
        except ValueError as e:
            sig, err = store.get(signal_id), str(e)
        return templates.TemplateResponse(request, "_signal.html", {
            "sig": sig, "s": settings_svc.get(), "error": err, "trades": store.trades_for([signal_id]),
            "exchange": exchange_label()}, headers={"HX-Trigger": "wallet-refresh"} if err is None else None)

    @app.post("/scan/{tf}", response_class=HTMLResponse)
    async def scan_now(request: Request, tf: Timeframe):
        guard(request, mutate=True)
        rep = await scanner.scan(tf)
        disabled = tf not in settings_svc.get().timeframes
        return templates.TemplateResponse(request, "_scan_result.html", {"tf": tf, "rep": rep, "disabled": disabled},
                                          headers={"HX-Trigger": "feed-refresh"})

    @app.get("/api/chart/{signal_id}")
    async def live_chart(request: Request, signal_id: int) -> dict:
        """Живой график сигнала: свечи с текущей, линия, уровни; панель опрашивает раз в несколько секунд."""
        guard(request)
        sig = store.get(signal_id)
        if sig is None:
            raise HTTPException(404, "сигнал не найден")
        try:
            bars = await market.live_bars(sig.symbol, sig.timeframe, CHART_BARS)
        except Exception as e:
            log.warning("живой график %s: %s", sig.symbol, e)
            raise HTTPException(502, "биржа не отдала свечи")
        return chart_payload(sig, bars, store.trades_for([signal_id]).get(signal_id))

    @app.get("/wallet", response_class=HTMLResponse)
    async def wallet(request: Request):
        guard(request)
        err = None
        try:
            w = await executor.wallet()
        except Exception as e:
            log.warning("кошелёк: %s", e)
            w, err = None, f"Bybit не ответил: {str(e)[:140]}"
        return templates.TemplateResponse(request, "_wallet.html", {"w": w, "error": err,
                                                                    "exchange": exchange_label()})

    @app.post("/exchange/keys", response_class=HTMLResponse)
    async def exchange_keys(request: Request, api_key: str = Form(...), secret: str = Form(...),
                            network: str = Form("demo")):
        guard(request, mutate=True)
        ctx = page_context(request)
        try:
            creds = BybitCredentials(api_key.strip(), secret.strip(), network)
        except ValueError as e:
            return templates.TemplateResponse(request, "_controls.html", ctx | {"keys_error": str(e)})
        probe = BybitBroker(creds)
        try:
            acc = await probe.account()                           # проверяем ключи до сохранения
        except Exception as e:
            log.warning("проверка ключей Bybit: %s", e)
            return templates.TemplateResponse(request, "_controls.html", ctx | {
                "keys_error": f"Bybit не принял ключи: {str(e)[:160]}"})
        finally:
            await probe.close()
        holder.save(creds)
        return templates.TemplateResponse(request, "_controls.html", page_context(request) | {
            "keys_saved": f"Подключено · капитал {acc.equity:,.2f} USDT".replace(",", " ")},
            headers={"HX-Trigger": "wallet-refresh"})

    @app.post("/exchange/forget", response_class=HTMLResponse)
    async def exchange_forget(request: Request):
        guard(request, mutate=True)
        holder.forget()
        return templates.TemplateResponse(request, "_controls.html", page_context(request),
                                          headers={"HX-Trigger": "wallet-refresh"})

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
