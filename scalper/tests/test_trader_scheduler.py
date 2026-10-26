import asyncio
from datetime import datetime, timedelta, timezone

from trader import scheduler
from trader.domain.models import Timeframe


class FakeScanner:
    def __init__(self, stop: asyncio.Event, limit: int):
        self.calls: list[Timeframe] = []
        self.backs: list[int] = []
        self.stop, self.limit = stop, limit

    async def scan(self, tf, back=0):
        self.calls.append(tf)
        self.backs.append(back)
        if len(self.calls) >= self.limit:
            self.stop.set()


def _clock(times):
    it = iter(times)
    last = [None]

    def now():
        last[0] = next(it, last[0])
        return last[0]
    return now


def test_last_and_next_close():
    t = datetime(2026, 10, 5, 13, 7, tzinfo=timezone.utc)
    assert scheduler.last_close(t, Timeframe("4h")) == datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    assert scheduler.next_close(t, Timeframe("15m")) == datetime(2026, 10, 5, 13, 15, tzinfo=timezone.utc)
    edge = datetime(2026, 10, 5, 16, tzinfo=timezone.utc)
    assert scheduler.last_close(edge, Timeframe("4h")) == edge


def test_no_scan_at_start_then_scans_due_timeframes_bigger_first():
    t0 = datetime(2026, 10, 5, 11, 59, 50, tzinfo=timezone.utc)
    stop = asyncio.Event()
    sc = FakeScanner(stop, limit=2)
    clock = _clock([t0, t0, t0 + timedelta(seconds=40)])
    asyncio.run(asyncio.wait_for(scheduler.run(sc, 20, stop, poll_s=0.01, clock=clock), 5))
    assert [tf.value for tf in sc.calls] == ["4h", "15m"]


def test_wake_from_sleep_catches_up_once_per_timeframe():
    t0 = datetime(2026, 10, 5, 1, 0, 30, tzinfo=timezone.utc)
    wake = datetime(2026, 10, 5, 9, 10, tzinfo=timezone.utc)       # проспали 8 часов: две свечи 4h и много 15m
    stop = asyncio.Event()
    sc = FakeScanner(stop, limit=2)
    asyncio.run(asyncio.wait_for(scheduler.run(sc, 20, stop, poll_s=0.01, clock=_clock([t0, t0, wake])), 5))
    assert [tf.value for tf in sc.calls] == ["4h", "15m"]
    assert sc.backs == [1, 12]                    # пропущенные свечи тоже проверяются, не больше CATCHUP_BARS


class _Kv(dict):
    def kv_get(self, key):
        return self.get(key)

    def kv_set(self, key, value):
        self[key] = value


def test_restart_catches_up_from_saved_state_and_saves_progress():
    kv = _Kv({"scan_done:4h": "2026-10-05T04:00:00+00:00"})        # до перезапуска проверена свеча, закрытая в 04:00
    t0 = datetime(2026, 10, 5, 12, 0, 30, tzinfo=timezone.utc)      # сейчас закрыта 12:00: пропущены 08:00 и 12:00
    stop = asyncio.Event()
    sc = FakeScanner(stop, limit=1)
    asyncio.run(asyncio.wait_for(scheduler.run(sc, 20, stop, poll_s=0.01, clock=_clock([t0]), state=kv), 5))
    assert [tf.value for tf in sc.calls] == ["4h"] and sc.backs == [1]
    assert kv["scan_done:4h"] == "2026-10-05T12:00:00+00:00"


def test_tick_runs_background_work_and_survives_errors():
    t0 = datetime(2026, 10, 5, 11, 59, 50, tzinfo=timezone.utc)
    stop = asyncio.Event()
    sc = FakeScanner(stop, limit=2)
    ticks = []

    async def tick():
        ticks.append(1)
        raise RuntimeError("биржа недоступна")

    clock = _clock([t0, t0, t0 + timedelta(seconds=40)])
    asyncio.run(asyncio.wait_for(scheduler.run(sc, 20, stop, poll_s=0.01, clock=clock, tick=tick, tick_s=30), 5))
    assert len(ticks) == 2 and len(sc.calls) == 2
