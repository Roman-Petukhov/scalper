import asyncio
import base64

import pytest

pytest.importorskip("pywebpush")

from trader.infrastructure import webpush as wp
from trader.infrastructure.sqlite_repo import SqliteStore

from test_trader_app import _signal
from test_trader_web import HX, _login, env  # noqa: F401  (фикстура панели)

SUB = {"endpoint": "https://fcm.googleapis.com/fcm/send/abc", "keys": {"p256dh": "BPk", "auth": "xyz"}}


def test_vapid_keys_created_once_and_public_key_format(tmp_path):
    p, pub = wp.load_or_create_keys(tmp_path / "k.pem")
    raw = base64.urlsafe_b64decode(pub + "=" * (-len(pub) % 4))
    assert len(raw) == 65 and raw[0] == 4                     # несжатая точка P-256
    assert wp.load_or_create_keys(tmp_path / "k.pem")[1] == pub


def test_send_removes_gone_subscriptions(tmp_path, monkeypatch):
    st = SqliteStore(tmp_path / "t.db")
    st.add_subscription(SUB)
    st.add_subscription({**SUB, "endpoint": "https://web.push.apple.com/ok"})
    sent = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

    def fake(sub, data, **kw):
        if "fcm" in sub["endpoint"]:
            raise wp.WebPushException("gone", response=Resp(410))
        sent.append((sub["endpoint"], data, kw["vapid_claims"]["sub"]))

    monkeypatch.setattr(wp, "webpush", fake)
    n = wp.WebPushNotifier(st, tmp_path / "k.pem")
    sig = _signal()
    sig = sig.__class__(**{**sig.__dict__, "id": 5})
    assert asyncio.run(n.send(wp.payload(sig))) == 1
    assert [s["endpoint"] for s in st.subscriptions()] == ["https://web.push.apple.com/ok"]
    assert "SOLUSDT · 4h · лонг" in sent[0][1] and "#signal-5" in sent[0][1] and sent[0][2].startswith("mailto:")


def test_push_api_requires_panel_header_and_valid_subscription(env, monkeypatch):  # noqa: F811
    c, tmp = env
    _login(c)
    assert len(c.get("/api/push/key").json()["key"]) > 80
    assert c.post("/api/push/subscribe", json=SUB).status_code == 403
    hdr = {"X-Panel": "1"}
    assert c.post("/api/push/subscribe", json={"endpoint": "http://evil"}, headers=hdr).status_code == 422
    assert c.post("/api/push/subscribe", json=SUB, headers=hdr).json()["devices"] == 1
    monkeypatch.setattr(wp, "webpush", lambda sub, data, **kw: None)
    assert c.post("/api/push/test", headers=hdr).json() == {"sent": 1, "devices": 1}
    c.post("/api/push/unsubscribe", json={"endpoint": SUB["endpoint"]}, headers=hdr)
    assert SqliteStore(tmp / "trader.db").subscriptions() == []
