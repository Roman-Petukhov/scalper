import json

import numpy as np
import pandas as pd

from research import social as S


def test_tagging_tickers_names_and_ambiguous_words():
    pats = S.sector_patterns()
    assert S.tag("PEPE and $WIF pumping", pats) == ["мемы"]
    assert set(S.tag("Arbitrum vs Solana: which layer 2 wins?", pats)) == {"L2", "L1"}
    assert S.tag("I have ONE question about AI and NOT much else", pats) == []
    assert "AI" in S.tag("Best AI agents tokens this cycle", pats)
    assert S.tag("pepe frog", pats) == ["мемы"] and S.tag("pepetual motion", pats) == []


def test_daily_counts_and_share_score():
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    items = [(t0 + pd.Timedelta(days=d, hours=1), "Bitcoin news") for d in range(120)]
    items += [(t0 + pd.Timedelta(days=d, hours=2), "DOGE to the moon") for d in range(120) if d % 4 == 0 or d >= 113]
    c = S.daily_counts(items, "reddit:x")
    assert set(c.columns) == {"day", "source", "sector", "n"}
    assert int(c[c.sector == "__all__"].n.sum()) == len(items)
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    sc = S.share_score(c, ["мемы", "AI"], idx)
    assert sc["мемы"].iloc[-1] > 0.5 and sc["AI"].isna().all()


def test_parsers():
    raw = json.dumps({"data": [{"id": "a", "created_utc": 1700000000, "title": "x"}, "junk"]}).encode()
    assert S.parse_arctic(raw) == [{"id": "a", "created_utc": 1700000000, "title": "x"}]
    assert S.parse_arctic(b"oops") == [] and S.parse_arctic(None) == []
    page = ('<div class="tgme_widget_message" data-post="chan/41"><div class="tgme_widget_message_text js-x" dir="auto">'
            'SOL &amp; <b>JUP</b> rally</div><time datetime="2024-03-01T10:00:00+00:00"></time></div>'
            '<div class="tgme_widget_message" data-post="chan/42"><time datetime="2024-03-02T10:00:00+00:00"></time></div>')
    got = S.parse_tme(page)
    assert [m for m, _, _ in got] == [41, 42] and "SOL &" in got[0][2] and got[1][2] == ""
    assert got[0][1] == pd.Timestamp("2024-03-01T10:00:00Z")


def test_lead_lag_detects_leading_score():
    idx = pd.date_range("2022-01-02", periods=400, freq="D")
    rng = np.random.default_rng(0)
    turn = pd.DataFrame(np.cumsum(rng.normal(0, 0.1, (400, 3)), axis=0), index=idx, columns=list("abc"))
    score = turn.shift(-7) - turn                                       # балл знает будущий рост оборота
    nxt, prv = S.lead_lag(score, turn)
    assert nxt > 0.9 and abs(prv) < 0.3
