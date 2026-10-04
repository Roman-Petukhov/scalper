from research import shard


def test_mine_partitions_items_across_shards(monkeypatch):
    items = [f"C{i}USDT" for i in range(200)]
    monkeypatch.setenv("NSHARDS", "4")
    monkeypatch.delenv("SMOKE", raising=False)
    got = []
    for i in range(4):
        monkeypatch.setenv("SHARD", str(i))
        got += shard.mine(items)
    assert sorted(got) == sorted(items)


def test_smoke_takes_first_items_of_own_shard(monkeypatch):
    monkeypatch.setenv("NSHARDS", "2")
    monkeypatch.setenv("SHARD", "1")
    monkeypatch.setenv("SMOKE", "1")
    monkeypatch.setenv("SMOKE_N", "2")
    full = [x for x in (f"C{i}USDT" for i in range(50)) if shard.owner(x) == 1]
    assert shard.mine(f"C{i}USDT" for i in range(50)) == full[:2]


def test_prune_keeps_own_shared_and_btc(tmp_path, monkeypatch):
    monkeypatch.setenv("NSHARDS", "2")
    monkeypatch.setenv("SHARD", "0")
    names = [f"C{i}USDT-1h-2024-01.parquet" for i in range(20)] + ["BTCUSDT-1h-2024-01.parquet", "leaderboard.json"]
    (tmp_path / "hl").mkdir()
    for n in names:
        (tmp_path / n).write_text("x")
    (tmp_path / "hl" / "fills-0xabc.parquet").write_text("x")
    shard.prune(tmp_path)
    left = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert "BTCUSDT-1h-2024-01.parquet" in left and "leaderboard.json" in left
    for i in range(20):
        sym = f"C{i}USDT"
        assert (f"{sym}-1h-2024-01.parquet" in left) == (shard.owner(sym) == 0)
    assert ("fills-0xabc.parquet" in left) == (shard.owner("0xabc") == 0)
