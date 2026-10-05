"""PNG-график сигнала: свечи, линия по двум точкам на закрытиях, вход, стоп, цель."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..domain.models import EntryKind, Signal  # noqa: E402

UP, DOWN, LINE_LONG, LINE_SHORT = "#26a69a", "#ef5350", "#1565c0", "#ef6c00"
BG, FG, GRID = "#0f1115", "#c9d1d9", "#232833"


class MatplotlibCharts:
    def __init__(self, out_dir: str | Path, bars: int = 160) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bars = bars

    def render(self, signal: Signal, bars: pd.DataFrame) -> str:
        (t1, v1), (t2, v2) = signal.line_points
        start = min(bars.index.get_indexer([pd.Timestamp(t1)], method="nearest")[0], len(bars) - self.bars)
        d = bars.iloc[max(0, start - 10):]
        x = np.arange(len(d))
        pos = {ts: i for i, ts in enumerate(d.index)}
        o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=110)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        for i in x:
            col = UP if c[i] >= o[i] else DOWN
            ax.vlines(i, lo[i], h[i], color=col, linewidth=0.7)
            ax.add_patch(plt.Rectangle((i - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), 1e-12), color=col))
        i1, i2 = pos.get(pd.Timestamp(t1)), pos.get(pd.Timestamp(t2))
        last = len(d) - 1
        lc = LINE_LONG if int(signal.side) > 0 else LINE_SHORT
        if i1 is not None and i2 is not None and i2 > i1:
            slope = (v2 - v1) / (i2 - i1)
            xs = np.arange(i1, last + 4)
            ax.plot(xs, v1 + slope * (xs - i1), color=lc, linewidth=1.8)
            ax.scatter([i1, i2], [v1, v2], s=60, facecolors="none", edgecolors=lc, linewidths=1.6, zorder=5)
        ax.scatter([last], [c[-1]], marker="*", s=160, color=lc, zorder=6)
        p = signal.plan
        span = (last - 2, last + 14)
        ax.hlines(p.entry, *span, color=FG, linewidth=1.1)
        ax.hlines(p.stop, *span, color=DOWN, linestyle="--", linewidth=1.1)
        ax.hlines(p.target, *span, color=UP, linestyle=":", linewidth=1.3)
        kind = "ретест" if p.entry_kind is EntryKind.RETEST else "рынок"
        for y, txt in ((p.entry, f"вход ({kind}) {p.entry:.6g}"), (p.stop, f"стоп {p.stop:.6g}"),
                       (p.target, f"цель {p.target:.6g}")):
            ax.annotate(txt, (span[1], y), color=FG, fontsize=8, xytext=(4, 0), textcoords="offset points",
                        va="center")
        ax.set_xlim(-1, last + 26)
        ymin, ymax = min(lo.min(), p.stop, p.target), max(h.max(), p.stop, p.target)
        pad = (ymax - ymin) * 0.04
        ax.set_ylim(ymin - pad, ymax + pad)
        ticks = np.linspace(0, last, 6).astype(int)
        ax.set_xticks(ticks, [d.index[i].strftime("%d.%m %H:%M") for i in ticks])
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_title(f"{signal.symbol} {signal.timeframe.value} · {signal.side.label} · агрессоры {signal.aggr:.0%}",
                     color=FG, fontsize=11, loc="left")
        fig.tight_layout()
        path = self.out_dir / f"{signal.id or 'x'}_{signal.symbol}_{signal.timeframe.value}.png"
        fig.savefig(path, facecolor=BG)
        plt.close(fig)
        return str(path)
