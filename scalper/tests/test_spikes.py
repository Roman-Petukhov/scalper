import numpy as np

from research.spikes import spike_machine


def _run(o, h, lo, c, sig=0.01, m=2.0, f=0.5, T=5, stop=np.inf):
    n = len(c)
    return spike_machine(np.array(o, float), np.array(h, float), np.array(lo, float), np.array(c, float),
                         np.full(n, sig), m, f, T, stop, 0.0, 0.0)


def test_spike_down_filled_and_half_retrace_taken():
    # ref 100, D = 2% -> покупка 98; минута 1 простреливает до 97.5; минута 2 откатывает выше 99 (тейк 50%)
    i, r = _run([100, 99, 98.5], [100.1, 99.5, 99.6], [99.9, 97.5, 98.4], [100, 98.6, 99.5])
    assert list(i) == [1] and np.isclose(r[0], 99 / 98 - 1)


def test_take_profit_not_counted_in_entry_minute_and_time_exit():
    i, r = _run([100, 99, 98.5, 98.4], [100.1, 99.9, 98.6, 98.5], [99.9, 97.9, 98.3, 98.2],
                [100, 99.8, 98.5, 98.3], T=2)
    assert list(i) == [1] and np.isclose(r[0], 98.3 / 98 - 1)


def test_stop_in_entry_minute_is_worst_case():
    i, r = _run([100, 99], [100.1, 99.5], [99.9, 96.0], [100, 99.4], stop=1.0)   # стоп 98 * (1 - 0.02) = 96.04
    assert list(i) == [1] and np.isclose(r[0], 96.04 / 98 - 1)
