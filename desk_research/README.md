# Desk `pullback_s` gold rule — walk-forward and coin-flip test

`wf_pullback.py` imports `app/desk/strategy.py` from the backend (set `sys.path` at the top) so the entries
are produced by the exact live code. Only the exits are simulated, the way `app/desk/worker.py` manages them.

Run on the broker's own data (FBS-Demo M1, converted to UTC, H4 aligned to the broker's server clock):

    H1_CSV=desk_research/data_fbs/xauusd_h1_utc.csv H4_CSV=desk_research/data_fbs/xauusd_h4_srvaligned_utc.csv \
    WF_START=2024-10-01 MID=2025-06-01 python3 desk_research/wf_pullback.py

Run on COMEX futures (GC=F from Yahoo):   `python3 desk_research/wf_pullback.py`

CORRECTION (13 Nov 2026, later the same day): the first version of these scripts converted timestamps with
`index.view("int64") // 10**9`, which assumes nanoseconds; pandas 3 stores microseconds, so every epoch second was
1,000x too small. That scrambled the 01-21 UTC session filter and matched the H4 bias to bars ~125 days stale.
All numbers published from the first version were wrong, and a change merged on their basis was reverted.
`epoch()` in wf_pullback.py is the fixed conversion; the figures below are from the fixed scripts.

Results, live rule, FBS data, per 0.01 lot, cost = the bar's real spread, Jan 2024 - 11 Nov 2026, 980 trades:
net +$506, win 55.0%, profit factor 1.06, max drawdown -$460, Sharpe 0.48, 8 of 12 quarters positive.
Beats a 50/50 coin flip in 92% of 2,000 runs (p = 0.08); beats random-timing trend-following in 70%.
COMEX futures (GC=F): net +$134, profit factor 1.02. Thin, positive, not yet statistically secure.

improve.py (exits; DEV 2024-25, sealed 2026): 4-hour time exit does NOT beat the baseline on DEV. The DEV pick,
"stop to entry+0.25 ATR once +0.75 ATR in profit, target 1.5 ATR", scored DEV +$528 / 2026 +$553 (PF 1.16);
it still needs an M1-resolution check and a coin-flip test before anyone acts on it.
research2.py (M1 simulation): limit entry at EMA20 failed on 2026 (-$55); daily-trend filter underperformed
the baseline on 2026 (+$173 vs +$411). scalp.py: every fixed-dollar target from $2 to $8 loses after spread.
