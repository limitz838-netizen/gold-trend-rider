# Desk `pullback_s` gold rule — walk-forward and coin-flip test

`wf_pullback.py` imports `app/desk/strategy.py` from the backend (set `sys.path` at the top) so the entries
are produced by the exact live code. Only the exits are simulated, the way `app/desk/worker.py` manages them.

Run on the broker's own data (FBS-Demo M1, converted to UTC, H4 aligned to the broker's server clock):

    H1_CSV=desk_research/data_fbs/xauusd_h1_utc.csv H4_CSV=desk_research/data_fbs/xauusd_h4_srvaligned_utc.csv \
    WF_START=2024-10-01 MID=2025-06-01 python3 desk_research/wf_pullback.py

Run on COMEX futures (GC=F from Yahoo):   `python3 desk_research/wf_pullback.py`

Results (per 0.01 lot, cost = the bar's real spread, Jan 2024 - 11 Nov 2026, 828 trades):
net -$137, win 55.8%, profit factor 0.98, max drawdown -$924. Beats a 50/50 coin flip in 65% of runs (p = 0.35)
and random-timing trend-following entries in 41% of runs. No edge is demonstrated. See summary_fbs.json.
