"""
Walk-forward + coin-flip test of the Desk's live gold rule (pullback_s) using the REAL strategy code
in app/desk/strategy.py. Nothing is reimplemented: entries come from S.evaluate(); only exits
(stop / target / H4 flip) are simulated here, the way app/desk/worker.py manages them.

Data: GC=F 1-hour bars (Yahoo), UTC. Results are in gold PRICE POINTS per 0.01 lot = US dollars.
"""
import sys, itertools, json
import numpy as np, pandas as pd

sys.path.insert(0, "/home/user/nolimitz_backend")
from app.desk import strategy as S

import os
CSV = os.environ.get("H1_CSV", "/home/user/gold-trend-rider/data/gold_1h.csv")
H4_CSV = os.environ.get("H4_CSV")   # broker-aligned H4 bars (UTC open time); else resampled from H1
SPREAD_COL = True                   # use the bar's real spread as the trade cost when present
COST = 0.30            # $ per 0.01-lot round trip (spread + commission). Exness standard gold ~0.2-0.35
WIN_H1 = 400           # H1 bars passed to evaluate(): EMA20/ATR14 fully converged (0.9^400 ~ 0)
WIN_H4 = 1000          # same as worker Cfg.H4_BARS
TP_MIN = S.TP_MIN_POINTS   # 13 points per 0.01 lot

# ---------------------------------------------------------------- data
def load():
    if H4_CSV:   # broker export: plain header, UTC index, Spread column (points)
        df = pd.read_csv(CSV, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["Open", "High", "Low", "Close", "Spread"]].dropna()
        return df[(df.High >= df.Low) & (df.Close > 0)]
    df = pd.read_csv(CSV, header=[0, 1], index_col=0)
    df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df[(df.High >= df.Low) & (df.Close > 0)]
    df = df[~df.index.duplicated()]
    return df


def bars_of(df):
    t = (df.index.view("int64") // 10**9).astype(int)
    return [dict(t=int(a), o=float(b), h=float(c), l=float(d), c=float(e))
            for a, b, c, d, e in zip(t, df.Open, df.High, df.Low, df.Close)]


def h4_from_h1(df):
    """H4 bars aligned to 00:00 UTC (Exness server time is UTC). Only complete-looking bars count."""
    g = df.resample("4h", label="left", closed="left").agg(Open=("Open", "first"), High=("High", "max"),
                                                           Low=("Low", "min"), Close=("Close", "last")).dropna()
    return g


# ---------------------------------------------------------------- signals (real strategy code)
def signals(df, loose, session, regime=(S.ATR_RATIO_MIN, S.ATR_RATIO_MAX)):
    """For every H1 bar i (closed), the Decision of the live strategy. Returns arrays sig, atr, bias."""
    h1 = bars_of(df)
    if H4_CSV:
        h4df = pd.read_csv(H4_CSV, index_col=0, parse_dates=True); h4df.index = pd.to_datetime(h4df.index, utc=True)
    else:
        h4df = h4_from_h1(df)
    h4 = bars_of(h4df)
    h4_close_t = np.array([b["t"] + 4 * 3600 for b in h4])      # when each H4 bar is closed
    old = (S.ATR_RATIO_MIN, S.ATR_RATIO_MAX)
    S.ATR_RATIO_MIN, S.ATR_RATIO_MAX = regime
    n = len(h1)
    sig = np.zeros(n, int); atr = np.zeros(n); bias = np.zeros(n, int)
    k4 = 0; cur_bias = 0; last_k4 = -1
    for i in range(S.MIN_H1_BARS + 1, n):
        t_close = h1[i]["t"] + 3600
        while k4 < len(h4) and h4_close_t[k4] <= t_close:
            k4 += 1
        if k4 != last_k4:                      # new H4 bar closed -> recompute bias
            cur_bias = S.h4_bias(h4[max(0, k4 - WIN_H4):k4])
            last_k4 = k4
        d = S.evaluate(h1[max(0, i - WIN_H1 + 1):i + 1], cur_bias, loose=loose, session=session)
        sig[i], atr[i], bias[i] = d.signal, d.atr, cur_bias
    S.ATR_RATIO_MIN, S.ATR_RATIO_MAX = old
    return sig, atr, bias


# ---------------------------------------------------------------- exits (as worker.py manages pullback_s)
def simulate(df, sig, atr, bias, stop_mult, tp_min, flip_exit=True, entries=None, cost=COST):
    """One position at a time. Signal on bar i close -> fill at bar i+1 open. Stop/target checked on each
    bar's high/low (stop first if both hit). H4 flip closes at the next bar open. Returns trade list."""
    o, h, l, c = df.Open.to_numpy(), df.High.to_numpy(), df.Low.to_numpy(), df.Close.to_numpy()
    spr = (df.Spread.to_numpy() / 100.0) if ("Spread" in df and cost is COST) else np.full(len(o), cost)
    n = len(o); trades = []; i = 0
    ent = entries if entries is not None else [(j, int(sig[j])) for j in range(n) if sig[j] != 0]
    ptr = 0
    while ptr < len(ent):
        j, d = ent[ptr]; ptr += 1
        if j < i or j + 1 >= n or atr[j] <= 0:
            continue
        a = atr[j]; px = o[j + 1]
        stop = px - d * stop_mult * a
        tp_dist = max(tp_min, a) if tp_min else a
        tgt = px + d * tp_dist
        k = j + 1; exit_px = None; reason = None
        while k < n:
            if k > j + 1 and flip_exit and bias[k - 1] != 0 and bias[k - 1] != d:
                exit_px, reason = o[k], "flip"; break
            hit_stop = l[k] <= stop if d > 0 else h[k] >= stop
            hit_tgt = h[k] >= tgt if d > 0 else l[k] <= tgt
            if hit_stop:
                exit_px, reason = stop, "stop"; break
            if hit_tgt:
                exit_px, reason = tgt, "target"; break
            k += 1
        if exit_px is None:
            exit_px, reason, k = c[-1], "open", n - 1
        pnl = d * (exit_px - px) - spr[j + 1]
        trades.append(dict(entry_i=j + 1, exit_i=k, t=df.index[j + 1], dir=d, pnl=pnl, r=pnl / (stop_mult * a),
                           reason=reason, hold=k - j))
        i = k + 1   # worker re-arms on the next bar after the close
    return pd.DataFrame(trades)


def stats(tr, label=""):
    if tr is None or len(tr) == 0:
        return dict(n=0, net=0, pf=0, win=0, avg=0, maxdd=0, sharpe=0)
    eq = tr.pnl.cumsum(); dd = (eq - eq.cummax()).min()
    w = tr.pnl[tr.pnl > 0].sum(); lo = -tr.pnl[tr.pnl <= 0].sum()
    # trade-level "Sharpe": mean/sd per trade * sqrt(trades per year)
    yrs = max((tr.t.iloc[-1] - tr.t.iloc[0]).total_seconds() / 86400 / 365.25, 1 / 52)
    sh = tr.pnl.mean() / tr.pnl.std() * np.sqrt(len(tr) / yrs) if tr.pnl.std() > 0 else 0
    return dict(n=len(tr), net=tr.pnl.sum(), pf=w / lo if lo > 0 else np.inf, win=(tr.pnl > 0).mean(),
                avg=tr.pnl.mean(), maxdd=dd, sharpe=sh, per_week=len(tr) / (yrs * 52))


def fmt(s):
    return (f"n {s['n']:4d} | net ${s['net']:8.1f} | avg ${s['avg']:6.2f} | win {s['win']*100:4.1f}% | "
            f"PF {s['pf']:4.2f} | maxDD ${s['maxdd']:7.1f} | Sharpe {s['sharpe']:5.2f} | {s.get('per_week',0):.1f}/wk")


def window(tr, a, b):
    return tr[(tr.t >= a) & (tr.t < b)]


# ---------------------------------------------------------------- main
def main():
    df = load()
    print(f"{os.path.basename(CSV)} {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}, {len(df)} bars, cost = {'real bar spread' if H4_CSV else COST}\n")

    ENTRY_GRID = {"loose": [False, True], "session": [(7, 20), (1, 21), (0, 24)]}
    EXIT_GRID = {"stop": [1.0, 1.5, 2.0], "tp_min": [0.0, 8.0, 13.0, 20.0]}
    LIVE = dict(loose=True, session=(1, 21), stop=1.5, tp_min=13.0)
    OLD = dict(loose=False, session=(7, 20), stop=1.5, tp_min=13.0)       # 30 Oct rule
    ORIG = dict(loose=False, session=(7, 20), stop=1.5, tp_min=None)      # September rule had 2.5 ATR TP + trail; approximated below

    # signals once per entry combo
    SIG = {}
    for lo, se in itertools.product(*ENTRY_GRID.values()):
        SIG[(lo, se)] = signals(df, lo, se)
        print(f"  signals loose={lo!s:5} session={se}: {int((SIG[(lo,se)][0]!=0).sum())} entries")
    print()

    # all configs, full trade lists
    ALL = {}
    for lo, se in SIG:
        sig, atr, bias = SIG[(lo, se)]
        for st, tp in itertools.product(*EXIT_GRID.values()):
            ALL[(lo, se, st, tp)] = simulate(df, sig, atr, bias, st, tp)
    key = lambda c: (c["loose"], c["session"], c["stop"], c["tp_min"])

    # ---- 1) the live rule, by quarter, and the periods quoted in strategy.py
    live = ALL[key(LIVE)]; old = ALL[key(OLD)]
    print("LIVE rule pullback_s (loose, 01-21 UTC, stop 1.5 ATR, TP max($13, 1 ATR), H4 flip exit), per 0.01 lot")
    print("  whole period :", fmt(stats(live)))
    print("  30 Oct rule  :", fmt(stats(old)), "  (break trigger, 07-20 UTC)")
    periods = [("May24-Jul25", "2024-05-01", "2025-08-01"), ("Aug25-Jan26", "2025-08-01", "2026-02-01"),
               ("Jan26-Oct26", "2026-02-01", "2026-11-01"), ("Nov26 (after the 2 Nov change)", "2026-11-01", "2026-12-01")]
    print("\n  Periods quoted in the strategy.py comment (their numbers: live +250/+352/+869, old -96/+180/+292):")
    for name, a, b in periods:
        print(f"    {name:32s} live {fmt(stats(window(live, a, b)))}")
        print(f"    {'':32s} old  {fmt(stats(window(old, a, b)))}")
    print("\n  By quarter (live rule):")
    q = live.set_index("t").pnl.resample("QE").agg(["sum", "count"])
    for t, r in q.iterrows():
        print(f"    {t:%Y-Q}{(t.month-1)//3+1}: ${r['sum']:7.1f}  ({int(r['count'])} trades)")

    # ---- 2) walk-forward: pick the config on the past, trade the next quarter blind
    print("\nWALK-FORWARD: every quarter, pick the best of 72 configs on ALL prior data (by net $, min 40 trades),"
          "\n  then trade the next quarter with it. The live config's score in the same quarter is shown beside it.")
    starts = pd.date_range(os.environ.get("WF_START", "2025-04-01"), "2026-10-01", freq="QS", tz="UTC")
    wf_rows = []
    for a in starts:
        b = a + pd.offsets.QuarterBegin(startingMonth=1)
        best, best_net = None, -1e9
        for k, tr in ALL.items():
            past = tr[tr.t < a]
            if len(past) >= 40 and past.pnl.sum() > best_net:
                best, best_net = k, past.pnl.sum()
        picked = stats(window(ALL[best], a, b)); lv = stats(window(live, a, b))
        wf_rows.append(dict(q=f"{a:%Y-%m}", picked=best, picked_net=picked["net"], picked_n=picked["n"],
                            live_net=lv["net"], live_n=lv["n"]))
        print(f"  {a:%Y-%m}: picked loose={best[0]!s:5} sess={best[1]} stop={best[2]} tp={best[3]:>4} "
              f"-> ${picked['net']:7.1f} ({picked['n']} tr) | live ${lv['net']:7.1f} ({lv['n']} tr)")
    wf = pd.DataFrame(wf_rows)
    print(f"  stitched blind quarters: walk-forward ${wf.picked_net.sum():.1f} | live config ${wf.live_net.sum():.1f}")

    # how special is the live config? rank among all 72 on each half
    mid = pd.Timestamp(os.environ.get("MID", "2025-09-01"), tz="UTC")
    r1 = sorted(((stats(window(t, df.index[0], mid))["net"], k) for k, t in ALL.items()), reverse=True)
    r2 = sorted(((stats(window(t, mid, df.index[-1]))["net"], k) for k, t in ALL.items()), reverse=True)
    rank1 = [k for _, k in r1].index(key(LIVE)) + 1; rank2 = [k for _, k in r2].index(key(LIVE)) + 1
    print(f"\n  Live config rank among 72 configs: first half #{rank1}, second half #{rank2}. "
          f"Configs profitable: first half {sum(1 for v,_ in r1 if v>0)}/72, second half {sum(1 for v,_ in r2 if v>0)}/72")
    print("  Top 5 second half:", [(k, round(v)) for v, k in r2[:5]])

    # ---- 3) coin flip: same number of trades, same session hours, random bar & direction, same exits
    print(f"\nCOIN FLIP (2000 runs): random entry bars inside the live session, random direction, SAME exits")
    sig, atr, bias = SIG[(True, (1, 21))]
    hours = ((df.index.hour + 1) % 24)
    eligible = np.where((atr > 0) & (hours >= 1) & (hours < 21))[0]
    n_tr = len(live); rng = np.random.default_rng(3)
    nets = []
    for _ in range(2000):
        picks = np.sort(rng.choice(eligible, size=int(n_tr * 1.6), replace=False))  # extra: overlaps get skipped
        ent = [(int(j), int(rng.choice([-1, 1]))) for j in picks]
        tr = simulate(df, sig, atr, bias, 1.5, 13.0, flip_exit=False, entries=ent)
        tr = tr.iloc[:n_tr]
        nets.append(tr.pnl.sum())
    nets = np.array(nets); live_net = live.pnl.sum()
    print(f"  live net ${live_net:.1f} vs coin-flip median ${np.median(nets):.1f} "
          f"[5%..95%: ${np.percentile(nets,5):.0f} .. ${np.percentile(nets,95):.0f}]  "
          f"-> live beats {100*(nets < live_net).mean():.1f}% of runs (p = {(nets >= live_net).mean():.3f})")
    # same, but direction follows the H4 bias (tests whether the pullback TIMING adds anything over just trend-following)
    nets2 = []
    for _ in range(2000):
        picks = np.sort(rng.choice(eligible, size=int(n_tr * 1.6), replace=False))
        ent = [(int(j), int(bias[j]) if bias[j] else 1) for j in picks]
        tr = simulate(df, sig, atr, bias, 1.5, 13.0, flip_exit=True, entries=ent).iloc[:n_tr]
        nets2.append(tr.pnl.sum())
    nets2 = np.array(nets2)
    print(f"  random timing, direction = H4 trend: median ${np.median(nets2):.1f} "
          f"[5%..95%: ${np.percentile(nets2,5):.0f} .. ${np.percentile(nets2,95):.0f}]  "
          f"-> live beats {100*(nets2 < live_net).mean():.1f}% of runs (p = {(nets2 >= live_net).mean():.3f})")

    # ---- 4) cost sensitivity
    print("\nCOST SENSITIVITY (live rule, whole period):")
    for cst in (0.0, 0.3, 0.5, 0.8):
        tr = simulate(df, sig, atr, bias, 1.5, 13.0, cost=cst)
        print(f"  ${cst:.2f}/trade: {fmt(stats(tr))}")

    live.to_csv("live_trades.csv", index=False)
    json.dump(dict(live=stats(live), wf=wf.to_dict("records"), coin_p=float((nets >= live_net).mean()),
                   trend_p=float((nets2 >= live_net).mean())), open("summary.json", "w"), default=str, indent=1)


if __name__ == "__main__":
    main()
