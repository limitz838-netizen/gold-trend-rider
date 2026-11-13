"""Fixed-dollar scalp exits on the live pullback_s entries, simulated on M1 bars (FBS XAUUSD, real spreads).
$ figures are per 0.01 lot (1 gold point = $1). Spread ~ $0.23 median is charged on every trade."""
import os, itertools
import numpy as np, pandas as pd
os.environ.setdefault("H1_CSV", "../exness/xauusd_h1_utc.csv"); os.environ.setdefault("H4_CSV", "../exness/xauusd_h4_srvaligned_utc.csv")
import wf_pullback as W

M1 = "../exness/c39416d9-gold_m1/XAUUSD_M1.csv"


def load_m1():
    m = pd.read_csv(M1, parse_dates=["time_utc"]).rename(columns={"time_utc": "srv"})
    ny = (m.srv - pd.Timedelta(hours=7)).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    m["utc"] = ny.dt.tz_convert("UTC"); m = m.dropna(subset=["utc"]).set_index("utc").sort_index()
    return m


def scalp(m, entries, tp, sl, amb="stop", max_min=None, cost=None):
    """entries: list of (utc_time_of_signal_bar_close, dir). Fill at first M1 open at/after that time."""
    t = W.epoch(m.index); o, h, l, c = m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy(), m.close.to_numpy()
    spr = m.spread_points.to_numpy() / 100.0 if cost is None else np.full(len(o), cost)
    out = []; busy_until = -1
    for ts, d in entries:
        i = np.searchsorted(t, ts)
        if i >= len(o) - 1 or i <= busy_until:
            continue
        px = o[i]; stop = px - d * sl; tgt = px + d * tp; k = i; ex = None; r = None
        last = len(o) - 1 if max_min is None else min(len(o) - 1, i + max_min)
        while k <= last:
            hs = l[k] <= stop if d > 0 else h[k] >= stop
            ht = h[k] >= tgt if d > 0 else l[k] <= tgt
            if hs and ht:
                ex, r = (stop, "stop") if amb == "stop" else (tgt, "target"); break
            if hs: ex, r = stop, "stop"; break
            if ht: ex, r = tgt, "target"; break
            k += 1
        if ex is None:
            ex, r = c[last], "time"; k = last
        out.append(dict(t=m.index[i], dir=d, pnl=d * (ex - px) - spr[i], reason=r, mins=k - i)); busy_until = k
    return pd.DataFrame(out)


def st(tr):
    if len(tr) == 0: return dict(n=0, net=0, win=0, pf=0, maxdd=0, avg=0, mins=0)
    eq = tr.pnl.cumsum(); w = tr.pnl[tr.pnl > 0].sum(); lo = -tr.pnl[tr.pnl <= 0].sum()
    return dict(n=len(tr), net=tr.pnl.sum(), win=(tr.pnl > 0).mean(), pf=w / lo if lo else np.inf,
                maxdd=(eq - eq.cummax()).min(), avg=tr.pnl.mean(), mins=tr.mins.median())


def main():
    df = W.load(); sig, atr, bias = W.signals(df, True, (1, 21))
    t_close = W.epoch(df.index) + 3600
    entries = [(int(t_close[j]), int(sig[j])) for j in range(len(sig)) if sig[j] != 0]
    m = load_m1()
    print(f"M1 bars {len(m)}, {len(entries)} live entries, spread median ${m.spread_points.median()/100:.2f}\n")

    print("FIXED $ TARGET / STOP per 0.01 lot, live entries, whole period Jan 2024 - Nov 2026 (stop wins if both hit in one minute)")
    print(f"{'TP':>4} {'SL':>4} | {'trades':>6} {'win%':>6} {'need%':>6} {'net $':>8} {'avg $':>7} {'PF':>5} {'maxDD $':>8} {'med min':>7}")
    grid = {}
    for tp, sl in itertools.product((2, 3, 4, 5, 8, 13), (2, 3, 4, 5, 8, 13)):
        tr = scalp(m, entries, tp, sl); s = st(tr); grid[(tp, sl)] = tr
        need = (sl + 0.23) / (tp + sl) * 100     # win rate needed to break even after spread
        print(f"{tp:>4} {sl:>4} | {s['n']:>6} {s['win']*100:>6.1f} {need:>6.1f} {s['net']:>8.0f} {s['avg']:>7.2f} {s['pf']:>5.2f} {s['maxdd']:>8.0f} {s['mins']:>7.0f}")

    print("\nThe exact sequence you wrote, used as a cycling target list (stop = same size as target), TP-first if both hit:")
    seq = [2, 3, 5, 3, 4, 2, 4, 5, 2]
    # cycle targets across trades
    rows = []
    for k, (ts, d) in enumerate(entries):
        tp = seq[k % len(seq)]
        rows.append((ts, d, tp))
    tot = [];
    for tp in sorted(set(seq)):
        sub = [(ts, d) for ts, d, x in rows if x == tp]
        tr = scalp(m, sub, tp, tp, amb="tp"); tot.append(tr)
    tr = pd.concat(tot).sort_values("t"); s = st(tr)
    print(f"  n {s['n']} win {s['win']*100:.1f}% net ${s['net']:.0f} avg ${s['avg']:.2f} PF {s['pf']:.2f} maxDD ${s['maxdd']:.0f}")

    print("\nOptimistic assumption (target wins when both hit in the same minute):")
    for tp, sl in ((2, 2), (3, 3), (5, 3), (5, 5), (4, 2)):
        s = st(scalp(m, entries, tp, sl, amb="tp")); print(f"  TP {tp} SL {sl}: n {s['n']} win {s['win']*100:.1f}% net ${s['net']:.0f} PF {s['pf']:.2f}")

    print("\nZero spread (what the entries are worth before costs):")
    for tp, sl in ((2, 2), (3, 3), (5, 3), (5, 5)):
        s = st(scalp(m, entries, tp, sl, cost=0.0)); print(f"  TP {tp} SL {sl}: n {s['n']} win {s['win']*100:.1f}% net ${s['net']:.0f} PF {s['pf']:.2f}")

    print("\nRandom entries (coin flip, same count) with TP 3 / SL 3, 200 runs:")
    rng = np.random.default_rng(1); idx = np.arange(0, len(m) - 1000); nets = []
    for _ in range(200):
        pk = np.sort(rng.choice(idx, size=len(entries), replace=False))
        mt = W.epoch(m.index); ent = [(int(mt[j]), int(rng.choice([-1, 1]))) for j in pk]
        nets.append(scalp(m, ent, 3, 3).pnl.sum())
    live = st(grid[(3, 3)])["net"]; nets = np.array(nets)
    print(f"  live entries ${live:.0f} vs random median ${np.median(nets):.0f} [5..95%: ${np.percentile(nets,5):.0f}..${np.percentile(nets,95):.0f}]")

    print("\nBy year, TP 3 / SL 3 and TP 5 / SL 3:")
    for k in ((3, 3), (5, 3)):
        g = grid[k].set_index("t").pnl.resample("YE").agg(["count", "sum"]); print(f"  {k}: " + ", ".join(f"{i.year}: {int(r['count'])} tr ${r['sum']:.0f}" for i, r in g.iterrows()))


if __name__ == "__main__":
    main()
