"""
Test the Desk's PAPER strategies with the real code in app/desk/strategy.py:
  BTCUSD  squeeze_r7n12  quiet ATR < 0.7x median, break of the 12-hour range, stop 1.5 ATR, target 3 ATR,
                         breakeven at +1R (flags: be=True), no trail, no H4 flip, 24/7
  V100    meanrev_k25    2.5 SD band, calm ATR <= 1.0x, stop 1.5 ATR, target EMA20, 12h time exit, no be
Signals on the H1 close, filled at the next H1 open, stop checked before target inside a bar.
Usage:  python3 paper_test.py <h1.csv> <strategy> <cost_in_price_units> [dev_end]
        the csv has a UTC index and Open/High/Low/Close columns.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/nolimitz_backend")
from app.desk import strategy as S

WIN = 400


def load(path):
    first = open(path).readline()
    if first.startswith("Price,"):          # yfinance two-row header
        df = pd.read_csv(path, header=[0, 1], index_col=0); df.columns = [c[0] for c in df.columns]
        df = df[[i not in ("Ticker", "Datetime", "Date") for i in df.index.astype(str)]]
    else:
        df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    return df[(df.High >= df.Low) & (df.Close > 0)]


def epoch(idx):
    idx = pd.DatetimeIndex(idx).tz_convert("UTC").tz_localize(None)
    return np.asarray(idx.values.astype("datetime64[s]").astype("int64"))


def decisions(df, name):
    t = epoch(df.index); o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    bars = [dict(t=int(a), o=float(b), h=float(cc), l=float(d), c=float(e)) for a, b, cc, d, e in zip(t, o, h, l, c)]
    fn = S.STRATEGIES[name]; n = len(bars)
    sig = np.zeros(n, int); atr = np.zeros(n); tgt = np.full(n, np.nan); stopm = np.zeros(n); tgtm = np.zeros(n)
    for i in range(S.MIN_H1_BARS + 1, n):
        d = fn(bars[max(0, i - WIN + 1):i + 1], 0)
        sig[i], atr[i], stopm[i], tgtm[i] = d.signal, d.atr, d.stop_mult, d.target_mult
        if d.target_price is not None:
            tgt[i] = d.target_price
    return sig, atr, tgt, stopm, tgtm, S.flags(name)


def simulate(df, sig, atr, tgtp, stopm, tgtm, fl, cost, entries=None):
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close")); n = len(o)
    ent = entries if entries is not None else [(j, int(sig[j])) for j in range(n) if sig[j]]
    out = []; i = 0
    for j, d in ent:
        if j < i or j + 1 >= n or atr[j] <= 0:
            continue
        a = atr[j]; px = o[j + 1]; risk = stopm[j] * a; stop = px - d * risk
        if not np.isnan(tgtp[j]):
            tg = tgtp[j]
            if (tg - px) * d <= 0:
                continue                        # worker: "price already at the target" -> skipped
        else:
            tg = px + d * tgtm[j] * a
        k = j + 1; ex = r = None; best = px
        while k < n:
            if fl["time_exit_h"] and (k - (j + 1)) >= fl["time_exit_h"]:
                ex, r = o[k], "time"; break
            hs = l[k] <= stop if d > 0 else h[k] >= stop
            ht = h[k] >= tg if d > 0 else l[k] <= tg
            if hs: ex, r = stop, "stop"; break
            if ht: ex, r = tg, "target"; break
            best = max(best, h[k]) if d > 0 else min(best, l[k])
            if fl["be"]:
                new = S.manage_stop(d, px, stop, risk, best, a, trail=fl["trail"])
                if new is not None: stop = new
            k += 1
        if ex is None:
            ex, r, k = c[-1], "open", n - 1
        pnl = d * (ex - px) - cost
        out.append(dict(t=df.index[j + 1], dir=d, pnl=pnl, r=pnl / risk, reason=r, hold=k - j)); i = k + 1
    return pd.DataFrame(out)


def st(tr):
    if len(tr) == 0: return dict(n=0, net=0, R=0, win=0, pf=0, maxdd=0, avg_r=0)
    eq = tr.r.cumsum(); w = tr.r[tr.r > 0].sum(); lo = -tr.r[tr.r <= 0].sum()
    return dict(n=len(tr), net=tr.pnl.sum(), R=tr.r.sum(), win=(tr.r > 0).mean(), pf=w / lo if lo else np.inf,
                maxdd=(eq - eq.cummax()).min(), avg_r=tr.r.mean())


def f(s):
    return f"n {s['n']:4d} | {s['R']:+7.1f} R | avg {s['avg_r']:+.3f} R | win {s['win']*100:4.1f}% | PF {s['pf']:4.2f} | maxDD {s['maxdd']:6.1f} R | net {s['net']:+.1f} pts"


def main():
    path, name, cost = sys.argv[1], sys.argv[2], float(sys.argv[3])
    dev_end = pd.Timestamp(sys.argv[4] if len(sys.argv) > 4 else "2026-01-01", tz="UTC")
    df = load(path); sig, atr, tgtp, stopm, tgtm, fl = decisions(df, name)
    tr = simulate(df, sig, atr, tgtp, stopm, tgtm, fl, cost)
    print(f"{name} on {os.path.basename(path)}: {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}, {len(df)} H1 bars, "
          f"{int((sig != 0).sum())} signals, cost {cost} per trade, flags {fl}")
    print("  ALL     ", f(st(tr)))
    print("  pre-2026", f(st(tr[tr.t < dev_end])))
    print("  2026    ", f(st(tr[tr.t >= dev_end])))
    q = tr.set_index("t").r.resample("QE").agg(["sum", "count"])
    print("  by quarter (R):", ", ".join(f"{i:%Y}Q{(i.month-1)//3+1} {r['sum']:+.1f} ({int(r['count'])})" for i, r in q.iterrows()))
    print("  by direction:", tr.groupby("dir").r.agg(["count", "sum"]).round(1).to_dict())
    print("  exits:", tr.groupby("reason").r.agg(["count", "sum"]).round(1).to_dict())
    # coin flip: random bars, random direction, same exits, same count
    rng = np.random.default_rng(2); ok = np.where(atr > 0)[0]; nets = []
    for _ in range(1000):
        pk = np.sort(rng.choice(ok, size=int(len(tr) * 1.5), replace=False))
        ent = [(int(j), int(rng.choice([-1, 1]))) for j in pk]
        r = simulate(df, sig, atr, np.full(len(df), np.nan), stopm if stopm.max() else np.full(len(df), 1.5),
                     tgtm if tgtm.max() else np.full(len(df), 3.0), fl, cost, entries=ent).iloc[:len(tr)]
        nets.append(r.r.sum())
    nets = np.array(nets); live = tr.r.sum()
    print(f"  COIN FLIP (1000 runs, same exits): {live:+.1f} R vs median {np.median(nets):+.1f} R "
          f"[5..95%: {np.percentile(nets,5):+.1f}..{np.percentile(nets,95):+.1f}] -> beats {100*(nets<live).mean():.0f}% (p={(nets>=live).mean():.3f})")
    tr.to_csv(f"trades_{name}_{os.path.basename(path)}", index=False)


if __name__ == "__main__":
    main()
