"""
Pre-declared exit improvements for pullback_s, tested honestly:
  DEV    = 2024-01 .. 2025-12   (choose here)
  SEALED = 2026-01 .. 2026-11   (opened once, at the end, for the chosen few)
Entries are the live strategy's own (app/desk/strategy.py). Only exits change.
Candidates (named BEFORE looking at 2026):
  A time stop          close at market N bars after entry if neither stop nor target hit
  B breakeven          stop to entry once price has been +X ATR in favour
  C lock               stop to entry + Y ATR once price has been +X ATR in favour
  D wider target       TP = max($13, T x ATR)
  E long only
  F wider stop         2.0 ATR (the walk-forward kept choosing it)
"""
import os, itertools, json
import numpy as np, pandas as pd
os.environ.setdefault("H1_CSV", "../exness/xauusd_h1_utc.csv")
os.environ.setdefault("H4_CSV", "../exness/xauusd_h4_srvaligned_utc.csv")
import wf_pullback as W

DEV_END = pd.Timestamp("2026-01-01", tz="UTC")


def simulate(df, sig, atr, bias, stop=1.5, tp_atr=1.0, tp_min=13.0, be_at=None, lock_to=0.0, time_bars=None,
             long_only=False, flip_exit=True, entries=None):
    o, h, l, c = df.Open.to_numpy(), df.High.to_numpy(), df.Low.to_numpy(), df.Close.to_numpy()
    spr = df.Spread.to_numpy() / 100.0
    n = len(o); trades = []; i = 0
    ent = entries if entries is not None else [(j, int(sig[j])) for j in range(n) if sig[j] != 0]
    for j, d in ent:
        if j < i or j + 1 >= n or atr[j] <= 0 or (long_only and d < 0):
            continue
        a = atr[j]; px = o[j + 1]
        stop_px = px - d * stop * a
        tgt = px + d * max(tp_min, tp_atr * a)
        k = j + 1; ex = None; reason = None; best = px
        while k < n:
            if k > j + 1 and flip_exit and bias[k - 1] != 0 and bias[k - 1] != d:
                ex, reason = o[k], "flip"; break
            if time_bars and k - (j + 1) >= time_bars:
                ex, reason = o[k], "time"; break
            hs = l[k] <= stop_px if d > 0 else h[k] >= stop_px
            ht = h[k] >= tgt if d > 0 else l[k] <= tgt
            if hs:
                ex, reason = stop_px, "stop"; break
            if ht:
                ex, reason = tgt, "target"; break
            # end-of-bar management (worker modifies stops between bars, on live ticks; bar close is conservative)
            best = max(best, h[k]) if d > 0 else min(best, l[k])
            if be_at is not None and (best - px) * d >= be_at * a:
                new = px + d * lock_to * a
                if (new - stop_px) * d > 0:
                    stop_px = new
            k += 1
        if ex is None:
            ex, reason, k = c[-1], "open", n - 1
        trades.append(dict(t=df.index[j + 1], dir=d, pnl=d * (ex - px) - spr[j + 1], reason=reason, hold=k - j))
        i = k + 1
    return pd.DataFrame(trades)


def st(tr):
    if len(tr) == 0:
        return dict(n=0, net=0, pf=0, win=0, avg=0, maxdd=0, sharpe=0)
    eq = tr.pnl.cumsum(); w = tr.pnl[tr.pnl > 0].sum(); lo = -tr.pnl[tr.pnl <= 0].sum()
    yrs = max((tr.t.iloc[-1] - tr.t.iloc[0]).total_seconds() / 86400 / 365.25, 0.05)
    return dict(n=len(tr), net=tr.pnl.sum(), pf=w / lo if lo else np.inf, win=(tr.pnl > 0).mean(), avg=tr.pnl.mean(),
                maxdd=(eq - eq.cummax()).min(), sharpe=(tr.pnl.mean() / tr.pnl.std() * np.sqrt(len(tr) / yrs)) if tr.pnl.std() else 0)


def f(s):
    return (f"n {s['n']:4d} | net ${s['net']:7.1f} | avg ${s['avg']:5.2f} | win {s['win']*100:4.1f}% | PF {s['pf']:4.2f} | "
            f"maxDD ${s['maxdd']:7.1f} | Sharpe {s['sharpe']:5.2f}")


def main():
    df = W.load(); sig, atr, bias = W.signals(df, True, (1, 21))
    dev = lambda tr: tr[tr.t < DEV_END]
    sealed = lambda tr: tr[tr.t >= DEV_END]
    base = simulate(df, sig, atr, bias)
    print("BASELINE live rule  DEV:", f(st(dev(base))))

    # ---------- one change at a time, DEV only
    print("\nONE CHANGE AT A TIME (DEV 2024-2025 only):")
    singles = {}
    for nb in (2, 3, 4, 6, 8):
        singles[f"A time stop {nb} bars"] = dict(time_bars=nb)
    for x in (0.5, 0.75, 1.0):
        singles[f"B breakeven at +{x} ATR"] = dict(be_at=x, lock_to=0.0)
    for x, y in ((0.75, 0.25), (1.0, 0.25), (1.0, 0.5)):
        singles[f"C lock +{y} ATR at +{x} ATR"] = dict(be_at=x, lock_to=y)
    for t in (1.25, 1.5, 2.0):
        singles[f"D target {t} ATR"] = dict(tp_atr=t)
    singles["E long only"] = dict(long_only=True)
    singles["F stop 2.0 ATR"] = dict(stop=2.0)
    res = {}
    for name, p in singles.items():
        tr = simulate(df, sig, atr, bias, **p); res[name] = (p, tr)
        print(f"  {name:28s}", f(st(dev(tr))))

    # ---------- combinations of the two mechanisms the diagnostics pointed at (time stop x protection), DEV
    print("\nCOMBINATIONS time stop x protection (DEV):")
    rows = []
    for nb, (be, lk), t in itertools.product((3, 4, 6, None), ((None, 0), (0.5, 0), (0.75, 0), (0.75, 0.25), (1.0, 0.25)), (1.0, 1.5)):
        p = dict(time_bars=nb, be_at=be, lock_to=lk, tp_atr=t)
        tr = simulate(df, sig, atr, bias, **p); s = st(dev(tr)); rows.append((s["net"], s["sharpe"], s["pf"], s["n"], p))
    rows.sort(key=lambda r: -r[1])
    for net, sh, pf, n, p in rows[:12]:
        print(f"  net ${net:7.1f} Sharpe {sh:5.2f} PF {pf:4.2f} n {n:4d}  {p}")
    print(f"  ... {sum(1 for r in rows if r[0] > 0)}/{len(rows)} combinations profitable on DEV")

    # ---------- pick on DEV (by Sharpe, must have >= 250 trades), THEN open SEALED once
    picks = {
        "live rule (baseline)": dict(),
        "DEV best combination": rows[0][4],
        "simple: time stop 4 + breakeven +0.75": dict(time_bars=4, be_at=0.75, lock_to=0.0),
        "simple: time stop 4 only": dict(time_bars=4),
        "simple: breakeven +0.75 only": dict(be_at=0.75, lock_to=0.0),
    }
    print("\nSEALED 2026 (opened once):")
    out = {}
    for name, p in picks.items():
        tr = simulate(df, sig, atr, bias, **p); sd, ss = st(dev(tr)), st(sealed(tr))
        out[name] = dict(params=p, dev=sd, sealed=ss)
        print(f"  {name:38s} DEV {f(sd)}\n  {'':38s} 2026 {f(ss)}")
        q = sealed(tr).set_index("t").pnl.resample("QE").sum().round(0).tolist()
        print(f"  {'':38s} 2026 by quarter: {q}")

    # ---------- coin flip on SEALED for the simple candidate: same exits, random bars/direction
    name = "simple: time stop 4 + breakeven +0.75"; p = picks[name]
    tr = sealed(simulate(df, sig, atr, bias, **p)); n_tr = len(tr); live_net = tr.pnl.sum()
    hours = ((df.index.hour + 1) % 24); idx26 = np.where((df.index >= DEV_END) & (atr > 0) & (hours >= 1) & (hours < 21))[0]
    rng = np.random.default_rng(11); nets = []
    for _ in range(1000):
        picks_i = np.sort(rng.choice(idx26, size=int(n_tr * 1.6), replace=False))
        ent = [(int(j), int(rng.choice([-1, 1]))) for j in picks_i]
        r = simulate(df, sig, atr, bias, entries=ent, flip_exit=False, **p).iloc[:n_tr]; nets.append(r.pnl.sum())
    nets = np.array(nets)
    print(f"\nCOIN FLIP on 2026 for '{name}': candidate ${live_net:.0f} vs random median ${np.median(nets):.0f} "
          f"[5..95%: ${np.percentile(nets,5):.0f}..${np.percentile(nets,95):.0f}] -> beats {100*(nets<live_net).mean():.0f}% (p={(nets>=live_net).mean():.3f})")
    json.dump(out, open("improve_summary.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
