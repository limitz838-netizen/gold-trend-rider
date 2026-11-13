"""
Research round 2 on the Desk gold rule (entries = live app/desk/strategy.py, exits = pullback_s + 4h time exit).
Simulated on FBS XAUUSD M1 with real spreads. DEV = 2024-01..2025-12 (choose), SEALED = 2026 (opened once).

Idea 1  LIMIT ENTRY: instead of market at the next H1 open, a limit order at EMA20 (+/- k ATR toward price),
        valid for V hours. Unfilled = no trade. Stop/target/time exit measured from the FILL.
Idea 2  DAILY TREND FILTER: only take H4-bias trades that agree with the daily trend (D1 EMA fast > slow),
        from GC=F daily closes lagged one day (no look-ahead).
"""
import os, itertools
import numpy as np, pandas as pd
os.environ.setdefault("H1_CSV", "../exness/xauusd_h1_utc.csv"); os.environ.setdefault("H4_CSV", "../exness/xauusd_h4_srvaligned_utc.csv")
import wf_pullback as W
from scalp import load_m1
from app.desk import strategy as S

DEV_END = pd.Timestamp("2026-01-01", tz="UTC")
DAILY = "/home/user/gold-trend-rider/data/gold_daily.csv"


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def daily_bias(fast, slow):
    d = pd.read_csv(DAILY, header=[0, 1], index_col=0); d.columns = [c[0] for c in d.columns]
    d.index = pd.to_datetime(d.index); c = d.Close.dropna()
    b = pd.Series(np.sign(ema(c.to_numpy(), fast) - ema(c.to_numpy(), slow)), index=c.index)
    # known from the NEXT day on (the daily close is complete at ~21:00-22:00 UTC; shift one calendar day to be safe)
    b.index = b.index + pd.Timedelta(days=1)
    return b


def signal_decisions(df):
    """Per H1 bar: signal, atr, bias, ema20 (from the live code)."""
    sig, atr, bias = W.signals(df, True, (1, 21))
    # ema20 of closes, same as S.ema
    e20 = np.array(S.ema(df.Close.tolist(), S.H1_EMA))
    return sig, atr, bias, e20


class M1Sim:
    def __init__(self, m, df, bias_h1):
        self.m = m; self.t = W.epoch(m.index)
        self.o, self.h, self.l, self.c = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.spr = m.spread_points.to_numpy() / 100.0
        self.h1_t = W.epoch(df.index); self.bias = bias_h1

    def bias_at(self, ts):      # bias of the last CLOSED H1 bar at time ts
        i = np.searchsorted(self.h1_t, ts - 3600, side="right") - 1
        return int(self.bias[i]) if i >= 0 else 0

    def run(self, orders, stop_mult=1.5, tp_min=13.0, time_h=4.0, flip=True):
        """orders: list of dict(ts=signal bar close utc s, dir, atr, limit=None|price, valid_min)"""
        out = []; busy = -1
        for od in orders:
            i = np.searchsorted(self.t, od["ts"])
            if i >= len(self.o) - 2 or i <= busy:
                continue
            d, a = od["dir"], od["atr"]
            # ---- entry
            if od.get("limit") is None:
                k0, px = i, self.o[i]
            else:
                lim = od["limit"]; last = min(len(self.o) - 2, i + od["valid_min"]); k0 = None
                for k in range(i, last + 1):
                    if flip and self.bias_at(self.t[k]) == -d:
                        break
                    if (d > 0 and self.o[k] <= lim) or (d < 0 and self.o[k] >= lim):
                        k0, px = k, self.o[k]; break                     # gap through: filled at open
                    if (d > 0 and self.l[k] <= lim) or (d < 0 and self.h[k] >= lim):
                        k0, px = k, lim; break
                if k0 is None:
                    out.append(dict(t=self.m.index[i], dir=d, pnl=0.0, reason="unfilled", filled=False)); continue
            stop = px - d * stop_mult * a; tgt = px + d * max(tp_min, a)
            last = min(len(self.o) - 1, k0 + int(time_h * 60)); k = k0 + (1 if od.get("limit") is None else 0); ex = r = None
            while k <= last:
                if flip and self.bias_at(self.t[k]) == -d:
                    ex, r = self.o[k], "flip"; break
                hs = self.l[k] <= stop if d > 0 else self.h[k] >= stop
                ht = self.h[k] >= tgt if d > 0 else self.l[k] <= tgt
                if hs: ex, r = stop, "stop"; break
                if ht: ex, r = tgt, "target"; break
                k += 1
            if ex is None:
                ex, r = self.c[last], "time"; k = last
            out.append(dict(t=self.m.index[k0], dir=d, pnl=d * (ex - px) - self.spr[k0], reason=r, filled=True)); busy = k
        return pd.DataFrame(out)


def st(tr):
    f = tr[tr.filled] if "filled" in tr else tr
    if len(f) == 0: return dict(n=0, net=0, win=0, pf=0, maxdd=0, avg=0, sharpe=0, fill=0)
    eq = f.pnl.cumsum(); w = f.pnl[f.pnl > 0].sum(); lo = -f.pnl[f.pnl <= 0].sum()
    yrs = max((f.t.iloc[-1] - f.t.iloc[0]).total_seconds() / 86400 / 365.25, 0.05)
    return dict(n=len(f), net=f.pnl.sum(), win=(f.pnl > 0).mean(), pf=w / lo if lo else np.inf, maxdd=(eq - eq.cummax()).min(),
                avg=f.pnl.mean(), sharpe=f.pnl.mean() / f.pnl.std() * np.sqrt(len(f) / yrs) if f.pnl.std() else 0,
                fill=len(f) / max(len(tr), 1))


def fm(s):
    return (f"n {s['n']:4d} (fill {s['fill']*100:3.0f}%) | net ${s['net']:7.0f} | avg ${s['avg']:5.2f} | win {s['win']*100:4.1f}% | "
            f"PF {s['pf']:4.2f} | maxDD ${s['maxdd']:6.0f} | Sh {s['sharpe']:5.2f}")


def main():
    df = W.load(); sig, atr, bias, e20 = signal_decisions(df); m = load_m1(); sim = M1Sim(m, df, bias)
    tc = W.epoch(df.index) + 3600
    base_orders = [dict(ts=int(tc[j]), dir=int(sig[j]), atr=float(atr[j]), close=float(df.Close.iloc[j]), e20=float(e20[j]), j=j)
                   for j in range(len(sig)) if sig[j] != 0]
    dev = lambda tr: tr[tr.t < DEV_END]; sea = lambda tr: tr[tr.t >= DEV_END]

    base = sim.run(base_orders)
    print(f"{len(base_orders)} live signals. BASELINE (market at next open, 4h exit) on M1:")
    print("  DEV ", fm(st(dev(base)))); print("  2026", fm(st(sea(base))))

    # ---------------- Idea 1: limit entries (DEV only for selection)
    print("\nIDEA 1 - LIMIT ENTRY at EMA20 (+k ATR toward price), valid V hours. DEV 2024-25:")
    res1 = {}
    for k, v in itertools.product((0.0, 0.25, 0.5), (2, 4, 8)):
        orders = []
        for od in base_orders:
            lim = od["e20"] + od["dir"] * k * od["atr"]
            if (od["close"] - lim) * od["dir"] <= 0:      # limit would be above price for a long: use market
                lim = None
            orders.append(dict(od, limit=lim, valid_min=v * 60))
        tr = sim.run(orders); res1[(k, v)] = tr
        print(f"  k={k:4} valid {v}h : DEV {fm(st(dev(tr)))}")

    # ---------------- Idea 2: daily trend filter (DEV only)
    print("\nIDEA 2 - DAILY TREND FILTER (take the H4 signal only if the daily trend agrees). DEV 2024-25:")
    res2 = {}
    for fast, slow in ((20, 50), (50, 200), (10, 30)):
        db = daily_bias(fast, slow)
        def ok(od):
            day = pd.Timestamp(od["ts"], unit="s").normalize()
            b = db[db.index <= day]
            return len(b) and b.iloc[-1] == od["dir"]
        orders = [od for od in base_orders if ok(od)]
        tr = sim.run(orders); res2[(fast, slow)] = tr
        print(f"  D1 EMA{fast}/{slow} agree: DEV {fm(st(dev(tr)))}   ({len(orders)} of {len(base_orders)} signals kept)")
    # long-only reference (regime bet, shown for context)
    tr_lo = sim.run([od for od in base_orders if od["dir"] > 0]); print(f"  (long only reference)  DEV {fm(st(dev(tr_lo)))}")

    # ---------------- pick on DEV, open 2026 once
    best1 = max(res1, key=lambda k: st(dev(res1[k]))["sharpe"]); best2 = max(res2, key=lambda k: st(dev(res2[k]))["sharpe"])
    print(f"\nDEV picks: limit k={best1[0]} valid {best1[1]}h ; daily filter EMA{best2[0]}/{best2[1]}")
    print("\nSEALED 2026 (opened once):")
    for name, tr in (("baseline market + 4h", base), (f"limit k={best1[0]} v={best1[1]}h", res1[best1]),
                     (f"daily EMA{best2[0]}/{best2[1]} filter", res2[best2])):
        print(f"  {name:28s} DEV  {fm(st(dev(tr)))}\n  {'':28s} 2026 {fm(st(sea(tr)))}")
        q = sea(tr[tr.filled]).set_index('t').pnl.resample('QE').sum().round(0).tolist(); print(f"  {'':28s} 2026 quarters {q}")
    # combination of both DEV picks
    db = daily_bias(*best2)
    comb = []
    for od in base_orders:
        day = pd.Timestamp(od["ts"], unit="s").normalize(); b = db[db.index <= day]
        if not (len(b) and b.iloc[-1] == od["dir"]): continue
        lim = od["e20"] + od["dir"] * best1[0] * od["atr"]; lim = None if (od["close"] - lim) * od["dir"] <= 0 else lim
        comb.append(dict(od, limit=lim, valid_min=best1[1] * 60))
    tr = sim.run(comb); print(f"  {'both combined':28s} DEV  {fm(st(dev(tr)))}\n  {'':28s} 2026 {fm(st(sea(tr)))}")

    # coin flip on 2026 for the daily-filter pick: random H1 bars, direction = daily trend, same exits
    tr2 = sea(res2[best2][res2[best2].filled]); n_tr = len(tr2); live = tr2.pnl.sum()
    hrs = ((df.index.hour + 1) % 24); idx = np.where((df.index >= DEV_END) & (atr > 0) & (hrs >= 1) & (hrs < 21))[0]
    rng = np.random.default_rng(9); nets = []
    for _ in range(300):
        pk = np.sort(rng.choice(idx, size=int(n_tr * 1.5), replace=False)); ords = []
        for j in pk:
            day = pd.Timestamp(int(tc[j]), unit="s").normalize(); b = db[db.index <= day]; d = int(b.iloc[-1]) if len(b) and b.iloc[-1] else 1
            ords.append(dict(ts=int(tc[j]), dir=d, atr=float(atr[j])))
        nets.append(sim.run(ords, flip=False).pnl.iloc[:n_tr].sum())
    nets = np.array(nets)
    print(f"\nCOIN FLIP 2026 (random timing, direction = daily trend) vs daily-filter pick: ${live:.0f} vs median ${np.median(nets):.0f} "
          f"[5..95%: ${np.percentile(nets,5):.0f}..${np.percentile(nets,95):.0f}] -> beats {100*(nets<live).mean():.0f}% (p={(nets>=live).mean():.3f})")


if __name__ == "__main__":
    main()
