"""
honest_resting_stop.py
======================
The realistic version of the continuation entry: a RESTING STOP order at the break
level (ref_weak). Unlike account_backtest's `--entry-timing level` (which only fills on
episodes that CONFIRMED), this fills the instant price first trades back through the
level during the pullback -- so it INCLUDES false breaks (price pokes the level, fills
the stop, then reverses into the stop-loss). Those false breaks are the losers the
confirmed-only backtest can't see.

It walks EVERY m15 pullback episode (continuation or not), arms a stop at ref_weak, and:
  * finds the pullback extreme after the swing top,
  * fires the entry at the FIRST M5 bar that re-touches ref_weak,
  * stop = pullback extreme -/+ stop_buf_atr*ATR(M5), target = major_high/liq_high/mmN,
  * resolves on M5 (stop-first, conservative),
then runs the same account simulation as account_backtest and reports the result PLUS a
breakdown of confirmed-continuation fills vs false-break fills.

Nothing in the pipeline is modified -- this is a new, additive analysis script.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import smc_labeling as L
import account_backtest as AB


def _target_price(bull, weak, strong, ctime, m15ts, m15H, m15L, dt15, target, liq_lb, major_lb):
    if target.startswith("mm"):
        mult = float(target[2:]); sign = 1.0 if bull else -1.0
        return weak + sign * mult * abs(weak - strong)
    lb = liq_lb if target == "liq_high" else major_lb
    mask = (m15ts >= ctime - lb * max(dt15, 1)) & (m15ts <= ctime)
    if bull:
        c = m15H[mask]; c = c[c > weak]
        return (float(np.min(c)) if target == "liq_high" else float(np.max(c))) if c.size else np.nan
    c = m15L[mask]; c = c[c < weak]
    return (float(np.max(c)) if target == "liq_high" else float(np.min(c))) if c.size else np.nan


def generate_resting_stop_trades(master, labels, *, target="major_high", stop_buf_atr=0.5,
                                 max_bars_m5=480, liq_lookback=96, major_lookback=480,
                                 fill_buffer_bars=6):
    pb = labels["m15_pullback"].copy()
    if pb.empty:
        return [], {}
    ts = AB._ns(master["timestamp"])
    H = master["high"].to_numpy(float); Lo = master["low"].to_numpy(float); C = master["close"].to_numpy(float)
    n = len(ts)
    atr5 = pd.Series(H - Lo).rolling(14, min_periods=1).mean().to_numpy()
    m15 = L.resample_ohlc(master, "m15")
    m15ts = AB._ns(m15["timestamp"]); m15H = m15["high"].to_numpy(float); m15L = m15["low"].to_numpy(float)
    dt15 = int(np.median(np.diff(m15ts))) if len(m15ts) > 2 else 0

    trades = []
    stats = dict(episodes=0, filled=0, confirmed_fill=0, falsebreak_fill=0, no_fill=0)
    for r in pb.itertuples(index=False):
        weak = float(getattr(r, "ref_weak", np.nan)); strong = float(getattr(r, "ref_strong", np.nan))
        dt = getattr(r, "decision_time", None); ct = getattr(r, "resolve_time", None)
        if not (np.isfinite(weak) and np.isfinite(strong)) or pd.isna(dt) or pd.isna(ct):
            continue
        bull = getattr(r, "direction", "bull") == "bull"
        cont = int(pd.to_numeric(getattr(r, "continuation_occurred", 0), errors="coerce") or 0)
        stats["episodes"] += 1
        dtime = pd.Timestamp(dt).value; ctime = pd.Timestamp(ct).value
        d_idx = int(np.searchsorted(ts, dtime, side="left"))
        r_idx = int(np.searchsorted(ts, ctime, side="left"))
        scan_end = min(n - 2, r_idx + fill_buffer_bars)
        if d_idx >= scan_end:
            stats["no_fill"] += 1; continue

        # pullback extreme after the swing top (low for bull, high for bear)
        if bull:
            ext_idx = d_idx + int(np.argmin(Lo[d_idx:scan_end + 1]))
            pull_ext = float(Lo[ext_idx])
        else:
            ext_idx = d_idx + int(np.argmax(H[d_idx:scan_end + 1]))
            pull_ext = float(H[ext_idx])

        # FIRST re-touch of the level after the pullback extreme = the resting-stop fill
        fill_idx = None
        for k in range(ext_idx, scan_end + 1):
            if bull and H[k] >= weak:
                fill_idx = k; break
            if (not bull) and Lo[k] <= weak:
                fill_idx = k; break
        if fill_idx is None:
            stats["no_fill"] += 1; continue

        entry_px = weak
        buf = stop_buf_atr * float(atr5[fill_idx])
        stop_px = (pull_ext - buf) if bull else (pull_ext + buf)
        tgt = _target_price(bull, weak, strong, ctime, m15ts, m15H, m15L, dt15,
                            target, liq_lookback, major_lookback)
        risk = (entry_px - stop_px) if bull else (stop_px - entry_px)
        if (not np.isfinite(tgt) or not np.isfinite(risk) or risk <= 0
                or (bull and tgt <= entry_px) or ((not bull) and tgt >= entry_px)):
            stats["no_fill"] += 1; continue

        exit_kind, exit_px = "timeout", C[min(n - 1, fill_idx + max_bars_m5)]
        for k in range(fill_idx + 1, min(n, fill_idx + max_bars_m5)):
            if bull:
                if Lo[k] <= stop_px:
                    exit_kind, exit_px = "stop", stop_px; break
                if H[k] >= tgt:
                    exit_kind, exit_px = "target", tgt; break
            else:
                if H[k] >= stop_px:
                    exit_kind, exit_px = "stop", stop_px; break
                if Lo[k] <= tgt:
                    exit_kind, exit_px = "target", tgt; break
        if not np.isfinite(exit_px):
            stats["no_fill"] += 1; continue

        stats["filled"] += 1
        stats["confirmed_fill" if cont == 1 else "falsebreak_fill"] += 1
        trades.append(dict(entry_time=ts[fill_idx], direction="bull" if bull else "bear",
                           entry=entry_px, stop=stop_px, target=tgt, exit_kind=exit_kind,
                           exit=exit_px, risk_price=risk, confirmed=cont))
    trades.sort(key=lambda d: d["entry_time"])
    return trades, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtf-csv", required=True)
    ap.add_argument("--h4-dir", required=True)
    ap.add_argument("--m15-dir", required=True)
    ap.add_argument("--m5-dir", required=True)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--target", default="major_high")
    ap.add_argument("--stop-buf-atr", type=float, default=0.5)
    ap.add_argument("--max-bars-m5", type=int, default=480)
    ap.add_argument("--liq-lookback", type=int, default=96)
    ap.add_argument("--major-lookback", type=int, default=480)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--tick-value", type=float, default=0.01)
    ap.add_argument("--lot-step", type=float, default=0.01)
    ap.add_argument("--min-lot", type=float, default=0.01)
    ap.add_argument("--max-lot", type=float, default=100.0)
    ap.add_argument("--spread-price", type=float, default=2.0)
    ap.add_argument("--commission", type=float, default=0.0)
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    args = ap.parse_args()

    master = pd.read_csv(args.mtf_csv, low_memory=False)
    master["timestamp"] = L._to_dt(master["timestamp"])
    for c in ("open", "high", "low", "close"):
        master[c] = pd.to_numeric(master[c], errors="coerce")
    master = master.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    labels = L.build_all_labels(master, {"h4": args.h4_dir, "m15": args.m15_dir, "m5": args.m5_dir},
                                horizon=args.horizon)
    trades, stats = generate_resting_stop_trades(
        master, labels, target=args.target, stop_buf_atr=args.stop_buf_atr,
        max_bars_m5=args.max_bars_m5, liq_lookback=args.liq_lookback, major_lookback=args.major_lookback)

    if args.start_date or args.end_date:
        s = pd.Timestamp(args.start_date, tz="UTC").value if args.start_date else -(2**63)
        e = pd.Timestamp(args.end_date, tz="UTC").value if args.end_date else (2**63 - 1)
        trades = [t for t in trades if s <= t["entry_time"] <= e]

    print(f"\nRESTING-STOP fill model (spread {args.spread_price}, risk {args.risk_pct}%)")
    print(f"  episodes scanned : {stats['episodes']}")
    print(f"  stop FILLED      : {stats['filled']}   "
          f"(confirmed continuations: {stats['confirmed_fill']}  |  FALSE breaks: {stats['falsebreak_fill']})")
    print(f"  never filled     : {stats['no_fill']}")

    # win-rate split confirmed vs false-break (target=win)
    conf_w = sum(1 for t in trades if t["confirmed"] == 1 and t["exit_kind"] == "target")
    conf_n = sum(1 for t in trades if t["confirmed"] == 1)
    fb_w = sum(1 for t in trades if t["confirmed"] == 0 and t["exit_kind"] == "target")
    fb_n = sum(1 for t in trades if t["confirmed"] == 0)
    if conf_n:
        print(f"  confirmed fills win rate : {100*conf_w/conf_n:.1f}%  (n={conf_n})")
    if fb_n:
        print(f"  false-break fills win rate: {100*fb_w/fb_n:.1f}%  (n={fb_n})")

    AB.run_account(trades, tick_size=args.tick_size, tick_value=args.tick_value,
                   lot_step=args.lot_step, min_lot=args.min_lot, max_lot=args.max_lot,
                   spread_price=args.spread_price, commission=args.commission,
                   balance=args.balance, risk_pct=args.risk_pct)


if __name__ == "__main__":
    main()
