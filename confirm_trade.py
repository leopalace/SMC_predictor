"""
confirm_trade.py
================
A REACTIVE backtest of your idea: stop predicting continuation, wait for the M15 to
CONFIRM it, drop to M5 for a tight entry, and aim for the M15 sweep (liquidity).

For every pullback whose continuation actually occurred (price closed back past the
originating weak high = the confirmation), we:

  ENTRY (two styles, compared):
    immediate  : enter at the M5 close on the M15 confirmation bar;
                 stop = below the M5 sweep low (lowest M5 low during the pullback).
    m5pullback : after confirmation, wait for a small M5 pullback then an M5 bos/choch
                 in trade direction; enter there; stop = below that M5 pullback low.
                 (tighter stop, better R:R, fewer fills.)

  TARGETS (five, compared):
    mm1.0 / mm1.5 / mm2.0 : weak_high +/- N * leg          (leg = |weak-strong|)
    liq_high              : nearest M15 swing high beyond the weak high (resting
                            liquidity to sweep)  -- "the M15 sweep"
    major_high            : the dominant M15 high over a longer lookback (the level the
                            original move came from / opposing extreme)

  RESOLUTION: simulate forward on M5; for each target record +R if it's reached before
  the stop, -1 if stopped first, 0 if it times out. R = |target-entry| / |entry-stop|.

No model is used here -- it's pure structure. (You can later layer the range filter on
top to skip the chop.)

Usage:
  python confirm_trade.py --mtf-csv mtf_state_table.csv \
      --h4-dir ... --m15-dir ... --m5-dir ... \
      --stop-buf-atr 0.5 --max-bars-m5 480 --confirm-bars-m5 48
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import smc_labeling as L

TARGET_NAMES = ["mm1.0", "mm1.5", "mm2.0", "liq_high", "major_high"]


def _ns(series) -> np.ndarray:
    """Timestamps -> int64 NANOSECONDS, regardless of the datetime resolution pandas
    chose (us/ms/ns). Mixing .astype('int64') (unit-dependent) with Timestamp.value
    (always ns) is what caused the 1000x clock mismatch."""
    return (pd.to_datetime(series, utc=True, errors="coerce")
            .to_numpy().astype("datetime64[ns]").astype("int64"))


def _ev_table(rows: list, label: str):
    if not rows:
        print(f"  {label}: no trades")
        return
    df = pd.DataFrame(rows)
    print(f"\n{label}   (trades filled = {len(df)})")
    print(f"{'target':>10} {'n':>5} {'win%':>6} {'scratch%':>9} {'avgR_to_tgt':>11} "
          f"{'EV(R)':>7} {'totalR':>8}")
    for t in TARGET_NAMES:
        col = f"out_{t}"
        rr = f"rmult_{t}"
        s = df[np.isfinite(df[col])]
        if s.empty:
            print(f"{t:>10} {0:>5}")
            continue
        out = s[col].to_numpy()
        win = float(np.mean(out > 0)) * 100
        scr = float(np.mean(out == 0)) * 100
        ev = float(np.mean(out))
        avgr = float(np.nanmean(s[rr]))
        print(f"{t:>10} {len(s):>5} {win:6.1f} {scr:9.1f} {avgr:11.2f} {ev:7.3f} {out.sum():8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtf-csv", required=True)
    ap.add_argument("--h4-dir", required=True)
    ap.add_argument("--m15-dir", required=True)
    ap.add_argument("--m5-dir", required=True)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--stop-buf-atr", type=float, default=0.5)
    ap.add_argument("--max-bars-m5", type=int, default=480)
    ap.add_argument("--confirm-bars-m5", type=int, default=48)
    ap.add_argument("--cost-price", type=float, default=0.0,
                    help="round-trip cost (spread+slippage) in PRICE units, charged per trade")
    ap.add_argument("--liq-lookback-m15", type=int, default=96)
    ap.add_argument("--major-lookback-m15", type=int, default=480)
    args = ap.parse_args()

    master = pd.read_csv(args.mtf_csv, low_memory=False)
    master["timestamp"] = L._to_dt(master["timestamp"])
    master = master.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    pdirs = {"h4": args.h4_dir, "m15": args.m15_dir, "m5": args.m5_dir}
    labels = L.build_all_labels(master, pdirs, horizon=args.horizon)
    pb = labels.get("m15_pullback")
    if pb is None or pb.empty:
        raise SystemExit("No m15 pullback episodes.")
    # ONLY confirmed continuations
    conf = pb[pd.to_numeric(pb["continuation_occurred"], errors="coerce") == 1].copy()
    print(f"confirmed M15 continuations: {len(conf)} / {len(pb)} pullbacks")

    # M5 arrays
    m5 = master[["timestamp", "open", "high", "low", "close"]].copy()
    m5ts = _ns(m5["timestamp"])
    H5 = m5["high"].to_numpy(float); L5 = m5["low"].to_numpy(float); C5 = m5["close"].to_numpy(float)
    n5 = len(m5ts)
    atr5 = pd.Series(H5 - L5).rolling(14, min_periods=1).mean().to_numpy()

    # M15 arrays for liquidity targets
    m15 = L.resample_ohlc(master, "m15")
    m15ts = _ns(m15["timestamp"])
    m15H = m15["high"].to_numpy(float); m15L = m15["low"].to_numpy(float)
    dt15 = int(np.median(np.diff(m15ts))) if len(m15ts) > 2 else 0

    # M5 confirmation events
    mev = L.load_tf(args.m5_dir).get("events", pd.DataFrame())
    def _evts(types):
        if mev.empty or "event_type" not in mev.columns:
            return np.array([], dtype="int64")
        s = mev[mev["event_type"].isin(types)]["timestamp"]
        arr = _ns(s)
        return np.sort(arr[arr > 0])
    bull_ev, bear_ev = _evts(["bos_bull", "choch_bull"]), _evts(["bos_bear", "choch_bear"])

    dt5 = int(np.median(np.diff(m5ts))) if n5 > 2 else 0
    confirm_ns = args.confirm_bars_m5 * max(dt5, 1)

    rows_imm, rows_pb = [], []
    dbg = {"ref_or_time": 0, "leg": 0, "conf_idx": 0, "ok": 0}
    _sampled = False
    for r in conf.itertuples(index=False):
        direction = getattr(r, "direction", "bull")
        bull = direction == "bull"
        weak = float(getattr(r, "ref_weak", np.nan))
        strong = float(getattr(r, "ref_strong", np.nan))
        ctime = pd.Timestamp(getattr(r, "resolve_time")).value if pd.notna(getattr(r, "resolve_time")) else None
        dtime = pd.Timestamp(getattr(r, "decision_time")).value if pd.notna(getattr(r, "decision_time")) else None
        if not (np.isfinite(weak) and np.isfinite(strong)) or ctime is None or dtime is None:
            dbg["ref_or_time"] += 1; continue
        leg = abs(weak - strong)
        if leg <= 0:
            dbg["leg"] += 1; continue

        # ---- targets ----
        tgts = {}
        sign = 1.0 if bull else -1.0
        tgts["mm1.0"] = weak + sign * 1.0 * leg
        tgts["mm1.5"] = weak + sign * 1.5 * leg
        tgts["mm2.0"] = weak + sign * 2.0 * leg
        # liquidity highs from M15 within lookback before confirmation
        lo15 = ctime - args.liq_lookback_m15 * max(dt15, 1)
        mask = (m15ts >= lo15) & (m15ts <= ctime)
        if bull:
            cand = m15H[mask]; cand = cand[cand > weak]
            tgts["liq_high"] = float(np.min(cand)) if cand.size else np.nan
            lo15b = ctime - args.major_lookback_m15 * max(dt15, 1)
            mb = (m15ts >= lo15b) & (m15ts <= ctime)
            cb = m15H[mb]; cb = cb[cb > weak]
            tgts["major_high"] = float(np.max(cb)) if cb.size else np.nan
        else:
            cand = m15L[mask]; cand = cand[cand < weak]
            tgts["liq_high"] = float(np.max(cand)) if cand.size else np.nan
            lo15b = ctime - args.major_lookback_m15 * max(dt15, 1)
            mb = (m15ts >= lo15b) & (m15ts <= ctime)
            cb = m15L[mb]; cb = cb[cb < weak]
            tgts["major_high"] = float(np.min(cb)) if cb.size else np.nan

        conf_idx = int(np.searchsorted(m5ts, ctime, side="left"))
        dec_idx = int(np.searchsorted(m5ts, dtime, side="left"))
        if not _sampled:
            print(f"[debug] sample dir={direction} weak={weak:.2f} strong={strong:.2f} "
                  f"leg={leg:.4f} conf_idx={conf_idx} dec_idx={dec_idx} n5={n5}  "
                  f"ctime={ctime} dtime={dtime}  m5ts0={m5ts[0]} m5tsN={m5ts[-1]}")
            _sampled = True
        if conf_idx >= n5 - 2:
            dbg["conf_idx"] += 1; continue
        dbg["ok"] += 1

        # ---- entry A: immediate ----
        entryA = C5[conf_idx]
        sweep_lo = np.min(L5[dec_idx:conf_idx + 1]) if conf_idx > dec_idx else L5[conf_idx]
        sweep_hi = np.max(H5[dec_idx:conf_idx + 1]) if conf_idx > dec_idx else H5[conf_idx]
        bufA = args.stop_buf_atr * float(atr5[conf_idx])
        stopA = (sweep_lo - bufA) if bull else (sweep_hi + bufA)
        rows_imm.append(_resolve(entryA, stopA, tgts, conf_idx + 1, bull,
                                 m5ts, H5, L5, n5, args.max_bars_m5, args.cost_price))

        # ---- entry B: M5 pullback + confirm ----
        evarr = bull_ev if bull else bear_ev
        j = int(np.searchsorted(evarr, ctime, side="left"))
        if j < len(evarr) and evarr[j] <= ctime + confirm_ns:
            eidx = int(np.searchsorted(m5ts, evarr[j], side="left"))
            eidx = min(max(eidx, conf_idx), n5 - 1)
            entryB = C5[eidx]
            pb_lo = np.min(L5[conf_idx:eidx + 1]) if eidx > conf_idx else L5[eidx]
            pb_hi = np.max(H5[conf_idx:eidx + 1]) if eidx > conf_idx else H5[eidx]
            bufB = args.stop_buf_atr * float(atr5[eidx])
            stopB = (pb_lo - bufB) if bull else (pb_hi + bufB)
            rows_pb.append(_resolve(entryB, stopB, tgts, eidx + 1, bull,
                                    m5ts, H5, L5, n5, args.max_bars_m5, args.cost_price))

    print(f"\n[debug] skip reasons: {dbg}  (filled A={len(rows_imm)}, B={len(rows_pb)})")
    print("\n=== ENTRY A: immediate at M15 confirmation (stop below M5 sweep low) ===")
    _ev_table(rows_imm, "[A] immediate")
    print("\n=== ENTRY B: M5 pullback + confirm (stop below M5 pullback low) ===")
    _ev_table(rows_pb, "[B] m5pullback")
    print("\nNotes: EV is per-trade in R. 'avgR_to_tgt' is the reward:risk that target "
          "offers from the entry (bigger target = bigger R but lower hit rate). Compare "
          "entries and targets; the best cell is high EV with enough trades.")


def _resolve(entry, stop, tgts, start, bull, m5ts, H5, L5, n5, max_bars, cost_price=0.0):
    risk = (entry - stop) if bull else (stop - entry)
    out = {}
    # precompute R multiple per target
    for t, tp in tgts.items():
        if not np.isfinite(tp) or risk <= 0:
            out[f"rmult_{t}"] = np.nan
            out[f"out_{t}"] = np.nan
            continue
        rmult = ((tp - entry) / risk) if bull else ((entry - tp) / risk)
        out[f"rmult_{t}"] = rmult if rmult > 0 else np.nan
        out[f"out_{t}"] = np.nan if rmult <= 0 else 0.0  # default scratch
    end = min(n5, start + max_bars)
    pending = {t: (np.isfinite(tgts[t]) and risk > 0 and
                   (((tgts[t] - entry) > 0) if bull else ((entry - tgts[t]) > 0)))
               for t in tgts}
    for k in range(start, end):
        # CONSERVATIVE: a bar that touches the stop is a loss for every target still
        # open, even if the bar's range also reaches the target (intrabar order unknown).
        hit_stop = (L5[k] <= stop) if bull else (H5[k] >= stop)
        if hit_stop:
            for t in tgts:
                if pending[t]:
                    out[f"out_{t}"] = -1.0; pending[t] = False
            break
        if bull:
            for t, tp in tgts.items():
                if pending[t] and H5[k] >= tp:
                    out[f"out_{t}"] = out[f"rmult_{t}"]; pending[t] = False
        else:
            for t, tp in tgts.items():
                if pending[t] and L5[k] <= tp:
                    out[f"out_{t}"] = out[f"rmult_{t}"]; pending[t] = False
    # subtract round-trip cost (spread+slippage in price) as a fraction of risk
    if cost_price and risk > 0:
        cr = cost_price / risk
        for t in tgts:
            v = out.get(f"out_{t}", np.nan)
            if np.isfinite(v):
                out[f"out_{t}"] = v - cr
    return out


if __name__ == "__main__":
    main()
