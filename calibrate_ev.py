"""
calibrate_ev.py
===============
Turn the M15 continuation-occurrence model into a money decision.

Pipeline
--------
1. Out-of-sample probabilities via the SAME purged walk-forward CV (no leakage):
   each test fold is scored by a model trained only on its strictly-past data.
2. (Optional) isotonic calibration so the probability means what it says.
3. A concrete TRADE per episode, using your structure:
      entry  = deep in the pullback (a configurable retrace toward the strong anchor,
               i.e. in discount for longs / premium for shorts)
      stop   = just beyond the strong low/high  (ref_strong) + an ATR buffer
               -> this is the CHOCH/invalidation level, the same event the label calls
                  a "reversal"
      target = entry +/- target_R * risk          (risk = |entry - stop|)
   The trade is simulated bar-by-bar on real M15 OHLC: it only counts if price
   actually retraces to the entry, then we see whether target or stop is hit first.
4. Expected value (in R) by probability threshold and target multiple, so you can
   read off where to act.

Usage:
  python calibrate_ev.py --mtf-csv mtf_state_table.csv \
      --h4-dir ... --m15-dir ... --m5-dir ... \
      --entry-frac 0.6 --stop-buf-atr 0.5 --max-bars 160
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import smc_labeling as L
import smc_models as M

try:
    from sklearn.isotonic import IsotonicRegression
    _HAVE_ISO = True
except Exception:
    _HAVE_ISO = False


# --------------------------------------------------------------------------- #
# 1. Out-of-sample probabilities (purged walk-forward)
# --------------------------------------------------------------------------- #
def oof_probabilities(df: pd.DataFrame, target: str, n_splits: int = 6):
    work = df[pd.to_numeric(df[target], errors="coerce").notna()].reset_index(drop=True)
    feats = M.select_feature_columns(work, extra_drop=[target])
    X, y = M._Xy(work, feats, target)
    t = pd.to_datetime(work["decision_time"], errors="coerce", utc=True).astype("int64").to_numpy()
    if "resolve_time" in work.columns:
        rt = pd.to_datetime(work["resolve_time"], errors="coerce", utc=True).astype("int64").to_numpy()
        rt = np.where(rt > t, rt, t)
    else:
        rt = t.copy()
    folds = M.purged_walkforward_folds(t, rt, n_splits=n_splits)
    oof = np.full(len(work), np.nan)
    for tr, te in folds:
        ytr = y[tr]
        if len(np.unique(ytr[np.isfinite(ytr)])) < 2:
            continue
        med = np.nanmedian(X[tr], axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med)
        Xte = np.where(np.isfinite(X[te]), X[te], med)
        mdl = M.make_classifier()
        mdl.fit(Xtr, np.nan_to_num(ytr))
        oof[te] = mdl.predict_proba(Xte)[:, 1]
    work["oof_prob"] = oof
    # isotonic calibration (fit on scored OOF; reported as cal_prob)
    if _HAVE_ISO:
        m = np.isfinite(oof)
        if m.sum() > 50 and len(np.unique(y[m])) == 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof[m], y[m])
            cal = np.full(len(work), np.nan)
            cal[m] = iso.predict(oof[m])
            work["cal_prob"] = cal
    if "cal_prob" not in work.columns:
        work["cal_prob"] = work["oof_prob"]
    return work


def reliability_table(work: pd.DataFrame, prob_col="cal_prob", target="continuation_occurred",
                      bins=5):
    w = work[np.isfinite(work[prob_col])].copy()
    if w.empty:
        return
    w["bin"] = pd.qcut(w[prob_col], bins, labels=False, duplicates="drop")
    print(f"\nCalibration ({prob_col}):  predicted vs actual continuation rate")
    print(f"{'bin':>3} {'pred':>7} {'actual':>7} {'n':>6}")
    for b, g in w.groupby("bin"):
        print(f"{int(b):>3} {g[prob_col].mean():7.3f} {g[target].mean():7.3f} {len(g):>6}")


# --------------------------------------------------------------------------- #
# 2. Trade simulation on real M15 OHLC
# --------------------------------------------------------------------------- #
def simulate(work: pd.DataFrame, ohlc: pd.DataFrame, *, entry_frac=0.6,
             stop_buf_atr=0.5, target_R=2.0, max_bars=160):
    ts = pd.to_datetime(ohlc["timestamp"], utc=True).astype("int64").to_numpy()
    O = ohlc["open"].to_numpy(float); H = ohlc["high"].to_numpy(float)
    Lo = ohlc["low"].to_numpy(float); C = ohlc["close"].to_numpy(float)
    n = len(ts)

    dec = pd.to_datetime(work["decision_time"], utc=True).astype("int64").to_numpy()
    outcome = np.full(len(work), np.nan)   # R multiple; NaN = no trade (never filled / bad geometry)
    filled = np.zeros(len(work), dtype=bool)

    for i, row in enumerate(work.itertuples(index=False)):
        direction = getattr(row, "direction", "bull")
        strong = float(getattr(row, "ref_strong", np.nan))
        weak = float(getattr(row, "ref_weak", np.nan))
        atrv = float(getattr(row, "atr_at_decision", np.nan))
        if not (np.isfinite(strong) and np.isfinite(weak) and np.isfinite(atrv)):
            continue
        buf = stop_buf_atr * atrv
        if direction == "bull":
            entry = weak - entry_frac * (weak - strong)
            stop = strong - buf
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + target_R * risk
        else:
            entry = weak + entry_frac * (strong - weak)
            stop = strong + buf
            risk = stop - entry
            if risk <= 0:
                continue
            target = entry - target_R * risk

        start = int(np.searchsorted(ts, dec[i], side="left"))
        end = min(n, start + max_bars)
        f = False
        res = np.nan
        for k in range(start, end):
            if not f:
                if direction == "bull" and Lo[k] <= entry:
                    f = True
                elif direction == "bear" and H[k] >= entry:
                    f = True
                else:
                    # invalidate if structure breaks before entry
                    if direction == "bull" and C[k] < stop:
                        break
                    if direction == "bear" and C[k] > stop:
                        break
                    continue
            # filled -> check stop first (conservative), then target
            if direction == "bull":
                if Lo[k] <= stop:
                    res = -1.0; break
                if H[k] >= target:
                    res = target_R; break
            else:
                if H[k] >= stop:
                    res = -1.0; break
                if Lo[k] <= target:
                    res = target_R; break
        if f:
            filled[i] = True
            outcome[i] = res if np.isfinite(res) else 0.0   # timeout -> scratch (0R)
    work = work.copy()
    work[f"out_{target_R:g}R"] = outcome
    work[f"filled"] = filled
    return work


# --------------------------------------------------------------------------- #
# 2b. MTF trade: M15 zone -> M5 confirmation entry -> stop below the M5 sweep low
# --------------------------------------------------------------------------- #
def simulate_mtf(work: pd.DataFrame, m5: pd.DataFrame,
                 bull_ev_ts: np.ndarray, bear_ev_ts: np.ndarray, *,
                 entry_frac=0.6, stop_buf_atr=0.5, target_R=2.0,
                 max_bars_m5=480, confirm_bars_m5=48):
    """
    Drop to M5 for entry + stop:
      1. wait for price to reach the M15 discount/premium zone (entry_frac of the leg);
      2. enter on the first M5 continuation event (bos/choch in trade direction) within
         `confirm_bars_m5` bars -> the sweep low is now in the past;
      3. stop = lowest M5 low between the zone touch and the confirmation, minus an
         ATR(M5) buffer  (i.e. below the actual sweep, not below the M15 strong low);
      4. target = entry +/- target_R * risk, simulated bar-by-bar on M5.
    """
    ts = pd.to_datetime(m5["timestamp"], utc=True).astype("int64").to_numpy()
    H = m5["high"].to_numpy(float); Lo = m5["low"].to_numpy(float)
    C = m5["close"].to_numpy(float)
    n = len(ts)
    dt = int(np.median(np.diff(ts))) if n > 2 else 0
    confirm_ns = confirm_bars_m5 * max(dt, 1)
    # rough M5 ATR for the stop buffer (range-based, cheap)
    atr5 = pd.Series(H - Lo).rolling(14, min_periods=1).mean().to_numpy()

    dec = pd.to_datetime(work["decision_time"], utc=True).astype("int64").to_numpy()
    outcome = np.full(len(work), np.nan)
    filled = np.zeros(len(work), dtype=bool)

    for i, row in enumerate(work.itertuples(index=False)):
        direction = getattr(row, "direction", "bull")
        strong = float(getattr(row, "ref_strong", np.nan))
        weak = float(getattr(row, "ref_weak", np.nan))
        if not (np.isfinite(strong) and np.isfinite(weak)):
            continue
        bull = direction == "bull"
        zone = weak - entry_frac * (weak - strong) if bull else weak + entry_frac * (strong - weak)
        start = int(np.searchsorted(ts, dec[i], side="left"))
        end = min(n, start + max_bars_m5)
        # 1) reach the zone
        touch = -1
        for k in range(start, end):
            if (bull and Lo[k] <= zone) or ((not bull) and H[k] >= zone):
                touch = k; break
        if touch < 0:
            continue
        # 2) M5 confirmation within the window
        evarr = bull_ev_ts if bull else bear_ev_ts
        lo_t, hi_t = ts[touch], ts[touch] + confirm_ns
        j = int(np.searchsorted(evarr, lo_t, side="left"))
        if j >= len(evarr) or evarr[j] > hi_t:
            continue
        entry_k = int(np.searchsorted(ts, evarr[j], side="left"))
        entry_k = min(max(entry_k, touch), n - 1)
        entry = C[entry_k]
        buf = stop_buf_atr * float(atr5[entry_k])
        if bull:
            sweep = float(np.min(Lo[touch:entry_k + 1]))
            stop = sweep - buf
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + target_R * risk
        else:
            sweep = float(np.max(H[touch:entry_k + 1]))
            stop = sweep + buf
            risk = stop - entry
            if risk <= 0:
                continue
            target = entry - target_R * risk
        # 3) resolve on M5
        res = np.nan
        for k in range(entry_k + 1, min(n, entry_k + max_bars_m5)):
            if bull:
                if Lo[k] <= stop:
                    res = -1.0; break
                if H[k] >= target:
                    res = target_R; break
            else:
                if H[k] >= stop:
                    res = -1.0; break
                if Lo[k] <= target:
                    res = target_R; break
        filled[i] = True
        outcome[i] = res if np.isfinite(res) else 0.0
    work = work.copy()
    work[f"out_{target_R:g}R"] = outcome
    work["filled"] = filled
    return work


# --------------------------------------------------------------------------- #
# 3. EV by threshold
# --------------------------------------------------------------------------- #
def ev_table(work: pd.DataFrame, prob_col, out_col, target_R,
             thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75)):
    w = work[np.isfinite(work[prob_col]) & np.isfinite(work[out_col])]
    total = len(w)
    print(f"\nEV @ target {target_R:g}R   (prob={prob_col}, n_scored_filled={total})")
    print(f"{'thresh':>7} {'trades':>7} {'win%':>6} {'scratch%':>8} "
          f"{'EV(R)':>7} {'totalR':>8}")
    for th in thresholds:
        s = w[w[prob_col] >= th]
        if len(s) == 0:
            print(f"{th:7.2f} {0:>7}")
            continue
        out = s[out_col].to_numpy()
        win = float(np.mean(out == target_R)) * 100
        scr = float(np.mean(out == 0.0)) * 100
        ev = float(np.mean(out))
        print(f"{th:7.2f} {len(s):>7} {win:6.1f} {scr:8.1f} {ev:7.3f} {out.sum():8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtf-csv", required=True)
    ap.add_argument("--h4-dir", required=True)
    ap.add_argument("--m15-dir", required=True)
    ap.add_argument("--m5-dir", required=True)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--entry-frac", type=float, default=0.6,
                    help="retrace fraction toward the strong anchor for entry (0=at weak, 1=at strong)")
    ap.add_argument("--stop-buf-atr", type=float, default=0.5,
                    help="ATR multiple added beyond the strong anchor for the stop")
    ap.add_argument("--max-bars", type=int, default=160, help="max M15 bars to resolve a trade")
    ap.add_argument("--confirm-bars-m5", type=int, default=48,
                    help="M5 bars after reaching the zone to wait for a confirmation entry")
    ap.add_argument("--targets", default="2,3", help="comma list of target R multiples")
    args = ap.parse_args()

    master = pd.read_csv(args.mtf_csv, low_memory=False)
    master["timestamp"] = L._to_dt(master["timestamp"])
    master = master.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    pdirs = {"h4": args.h4_dir, "m15": args.m15_dir, "m5": args.m5_dir}
    labels = L.build_all_labels(master, pdirs, horizon=args.horizon)
    pb = labels.get("m15_pullback")
    if pb is None or pb.empty:
        raise SystemExit("No m15 pullback episodes.")

    print(f"ML backend: {M.BACKEND} | isotonic calibration: {_HAVE_ISO}")
    work = oof_probabilities(pb, "continuation_occurred", n_splits=args.n_splits)
    reliability_table(work, "cal_prob")

    m15_ohlc = L.resample_ohlc(master, "m15")
    # M5 data for the MTF variant
    m5_ohlc = master[["timestamp", "open", "high", "low", "close"]].copy()
    m5_ev = L.load_tf(args.m5_dir).get("events", pd.DataFrame())
    def _ev_ts(types):
        if m5_ev.empty or "event_type" not in m5_ev.columns:
            return np.array([], dtype="int64")
        s = m5_ev[m5_ev["event_type"].isin(types)]["timestamp"]
        return np.sort(pd.to_datetime(s, utc=True, errors="coerce").dropna().astype("int64").to_numpy())
    bull_ev = _ev_ts(["bos_bull", "choch_bull"])
    bear_ev = _ev_ts(["bos_bear", "choch_bear"])
    print(f"M5 confirmation events: {len(bull_ev)} bull, {len(bear_ev)} bear")

    for tR in [float(x) for x in args.targets.split(",")]:
        print(f"\n================  TARGET {tR:g}R  ================")
        # A) baseline: M15 strong-low stop, limit entry at the retrace
        simA = simulate(work, m15_ohlc, entry_frac=args.entry_frac,
                        stop_buf_atr=args.stop_buf_atr, target_R=tR, max_bars=args.max_bars)
        print(f"[A] M15 strong-low stop, retrace entry  | fill={100*simA['filled'].mean():.0f}%")
        ev_table(simA, "cal_prob", f"out_{tR:g}R", tR)
        # B) MTF: M5 confirmation entry + stop below the M5 sweep low
        simB = simulate_mtf(work, m5_ohlc, bull_ev, bear_ev, entry_frac=args.entry_frac,
                            stop_buf_atr=args.stop_buf_atr, target_R=tR,
                            max_bars_m5=args.max_bars * 3, confirm_bars_m5=args.confirm_bars_m5)
        print(f"[B] M5 sweep-low stop, M5-confirm entry | fill={100*simB['filled'].mean():.0f}%")
        ev_table(simB, "cal_prob", f"out_{tR:g}R", tR)

    print("\nNotes: EV is per-trade in R units (EV>0 = positive expectancy). [A] stops at "
          "the M15 strong low (gets swept); [B] enters on M5 confirmation after the sweep "
          "and stops below the M5 sweep low. Compare the two.")


if __name__ == "__main__":
    main()
