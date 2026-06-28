"""
account_backtest.py
====================
Turn the reactive confirmed-continuation trades into a REAL MT5-style account result:
position sizing off risk %, tick value and lot step, spread + commission, compounding,
and an equity curve with drawdown / profit factor.

You choose ONE setup (entry style + target). Each trade is sized so that hitting the
stop loses `--risk-pct` of the CURRENT balance (standard fixed-fractional sizing), then
P&L is computed in account currency from tick value and lots, net of spread + commission.

Works on ANY instrument: point it at that instrument's parser output + fused table and
pass its MT5 contract spec. (For a real instrument: run your MarketStructureParser on its
M5/M15/H4 OHLC, build the fused table with mtf_state_builder, then run this.)

Example (fill in YOUR instrument's MT5 spec):
  python account_backtest.py --mtf-csv mtf_state_table.csv \
      --h4-dir ... --m15-dir ... --m5-dir ... \
      --entry immediate --target major_high \
      --tick-size 0.01 --tick-value 0.01 --lot-step 0.01 --min-lot 0.001 --max-lot 50 \
      --spread-price 5 --commission 0 --balance 1000 --risk-pct 1.0
"""
from __future__ import annotations

import argparse
import pickle
import numpy as np
import pandas as pd

import smc_labeling as L


def _ns(series) -> np.ndarray:
    return (pd.to_datetime(series, utc=True, errors="coerce")
            .to_numpy().astype("datetime64[ns]").astype("int64"))


# --------------------------------------------------------------------------- #
# Build the chronological trade list for ONE (entry, target) configuration
# --------------------------------------------------------------------------- #
def _model_probs(conf, model_d):
    """Probability per continuation episode from a saved {model, features} pickle."""
    import numpy as _np
    feats = model_d.get("features") or []
    X = conf.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    m = model_d["model"]
    try:
        return m.predict_proba(X)[:, 1]
    except Exception:
        return _np.asarray(m.predict(X), dtype=float)


def generate_trades(master, labels, m5dir, *, entry="immediate", target="major_high",
                    stop_buf_atr=0.5, max_bars_m5=480, confirm_bars_m5=48,
                    liq_lookback=96, major_lookback=480, model_d=None, prob_threshold=0.5,
                    entry_timing="confirm_open"):
    pb = labels["m15_pullback"]
    conf = pb[pd.to_numeric(pb["continuation_occurred"], errors="coerce") == 1].copy()
    if model_d is not None and not conf.empty:           # optional model probability filter
        conf = conf.assign(model_prob=_model_probs(conf, model_d))
        before = len(conf)
        conf = conf[conf["model_prob"] >= prob_threshold]
        print(f"  model filter: {len(conf)}/{before} continuations pass prob >= {prob_threshold}")

    m5 = master[["timestamp", "open", "high", "low", "close"]].copy()
    ts = _ns(m5["timestamp"])
    H = m5["high"].to_numpy(float); Lo = m5["low"].to_numpy(float); C = m5["close"].to_numpy(float)
    n = len(ts)
    atr5 = pd.Series(H - Lo).rolling(14, min_periods=1).mean().to_numpy()

    m15 = L.resample_ohlc(master, "m15")
    m15ts = _ns(m15["timestamp"]); m15H = m15["high"].to_numpy(float); m15L = m15["low"].to_numpy(float)
    dt15 = int(np.median(np.diff(m15ts))) if len(m15ts) > 2 else 0
    dt5 = int(np.median(np.diff(ts))) if n > 2 else 0
    confirm_ns = confirm_bars_m5 * max(dt5, 1)

    mev = L.load_tf(m5dir).get("events", pd.DataFrame())
    def _evts(types):
        if mev.empty or "event_type" not in mev.columns:
            return np.array([], dtype="int64")
        a = _ns(mev[mev["event_type"].isin(types)]["timestamp"])
        return np.sort(a[a > 0])
    bull_ev, bear_ev = _evts(["bos_bull", "choch_bull"]), _evts(["bos_bear", "choch_bear"])

    def _target_price(bull, weak, strong, ctime):
        if target.startswith("mm"):
            mult = float(target[2:]); sign = 1.0 if bull else -1.0
            return weak + sign * mult * abs(weak - strong)
        lb = liq_lookback if target == "liq_high" else major_lookback
        mask = (m15ts >= ctime - lb * max(dt15, 1)) & (m15ts <= ctime)
        if bull:
            cand = m15H[mask]; cand = cand[cand > weak]
            if cand.size == 0:
                return np.nan
            return float(np.min(cand)) if target == "liq_high" else float(np.max(cand))
        else:
            cand = m15L[mask]; cand = cand[cand < weak]
            if cand.size == 0:
                return np.nan
            return float(np.max(cand)) if target == "liq_high" else float(np.min(cand))

    trades = []
    for r in conf.itertuples(index=False):
        bull = getattr(r, "direction", "bull") == "bull"
        weak = float(getattr(r, "ref_weak", np.nan)); strong = float(getattr(r, "ref_strong", np.nan))
        ct = getattr(r, "resolve_time"); dt = getattr(r, "decision_time")
        if not (np.isfinite(weak) and np.isfinite(strong)) or pd.isna(ct) or pd.isna(dt):
            continue
        ctime = pd.Timestamp(ct).value; dtime = pd.Timestamp(dt).value
        conf_idx = int(np.searchsorted(ts, ctime, side="left"))
        dec_idx = int(np.searchsorted(ts, dtime, side="left"))
        if conf_idx >= n - 2:
            continue
        tgt = _target_price(bull, weak, strong, ctime)
        if not np.isfinite(tgt):
            continue

        if entry == "immediate":
            e_idx = conf_idx
            ext = (np.min(Lo[dec_idx:conf_idx + 1]) if conf_idx > dec_idx else Lo[conf_idx]) if bull \
                else (np.max(H[dec_idx:conf_idx + 1]) if conf_idx > dec_idx else H[conf_idx])
        else:  # m5pullback
            evarr = bull_ev if bull else bear_ev
            j = int(np.searchsorted(evarr, ctime, side="left"))
            if j >= len(evarr) or evarr[j] > ctime + confirm_ns:
                continue
            e_idx = min(max(int(np.searchsorted(ts, evarr[j], side="left")), conf_idx), n - 1)
            ext = (np.min(Lo[conf_idx:e_idx + 1]) if e_idx > conf_idx else Lo[e_idx]) if bull \
                else (np.max(H[conf_idx:e_idx + 1]) if e_idx > conf_idx else H[e_idx])

        # --- entry FILL by timing model (realism axis) ---
        #   confirm_open  : M5 close at the M15 confirmation bar's OPEN  (ORIGINAL; optimistic
        #                   -- it fills ~10min before that bar closes and confirms)
        #   confirm_close : M5 close at the M15 confirmation bar's CLOSE (what the LIVE engine does)
        #   level         : resting stop/limit filled at the break level ref_weak (realistic best case)
        if entry_timing == "level":
            ex_idx = e_idx
            entry_px = float(weak)
        elif entry_timing == "confirm_close":
            ex_idx = int(np.searchsorted(ts, ctime + max(dt15, 0), side="left")) - 1
            ex_idx = min(max(ex_idx, e_idx), n - 1)
            entry_px = C[ex_idx]
        else:  # confirm_open (default — unchanged original behaviour)
            ex_idx = e_idx
            entry_px = C[e_idx]

        buf = stop_buf_atr * float(atr5[ex_idx])
        stop_px = (ext - buf) if bull else (ext + buf)
        risk = (entry_px - stop_px) if bull else (stop_px - entry_px)
        if (not np.isfinite(entry_px) or not np.isfinite(stop_px) or not np.isfinite(tgt)
                or not np.isfinite(risk) or risk <= 0
                or (bull and tgt <= entry_px) or ((not bull) and tgt >= entry_px)):
            continue

        # resolve on M5 (stop-first, conservative)
        exit_kind, exit_px = "timeout", C[min(n - 1, ex_idx + max_bars_m5)]
        for k in range(ex_idx + 1, min(n, ex_idx + max_bars_m5)):
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
            continue
        trades.append(dict(entry_time=ts[ex_idx], direction="bull" if bull else "bear",
                           entry=entry_px, stop=stop_px, target=tgt,
                           exit_kind=exit_kind, exit=exit_px, risk_price=risk))
    trades.sort(key=lambda d: d["entry_time"])
    return trades


# --------------------------------------------------------------------------- #
# Account simulation with MT5 contract spec
# --------------------------------------------------------------------------- #
def run_account(trades, *, tick_size, tick_value, lot_step, min_lot, max_lot,
                spread_price, commission, balance, risk_pct):
    bal = balance
    peak = balance
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    wins = losses = scratches = 0
    skipped = 0
    rows = []
    for t in trades:
        sign = 1.0 if t["direction"] == "bull" else -1.0
        risk_price = t["risk_price"]
        risk_amt = bal * (risk_pct / 100.0)
        ticks_risk = risk_price / tick_size
        loss_per_lot = ticks_risk * tick_value          # $ lost per 1.0 lot if stopped
        if loss_per_lot <= 0:
            skipped += 1; continue
        raw_lots = risk_amt / loss_per_lot
        lots = np.floor(raw_lots / lot_step) * lot_step
        lots = min(max(lots, 0.0), max_lot)
        if not np.isfinite(lots) or lots < min_lot:
            skipped += 1; continue
        # spread charged on entry; commission round-turn per lot
        gross = sign * (t["exit"] - t["entry"]) / tick_size * tick_value * lots
        cost = (spread_price / tick_size) * tick_value * lots + commission * lots
        pnl = gross - cost
        if not np.isfinite(pnl):
            skipped += 1; continue
        bal += pnl
        peak = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak if peak > 0 else 0.0)
        if pnl > 0:
            gross_win += pnl; wins += 1
        elif pnl < 0:
            gross_loss += -pnl; losses += 1
        else:
            scratches += 1
        rows.append(dict(entry_time=t["entry_time"], direction=t["direction"],
                         lots=round(lots, 4), exit_kind=t["exit_kind"],
                         pnl=round(pnl, 2), balance=round(bal, 2)))
        if bal <= 0:
            print("  *** ACCOUNT BLEW UP (balance <= 0) ***")
            break

    eq = pd.DataFrame(rows)
    n = len(eq)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    ret = (bal / balance - 1.0) * 100.0
    print(f"\n{'='*54}\nACCOUNT RESULT\n{'='*54}")
    print(f"  trades taken      : {n}   (skipped/unsizable: {skipped})")
    print(f"  start balance     : {balance:,.2f}")
    print(f"  final balance     : {bal:,.2f}")
    print(f"  total return      : {ret:+.1f}%")
    print(f"  max drawdown      : {max_dd*100:.1f}%")
    print(f"  win / loss / scr  : {wins} / {losses} / {scratches}")
    print(f"  win rate          : {(100*wins/n):.1f}%" if n else "  win rate          : -")
    print(f"  profit factor     : {pf:.2f}")
    if n:
        ts0 = pd.Timestamp(eq['entry_time'].iloc[0]); ts1 = pd.Timestamp(eq['entry_time'].iloc[-1])
        print(f"  period            : {ts0.date()} -> {ts1.date()}")
        print(f"  avg P&L / trade   : {(bal-balance)/n:,.2f}")
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtf-csv", required=True)
    ap.add_argument("--h4-dir", required=True)
    ap.add_argument("--m15-dir", required=True)
    ap.add_argument("--m5-dir", required=True)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--entry", choices=["immediate", "m5pullback"], default="immediate")
    ap.add_argument("--entry-timing", choices=["confirm_open", "confirm_close", "level"],
                    default="confirm_open",
                    help="fill model: confirm_open=orig (optimistic) | confirm_close=at M15 bar "
                         "close (what live does) | level=resting order at the break level (realistic)")
    ap.add_argument("--target", default="major_high",
                    help="mm1.0 | mm1.5 | mm2.0 | liq_high | major_high")
    ap.add_argument("--stop-buf-atr", type=float, default=0.5)
    ap.add_argument("--max-bars-m5", type=int, default=480)
    ap.add_argument("--confirm-bars-m5", type=int, default=48)
    # ---- MT5 contract spec (FILL IN for your instrument) ----
    ap.add_argument("--tick-size", type=float, required=True, help="minimum price increment")
    ap.add_argument("--tick-value", type=float, required=True, help="account-currency value of 1 tick per 1.0 lot")
    ap.add_argument("--lot-step", type=float, default=0.01)
    ap.add_argument("--min-lot", type=float, default=0.01)
    ap.add_argument("--max-lot", type=float, default=100.0)
    ap.add_argument("--spread-price", type=float, default=0.0, help="spread in PRICE units")
    ap.add_argument("--commission", type=float, default=0.0, help="round-turn commission per 1.0 lot")
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--start-date", default=None, help="only trade on/after this date, e.g. 2024-01-01")
    ap.add_argument("--end-date", default=None, help="only trade on/before this date, e.g. 2025-12-31")
    ap.add_argument("--out-csv", default="equity_curve.csv")
    ap.add_argument("--model", default=None,
                    help="saved model .pkl from smc_train to gate entries by probability (omit = pure structure)")
    ap.add_argument("--prob-threshold", type=float, default=0.5,
                    help="with --model, only take continuations the model rates >= this")
    args = ap.parse_args()

    master = pd.read_csv(args.mtf_csv, low_memory=False)
    master["timestamp"] = L._to_dt(master["timestamp"])
    master = master.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    pdirs = {"h4": args.h4_dir, "m15": args.m15_dir, "m5": args.m5_dir}
    labels = L.build_all_labels(master, pdirs, horizon=args.horizon)

    model_d = None
    if args.model:
        with open(args.model, "rb") as fh:
            model_d = pickle.load(fh)
        print(f"  MODEL FILTER ON: {args.model}  (prob >= {args.prob_threshold})")
    else:
        print("  MODEL FILTER OFF (pure structure)")
    print(f"Setup: entry={args.entry}  entry_timing={args.entry_timing}  target={args.target}  "
          f"risk={args.risk_pct}%  tick_size={args.tick_size}  tick_value={args.tick_value}  "
          f"spread={args.spread_price}")
    trades = generate_trades(master, labels, args.m5_dir, entry=args.entry, target=args.target,
                             stop_buf_atr=args.stop_buf_atr, max_bars_m5=args.max_bars_m5,
                             confirm_bars_m5=args.confirm_bars_m5,
                             model_d=model_d, prob_threshold=args.prob_threshold,
                             entry_timing=args.entry_timing)
    print(f"generated {len(trades)} trades")
    if args.start_date or args.end_date:
        s = pd.Timestamp(args.start_date, tz="UTC").value if args.start_date else -(2 ** 63)
        e = pd.Timestamp(args.end_date, tz="UTC").value if args.end_date else (2 ** 63 - 1)
        trades = [t for t in trades if s <= t["entry_time"] <= e]
        print(f"date filter [{args.start_date} .. {args.end_date}] -> {len(trades)} trades")
    eq = run_account(trades, tick_size=args.tick_size, tick_value=args.tick_value,
                     lot_step=args.lot_step, min_lot=args.min_lot, max_lot=args.max_lot,
                     spread_price=args.spread_price, commission=args.commission,
                     balance=args.balance, risk_pct=args.risk_pct)
    if not eq.empty:
        eq.to_csv(args.out_csv, index=False)
        print(f"\nequity curve -> {args.out_csv}")


if __name__ == "__main__":
    main()
