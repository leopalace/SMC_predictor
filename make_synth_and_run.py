"""
make_synth_and_run.py
=====================
End-to-end smoke test / demo on synthetic 'Volatility-75-like' data.

Pipeline:
  1. simulate 1-minute OHLC with regime-switching drift (creates trends + pullbacks)
  2. resample -> M5 / M15 / H4
  3. run MarketStructureParser on each TF  -> parser_out/{tf}/{states,events,pois}.csv
  4. mtf_state_builder.build_mtf_state_table -> fused master table
  5. smc_train.train_all -> labels + baselines + ML models + report

Swap step 1/2 for your real V75 data and the rest is unchanged.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF
import smc_train


def simulate_1min(n_minutes: int = 110_000, seed: int = 7, start: float = 100_000.0):
    rng = np.random.default_rng(seed)
    # regime-switching drift to manufacture trends and pullbacks
    drift = 0.0
    drifts = np.zeros(n_minutes)
    vol = 0.0009
    for i in range(n_minutes):
        if rng.random() < 0.004:                 # ~ switch every 250 bars
            drift = rng.normal(0, 1.2e-4)
        drifts[i] = drift
    shocks = rng.normal(0, vol, n_minutes) + drifts
    log_price = np.log(start) + np.cumsum(shocks)
    close = np.exp(log_price)
    openp = np.empty(n_minutes); openp[0] = start; openp[1:] = close[:-1]
    # intrabar wick noise
    wick = np.abs(rng.normal(0, vol * 0.7, n_minutes)) * close
    high = np.maximum(openp, close) + wick
    low = np.minimum(openp, close) - wick
    ts = pd.date_range("2024-01-01", periods=n_minutes, freq="1min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high,
                         "low": low, "close": close, "volume": 1.0})


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("timestamp").resample(rule)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum()})
    return out.dropna().reset_index()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work = os.path.join(base_dir, "_demo")
    parser_out = os.path.join(work, "parser_out")
    os.makedirs(parser_out, exist_ok=True)

    print("1) simulating 1-min data ...")
    m1 = simulate_1min()

    print("2) resampling to M5 / M15 / H4 ...")
    tf_rules = {"m5": "5min", "m15": "15min", "h4": "4h"}
    ohlc = {tf: resample(m1, rule) for tf, rule in tf_rules.items()}
    for tf, df in ohlc.items():
        print(f"   {tf}: {len(df):,} bars")

    print("3) running parser on each timeframe ...")
    for tf in ("m5", "m15", "h4"):
        pkg = run_parser_on_ohlc(ohlc[tf], source_tf=tf.upper())
        export_parser_dir(os.path.join(parser_out, tf), pkg)
        n_ev = len(pkg["events"]); n_poi = len(pkg["pois"])
        print(f"   {tf}: {len(pkg['states']):,} states, {n_ev} events, {n_poi} pois")

    # M5 OHLC csv for the merge step
    m5_ohlc_csv = os.path.join(work, "m5_ohlc.csv")
    ohlc["m5"].to_csv(m5_ohlc_csv, index=False)

    print("4) building fused MTF table ...")
    master = MTF.build_mtf_state_table(
        os.path.join(parser_out, "m5"), os.path.join(parser_out, "m15"),
        os.path.join(parser_out, "h4"), m5_ohlc_csv)
    master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
    master_csv = os.path.join(work, "mtf_state.csv")
    master.to_csv(master_csv, index=False)
    print(f"   fused table: {len(master):,} rows x {master.shape[1]} cols")

    print("5) training all systems ...")
    parser_dirs = {tf: os.path.join(parser_out, tf) for tf in ("m5", "m15", "h4")}
    smc_train.train_all(master, parser_dirs, os.path.join(work, "models_out"), horizon=75)

    print("\nDONE. See:", os.path.join(work, "models_out"))


if __name__ == "__main__":
    main()
