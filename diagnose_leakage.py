"""
diagnose_leakage.py
===================
Leakage probe. For each system it trains TWICE under identical CV:
  * real     - the true target
  * shuffled - the target randomly permuted (all real signal destroyed)

Interpretation:
  * shuffled AUC ~ 0.50 and shuffled R2 ~ 0     -> pipeline is clean; any gap the
                                                   'real' run shows over baseline is
                                                   (probably) genuine signal.
  * shuffled AUC >> 0.50 or shuffled R2 >> 0     -> the pipeline LEAKS: features carry
                                                   information that shouldn't be knowable
                                                   at decision time. Do not trust 'real'.

Usage (same paths as smc_train.py):
  python diagnose_leakage.py --mtf-csv mtf_state_table.csv \
      --h4-dir ... --m15-dir ... --m5-dir ... --horizon 75
"""
from __future__ import annotations

import argparse
import pandas as pd

import smc_labeling as L
import smc_models as M


SYSTEMS = [
    # (label_key, target, classification)
    ("h4_pullback",     "pullback_depth",        False),
    ("h4_pullback",     "continuation_occurred", True),
    ("h4_continuation", "cont_extent_atr",       False),
    ("m15_pullback",    "pullback_depth",        False),
    ("m15_pullback",    "continuation_occurred", True),
    ("m15_continuation","cont_extent_atr",       False),
]


def _metric_str(res):
    if res is None:
        return "skipped"
    return "  ".join(f"{k}={v:.4f}" if v == v else f"{k}=nan" for k, v in res.metrics.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtf-csv", required=True)
    ap.add_argument("--h4-dir", required=True)
    ap.add_argument("--m15-dir", required=True)
    ap.add_argument("--m5-dir", required=True)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--n-splits", type=int, default=4)
    args = ap.parse_args()

    master = pd.read_csv(args.mtf_csv, low_memory=False)
    master["timestamp"] = L._to_dt(master["timestamp"])
    master = master.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    pdirs = {"h4": args.h4_dir, "m15": args.m15_dir, "m5": args.m5_dir}
    labels = L.build_all_labels(master, pdirs, horizon=args.horizon)

    print(f"\nML backend: {M.BACKEND}\n")
    print(f"{'system':40s} {'real':28s} {'shuffled (control)':28s}")
    print("-" * 98)
    rows = list(SYSTEMS) + [
        ("m5_forward", f"fwd{args.horizon}_net_atr", False),
        ("m5_forward", f"fwd{args.horizon}_dir", True),
    ]
    for key, target, clf in rows:
        df = labels.get(key)
        name = f"{key}:{target}"
        if df is None or df.empty or target not in df.columns:
            print(f"{name:40s} {'n/a':28s}")
            continue
        real = M.run_task(df, target, classification=clf, name=name,
                          n_splits=args.n_splits)
        shuf = M.run_task(df, target, classification=clf, name=name,
                          n_splits=args.n_splits, shuffle_y=True)
        print(f"{name:40s} {_metric_str(real):28s} {_metric_str(shuf):28s}")


if __name__ == "__main__":
    main()
