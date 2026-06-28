import pandas as pd, numpy as np, smc_labeling as L
master = pd.read_csv("_demo/mtf_state.csv", low_memory=False)
master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
pdirs = {tf: f"_demo/parser_out/{tf}" for tf in ("m5","m15","h4")}
import time; t=time.time()
labels = L.build_all_labels(master, pdirs, horizon=75)
print(f"labeling took {time.time()-t:.1f}s\n")
for k,df in labels.items():
    print(f"== {k}: {df.shape}")
for tf in ("h4","m15"):
    pb = labels[f"{tf}_pullback"]; co = labels[f"{tf}_continuation"]
    if not pb.empty:
        print(f"\n{tf} PULLBACK n={len(pb)}  depth mean={pb.pullback_depth.mean():.3f} "
              f"median={pb.pullback_depth.median():.3f}  cont_rate={pb.continuation_occurred.mean():.2f}")
        print(pb[["direction","pullback_depth","continuation_occurred","bars_to_continuation"]].head(6).to_string())
    if not co.empty:
        print(f"{tf} CONT n={len(co)}  extent_atr mean={co.cont_extent_atr.mean():.2f} median={co.cont_extent_atr.median():.2f}")
fwd = labels["m5_forward"]
print("\nM5 forward (h=75) describe:")
print(fwd[["fwd75_mfe","fwd75_mae","fwd75_net","fwd75_net_atr","fwd75_dir"]].describe().round(3).to_string())
bs = labels["m5_block_summary"]
print(f"\nM5 block summary n={len(bs)} avg up={bs.up_move.mean():.1f} avg down={bs.down_move.mean():.1f} avg net={bs.net_move.mean():.1f}")
# leakage check
from smc_models import select_feature_columns
feats = select_feature_columns(fwd, extra_drop=["fwd75_net_atr"])
leaky = [c for c in feats if any(s in c.lower() for s in ("fwd","net_move","up_move","down_move"))]
print(f"\nfeature cols selected: {len(feats)}; leaky among them: {leaky}")
