import time, pandas as pd, numpy as np
import smc_labeling as L, smc_baselines as B, smc_models as M
master = pd.read_csv("_demo/mtf_state.csv", low_memory=False)
master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
pdirs = {tf: f"_demo/parser_out/{tf}" for tf in ("m5","m15","h4")}
labels = L.build_all_labels(master, pdirs, horizon=75)
fwd = labels["m5_forward"]
feats = M.select_feature_columns(fwd, extra_drop=["fwd75_net_atr"])
print("backend:", M.BACKEND, "| n features:", len(feats))
bad = [c for c in feats if any(s in c.lower() for s in ("bar_index","_idx")) or c in ("open","high","low","close")]
print("leaky idx/price still present:", bad[:10])
t=time.time()
bl = B.make_baseline("fwd_net_atr", conditioners=B.DEFAULT_CONDITIONERS["fwd_net_atr"]); bl.target="fwd75_net_atr"
res = M.run_task(fwd, "fwd75_net_atr", classification=False, baseline=bl, name="m5_fwd_net", n_splits=4)
print(f"\nM5 forward-net task took {time.time()-t:.1f}s")
print(M.format_result(res))
