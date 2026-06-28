import time, pandas as pd, numpy as np
import smc_labeling as L, smc_baselines as B, smc_models as M
print("backend:", M.BACKEND)
master = pd.read_csv("_demo/mtf_state.csv", low_memory=False)
master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
pdirs = {tf: f"_demo/parser_out/{tf}" for tf in ("m5","m15","h4")}
labels = L.build_all_labels(master, pdirs, horizon=75)

P = {"n_estimators":60, "learning_rate":0.08, "max_depth":4}

# leakage audit on M5 forward features
fwd = labels["m5_forward"]
feats = M.select_feature_columns(fwd, extra_drop=["fwd75_net_atr"])
bad = [c for c in feats if M._is_abs_price(c) or M._is_id_or_time(c) or "fwd" in c.lower()]
print(f"M5 features={len(feats)}  leaky_remaining={bad}")

# --- M15 SYSTEM 1: pullback depth (regression) ---
pb = labels["m15_pullback"]
bl = B.make_baseline("pullback_depth")
r = M.run_task(pb, "pullback_depth", classification=False, baseline=bl, name="m15_S1_depth", n_splits=3, params=P)
print(M.format_result(r))

# --- M15 SYSTEM 2a: continuation occurrence (classification) ---
bl = B.make_baseline("continuation_occurred", classification=True)
r = M.run_task(pb, "continuation_occurred", classification=True, baseline=bl, name="m15_S2a_cont", n_splits=3, params=P)
print(M.format_result(r))

# --- M15 SYSTEM 3: continuation extent (regression) ---
co = labels["m15_continuation"]
bl = B.make_baseline("cont_extent_atr")
r = M.run_task(co, "cont_extent_atr", classification=False, baseline=bl, name="m15_S3_extent", n_splits=3, params=P)
print(M.format_result(r))

# --- M5 SYSTEM 4a: forward net move (regression), subsampled for the numpy fallback ---
sub = fwd.iloc[::4].reset_index(drop=True)
bl = B.make_baseline("fwd_net_atr", conditioners=B.DEFAULT_CONDITIONERS["fwd_net_atr"]); bl.target="fwd75_net_atr"
t=time.time()
r = M.run_task(sub, "fwd75_net_atr", classification=False, baseline=bl, name="m5_S4a_fwd_net", n_splits=3, params=P)
print(M.format_result(r), f"\n(M5 fwd task {time.time()-t:.1f}s on {len(sub)} rows)")

# --- M5 SYSTEM 4b: forward direction (classification) ---
bl = B.make_baseline("fwd_net_atr", conditioners=B.DEFAULT_CONDITIONERS["fwd_net_atr"]); bl.target="fwd75_dir"; bl.classification=True
r = M.run_task(sub, "fwd75_dir", classification=True, baseline=bl, name="m5_S4b_fwd_dir", n_splits=3, params=P)
print(M.format_result(r))
