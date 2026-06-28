import pandas as pd, smc_train
master = pd.read_csv("_demo/mtf_state.csv", low_memory=False)
master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
master = master.iloc[::7].reset_index(drop=True)   # subsample so numpy fallback fits time budget
pdirs = {tf: f"_demo/parser_out/{tf}" for tf in ("m5","m15","h4")}
res = smc_train.train_all(master, pdirs, "_demo/models_out", horizon=40, n_splits=3)
print("\nTASKS TRAINED:", list(res.keys()))
