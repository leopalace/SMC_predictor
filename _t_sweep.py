import os, pandas as pd, numpy as np
from make_synth_and_run import simulate_1min, resample
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF, smc_labeling as L
import sweep_cont_horizon as SW
work="_tsw"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min(n_minutes=80000, seed=2)
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master["timestamp"]=pd.to_datetime(master["timestamp"],utc=True)
pkg=L.load_tf(f"{pout}/m15"); o=L.resample_ohlc(master,"m15")
for H in (120,480):
    ep=L.build_episodes_for_tf(pkg,"m15",ohlc_df=o,cont_horizon=H)
    pb=L.attach_features(ep.pullbacks,master); pb["is_range"]=(pb["resolution_type"]=="range").astype(int)
    print(f"H={H} n={len(pb)} cont%={100*pb['continuation_occurred'].mean():.0f} range%={100*pb['is_range'].mean():.0f} types={pb['resolution_type'].value_counts().to_dict()}")
print("SWEEP CORE OK")
