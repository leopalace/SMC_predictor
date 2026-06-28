import os, pandas as pd, numpy as np
from make_synth_and_run import simulate_1min, resample
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF, smc_labeling as L

work="_t"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min(n_minutes=40000, seed=3)
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper())
    # MIMIC user's parser: states.csv WITHOUT ohlc columns
    pkg["states"]=pkg["states"].drop(columns=[c for c in ("open","high","low","close") if c in pkg["states"].columns])
    export_parser_dir(os.path.join(pout,tf), pkg)
print("states columns have OHLC? ->", all(c in pd.read_csv(f'{pout}/m15/states.csv').columns for c in ('open','high','low','close')))
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master["timestamp"]=pd.to_datetime(master["timestamp"],utc=True)

# BEFORE fix behaviour: no OHLC source -> depth NaN
pkg15=L.load_tf(f"{pout}/m15")
ep_no=L.build_episodes_for_tf(pkg15,"m15",ohlc_df=None)
n_no = ep_no.pullbacks["pullback_depth"].notna().sum() if not ep_no.pullbacks.empty else 0
print(f"\n[no OHLC source]  m15 pullback rows={len(ep_no.pullbacks)}  depth NOT-nan={n_no}")

# AFTER fix: build_all_labels auto-resamples OHLC from master
labels=L.build_all_labels(master,{tf:f'{pout}/{tf}' for tf in ('m5','m15','h4')})
for tf in ("m15","h4"):
    pb=labels[f"{tf}_pullback"]; co=labels[f"{tf}_continuation"]
    dn = pb["pullback_depth"].notna().sum() if not pb.empty else 0
    en = co["cont_extent_atr"].notna().sum() if not co.empty else 0
    dm = pb["pullback_depth"].median() if dn else float('nan')
    print(f"[auto-resample]   {tf}: pullback rows={len(pb)} depth NOT-nan={dn} (median={dm:.3f}) | cont rows={len(co)} extent_atr NOT-nan={en}")
