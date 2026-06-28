import os, pandas as pd, numpy as np
from make_synth_and_run import simulate_1min, resample
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF, smc_labeling as L, smc_models as M
work="_t3"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min(n_minutes=90000, seed=5)
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master["timestamp"]=pd.to_datetime(master["timestamp"],utc=True)
labels=L.build_all_labels(master,{tf:f'{pout}/{tf}' for tf in ('m5','m15','h4')})
pb=labels["m15_pullback"]; print("m15 pullback episodes:", len(pb))
for tgt,clf in [("pullback_depth",False),("continuation_occurred",True)]:
    r=M.run_task(pb,tgt,classification=clf,name="real",n_splits=2,min_rows=15)
    s=M.run_task(pb,tgt,classification=clf,name="shuf",n_splits=2,min_rows=15,shuffle_y=True)
    def f(x): return None if x is None else {k:round(v,3) for k,v in x.metrics.items()}
    print(f"{tgt:24s} real={f(r)}  shuffled={f(s)}")
