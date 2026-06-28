import os, pandas as pd, numpy as np
from make_synth_and_run import simulate_1min, resample
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF, smc_labeling as L, smc_models as M
import calibrate_ev as EV
work="_tmtf"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min(n_minutes=85000, seed=4)
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master["timestamp"]=pd.to_datetime(master["timestamp"],utc=True)
labels=L.build_all_labels(master,{tf:f'{pout}/{tf}' for tf in ('m5','m15','h4')})
pb=labels["m15_pullback"]
w=EV.oof_probabilities(pb,"continuation_occurred",n_splits=3)
m5o=master[["timestamp","open","high","low","close"]].copy()
mev=L.load_tf(f"{pout}/m5").get("events",pd.DataFrame())
def ets(t): 
    s=mev[mev["event_type"].isin(t)]["timestamp"]; 
    return np.sort(pd.to_datetime(s,utc=True,errors="coerce").dropna().astype("int64").to_numpy())
bull=ets(["bos_bull","choch_bull"]); bear=ets(["bos_bear","choch_bear"])
print("episodes:",len(w),"| m5 bull ev:",len(bull),"bear:",len(bear))
sim=EV.simulate_mtf(w,m5o,bull,bear,entry_frac=0.6,stop_buf_atr=0.5,target_R=2.0,max_bars_m5=480,confirm_bars_m5=12)
print("MTF fill rate %:",round(100*sim['filled'].mean(),1),"| outcomes:",pd.Series(sim['out_2R'].dropna()).value_counts().to_dict())
EV.ev_table(sim,"cal_prob","out_2R",2.0,thresholds=(0.5,0.6))
print("MTF SIM OK")
