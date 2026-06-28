import os, pandas as pd, numpy as np
from make_synth_and_run import simulate_1min, resample
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF, smc_labeling as L, smc_models as M
import calibrate_ev as EV
work="_tev"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min(n_minutes=85000, seed=4)
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master["timestamp"]=pd.to_datetime(master["timestamp"],utc=True)
labels=L.build_all_labels(master,{tf:f'{pout}/{tf}' for tf in ('m5','m15','h4')})
pb=labels["m15_pullback"]
print("episodes:", len(pb))
print("ref cols present:", [c for c in ('ref_strong','ref_weak','ref_close','sh_bar','direction') if c in pb.columns])
feats=M.select_feature_columns(pb, extra_drop=["continuation_occurred"])
leaky=[c for c in feats if c.startswith('ref_') or c=='sh_bar']
print("ref leaked into features?:", leaky)
w=EV.oof_probabilities(pb,"continuation_occurred",n_splits=3)
print("oof scored:", int(np.isfinite(w['oof_prob']).sum()), "| cal_prob present:", 'cal_prob' in w.columns)
m15o=L.resample_ohlc(master,"m15")
sim=EV.simulate(w, m15o, entry_frac=0.6, stop_buf_atr=0.5, target_R=2.0, max_bars=160)
print("fill rate %:", round(100*sim['filled'].mean(),1), "| outcomes:", pd.Series(sim['out_2R'].dropna()).value_counts().to_dict())
EV.ev_table(sim,"cal_prob","out_2R",2.0,thresholds=(0.5,0.6))
print("EV ENGINE OK")
