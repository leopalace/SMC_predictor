import os, pandas as pd, numpy as np
from make_synth_and_run import simulate_1min, resample
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF, smc_labeling as L, smc_models as M
work="_trl"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min(n_minutes=90000, seed=8)
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master["timestamp"]=pd.to_datetime(master["timestamp"],utc=True)
labels=L.build_all_labels(master,{tf:f'{pout}/{tf}' for tf in ('m5','m15','h4')})
pb=labels["m15_pullback"]
print("episodes:",len(pb))
print("new cols:",[c for c in ('continuation_occurred','continuation_event','resolution_type','bars_to_continuation') if c in pb.columns])
print("resolution_type dist:", pb['resolution_type'].value_counts().to_dict())
print(f"path cont rate={pb['continuation_occurred'].mean():.3f}  event cont rate={pb['continuation_event'].mean():.3f}")
agree=(pb['continuation_occurred']==pb['continuation_event']).mean()
print(f"path vs event agreement: {agree:.1%}  (disagreements = {int((1-agree)*len(pb))})")
feats=M.select_feature_columns(pb, extra_drop=["continuation_occurred"])
leaky=[c for c in feats if 'continuation' in c or c=='resolution_type' or 'reclaim' in c or c.startswith('ref_')]
print("label cols leaked into features:", leaky)
print("OK")
