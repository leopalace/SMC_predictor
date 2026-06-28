import sys, os, pandas as pd
from make_synth_and_run import simulate_1min, resample
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF
work="_tct"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min(n_minutes=80000, seed=4)
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master.to_csv(os.path.join(work,"mtf.csv"),index=False)
sys.argv=["confirm_trade.py","--mtf-csv",os.path.join(work,"mtf.csv"),
          "--h4-dir",f"{pout}/h4","--m15-dir",f"{pout}/m15","--m5-dir",f"{pout}/m5",
          "--max-bars-m5","480","--confirm-bars-m5","48"]
import confirm_trade; confirm_trade.main()
