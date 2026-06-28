import sys, os, numpy as np, pandas as pd
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF
def simulate_1min(n=80000, seed=4, start=100000.0):
    rng=np.random.default_rng(seed); drift=0.0; drifts=np.zeros(n); vol=0.0009
    for i in range(n):
        if rng.random()<0.004: drift=rng.normal(0,1.2e-4)
        drifts[i]=drift
    close=np.exp(np.log(start)+np.cumsum(rng.normal(0,vol,n)+drifts))
    op=np.empty(n); op[0]=start; op[1:]=close[:-1]
    w=np.abs(rng.normal(0,vol*0.7,n))*close
    ts=pd.date_range("2024-01-01",periods=n,freq="1min",tz="UTC")
    return pd.DataFrame({"timestamp":ts,"open":op,"high":np.maximum(op,close)+w,"low":np.minimum(op,close)-w,"close":close,"volume":1.0})
def resample(df,rule):
    g=df.set_index("timestamp").resample(rule)
    return pd.DataFrame({"open":g["open"].first(),"high":g["high"].max(),"low":g["low"].min(),"close":g["close"].last(),"volume":g["volume"].sum()}).dropna().reset_index()
work="_tct"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=simulate_1min()
ohlc={tf:resample(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(ohlc[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
m5csv=os.path.join(work,"m5.csv"); ohlc["m5"].to_csv(m5csv,index=False)
master=MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",m5csv)
master.to_csv(os.path.join(work,"mtf.csv"),index=False)
sys.argv=["confirm_trade.py","--mtf-csv",os.path.join(work,"mtf.csv"),
          "--h4-dir",f"{pout}/h4","--m15-dir",f"{pout}/m15","--m5-dir",f"{pout}/m5"]
import confirm_trade; confirm_trade.main()
