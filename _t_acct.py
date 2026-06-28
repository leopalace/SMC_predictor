import sys, os, numpy as np, pandas as pd
from run_parser import run_parser_on_ohlc, export_parser_dir
import mtf_state_builder as MTF
def sim(n=85000, seed=4, start=100000.0):
    rng=np.random.default_rng(seed); d=0.0; ds=np.zeros(n); vol=0.0009
    for i in range(n):
        if rng.random()<0.004: d=rng.normal(0,1.2e-4)
        ds[i]=d
    c=np.exp(np.log(start)+np.cumsum(rng.normal(0,vol,n)+ds))
    o=np.empty(n); o[0]=start; o[1:]=c[:-1]; w=np.abs(rng.normal(0,vol*0.7,n))*c
    ts=pd.date_range("2021-01-01",periods=n,freq="1min",tz="UTC")
    return pd.DataFrame({"timestamp":ts,"open":o,"high":np.maximum(o,c)+w,"low":np.minimum(o,c)-w,"close":c,"volume":1.0})
def rs(df,r):
    g=df.set_index("timestamp").resample(r)
    return pd.DataFrame({"open":g["open"].first(),"high":g["high"].max(),"low":g["low"].min(),"close":g["close"].last(),"volume":g["volume"].sum()}).dropna().reset_index()
work="_tac"; pout=os.path.join(work,"parser_out"); os.makedirs(pout,exist_ok=True)
m1=sim(); oh={tf:rs(m1,r) for tf,r in {"m5":"5min","m15":"15min","h4":"4h"}.items()}
for tf in ("m5","m15","h4"):
    pkg=run_parser_on_ohlc(oh[tf], tf.upper()); export_parser_dir(os.path.join(pout,tf), pkg)
oh["m5"].to_csv(os.path.join(work,"m5.csv"),index=False)
MTF.build_mtf_state_table(f"{pout}/m5",f"{pout}/m15",f"{pout}/h4",os.path.join(work,"m5.csv")).to_csv(os.path.join(work,"mtf.csv"),index=False)
sys.argv=["account_backtest.py","--mtf-csv",os.path.join(work,"mtf.csv"),
          "--h4-dir",f"{pout}/h4","--m15-dir",f"{pout}/m15","--m5-dir",f"{pout}/m5",
          "--entry","immediate","--target","major_high",
          "--tick-size","0.01","--tick-value","0.01","--lot-step","0.001","--min-lot","0.001",
          "--max-lot","50","--spread-price","5","--commission","0","--balance","1000","--risk-pct","1.0",
          "--out-csv",os.path.join(work,"eq.csv")]
import account_backtest as A; A.main()
