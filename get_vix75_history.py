import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os

# ============================================================
# CONFIG
# ============================================================

SYMBOL = [#"EURUSD",
          "Volatility 75 Index"
          #, "Volatility 100 (1s) Index"
          #, "Volatility 10 (1s) Index"
          #, "GBPUSD",
           #"GBPJPY",
          #, "USDCAD"
          #"XAUUSD"
          #"UK 100"
          ]               # Make sure your broker uses this exact name
TIMEFRAMES = {
    #"M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H4": mt5.TIMEFRAME_H4,
    #"D1": mt5.TIMEFRAME_D1
}

# How far back to download
YEARS_BACK = 15

# Output folder
OUTPUT_DIR = "./historical_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    # Loop through each individual item instead of passing the whole list
    for symbol in SYMBOL:
        # FIX: Changed SYMBOL to symbol
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}. Check if your broker provides this exact name.")
        else:
            print(f"[OK] Selected symbol: {symbol}")

def download_data_simple(symbol, timeframe, start_time):
    """
    Downloads data iteratively starting from a specific time until the present.
    This avoids complex manual chunk calculations and guarantees the latest data.
    """
    result = []
    
    # Request data starting from the specified time
    # We rely on MT5 to fetch data up to the present moment if available
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, datetime.now(pytz.utc))

    if rates is None:
        raise RuntimeError(f"Failed to copy rates: {mt5.last_error()}")
    
    if len(rates) == 0:
        raise RuntimeError("No data downloaded. Check your date range or symbol name.")
    else:
        result.append(pd.DataFrame(rates))
        print(f"[INFO] Downloaded {len(rates)} bars from {start_time.strftime('%Y-%m-%d %H:%M')}")

    df = pd.concat(result, ignore_index=True)
    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time")

    return df

def process_dataframe(df):
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={
        "open": "Open",
        "spread": "Spread",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume"
    })
    df = df[["Open", "High", "Low", "Close", "Volume", "Spread"]]

    df = df[df.index.notnull()]
    df = df[df["Open"].notnull()]

    return df


def save_data(df, name):
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    parquet_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")

    df.to_csv(csv_path)
    df.to_parquet(parquet_path)

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {parquet_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    init_mt5()

    end = datetime.now(pytz.utc)
    start = end - timedelta(days=YEARS_BACK * 365)

    # FIX: Loop through each individual symbol FIRST
    for symbol in SYMBOL:
        for label, tf in TIMEFRAMES.items():
            print(f"\n=== Downloading {symbol} {label} ===")

            try:
                # Pass the single string 'symbol' here, not the list 'SYMBOL'
                raw_df = download_data_simple(symbol, tf, start)
                print(f"[INFO] Raw rows: {len(raw_df)}")

                df = process_dataframe(raw_df)
                print(f"[INFO] Clean rows: {len(df)}")

                # Use a clean file name format for each unique asset
                clean_filename = symbol.replace(" ", "_").replace("(", "").replace(")", "")
                save_data(df, f"{clean_filename}_{label}")
                
            except RuntimeError as e:
                print(f"[ERROR] Could not download {symbol} on {label}: {e}")
                print("Skipping to next task...")

    mt5.shutdown()
    print("\n[COMPLETE] All tasks processed.")

if __name__ == "__main__":
    main()
 
