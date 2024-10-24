import os
import pandas as pd

tickers = pd.read_csv("data/raw/random_tickers.csv")["ticker"]
dates = [f for f in os.listdir("data/raw/quotes") if not f.startswith('.')]

for date in dates:
    print(f"Processing {date}...")
    
    quote_path = os.path.join("data/raw/quotes", date)
    trade_path = os.path.join("data/raw/trades", date)

    dfq = pd.read_csv(quote_path)
    dft = pd.read_csv(trade_path)

    df = pd.merge(dfq, dft, on=["ticker", "sip_timestamp"], how="outer")
    
    df = df.sort_values(by=["ticker", "sip_timestamp"])
    
    df.update(df.groupby("ticker").ffill())
    
    df.rename(columns={"price": "last_trade_price", "size": "last_trade_size"}, inplace=True)
    
    nancount = df.isna().sum().sum()
    print(f"Dropping {100 * nancount / len(df):.2f}% NaN values")
    df.dropna(inplace=True)
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    save_path = os.path.join("data/processed", date)
    df.to_csv(save_path, index=False)

    print(f"Completed processing for {date}.")
