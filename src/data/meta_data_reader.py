import pandas as pd
import yfinance as yf
import numpy as np
from config import features

def clean_polygon_tickers():
    df = pd.read_csv("data/raw/polygon_tickers_messy.csv")
    df1 = pd.read_csv("data/raw/Wilshire-5000-Stocks.csv")
    df = df.merge(df1, how="inner", left_on="ticker", right_on="Ticker")
    df = df[["ticker", "name"]]

    df.to_csv("data/processed/tickers.csv", index=False)

def get_tickers():
    tickers = pd.read_csv("data/processed/tickers.csv")
    return tickers

def get_data_single_ticker(ticker, features):
    stock = yf.Ticker(ticker)
    data = [ticker]
    print(ticker)
    for feature in features:
        try:
            data.append(stock.info[feature])
        except KeyError:
            data.append(np.nan)
    return data

def get_data(tickers, features):
    data = []
    for ticker in tickers["ticker"]:
        data.append(get_data_single_ticker(ticker, features))

    df = pd.DataFrame(columns=["ticker", *features], data=data)
    df.to_csv("data/raw/meta_data.csv")
    

def main():
    clean_polygon_tickers()
    tickers = get_tickers()
    get_data(tickers, features)


if __name__ == "__main__":
    main()