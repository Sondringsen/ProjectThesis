import os
import pandas as pd
from config import time_freq

def merge_trade_quote(dfq: pd.DataFrame, dft: pd.DataFrame):
    """
    Merges the raw data from trade and quote.

    Args:
        dfq (pd.DataFrame): dataframe with quote data.
        dft (pd.DataFrame): dataframe with trade data.
    
    Returns:
        pd.DataFrame: dataframe with merged data.
    """
    df = pd.merge(dfq, dft, on=["ticker", "sip_timestamp"], how="outer")
    
    df = df.sort_values(by=["ticker", "sip_timestamp"])
    
    df.update(df.groupby("ticker").ffill())
    
    df.rename(columns={"price": "last_trade_price", "size": "last_trade_size"}, inplace=True)
    
    nancount = df.isna().sum().sum()
    print(f"Dropping {100 * nancount / len(df):.2f}% NaN values")
    df.dropna(inplace=True)
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df

def add_mid_price(df: pd.DataFrame):
    """
    Adds the mid price defined as the average between the bid and offer.

    Args:
        df (pd.Dataframe): dataframe to add the mid price to.

    Returns:
        pd.DataFrame: dataframe with mid price.
    """

    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    return df

def filter_time(df: pd.DataFrame):
    """
    Filter a dataframe on time to only include opening hours (9:30-16:00). 

    Args:
        df (pd.DataFrame): dataframe to filter on time.
    
    Returns:
        pd.DataFrame: dataframe with only prices between 9:30 and 16:00
    """ 
    df["sip_timestamp"] = pd.to_datetime(df["sip_timestamp"])
    df = df.loc[(df.loc[:, "sip_timestamp"].dt.time > pd.to_datetime("9:30").time()) & 
                            (df.loc[:, "sip_timestamp"].dt.time < pd.to_datetime("16:00").time()), :]
    return df

def grid_data(df: pd.DataFrame, time_freq: int):
    """
    Grids the data for the AR-model to ensure equal distance between timestamps. Uses timestamp of 10 seconds by default.

    Args:
        df (pd.DataFrame): dataframe to grid.
        time_freq (int): the number of seconds between each point in the grid.
    """
    df["sip_timestamp"] = pd.to_datetime(df["sip_timestamp"])
    
    df = df.set_index("sip_timestamp", drop=True)
    df = df.resample(f"{time_freq}s").last()
    df = df.ffill()

    return df

def add_features(df: pd.DataFrame):
    df = add_mid_price(df)
    return df

def process_single_file(df: pd.DataFrame):
    """
    Applies all preprocessing done for a single day.

    Args:
        df (pd.DataFrame): dataframe with trade data from a single day.

    Returns:
        pd.DataFrame: returns a processed dataframe.
    """
    df = add_features(df)
    df = filter_time(df)
    
    return df

def main():
    tickers = pd.read_csv("data/raw/random_tickers.csv")["ticker"]
    dates = [f for f in os.listdir("data/raw/quotes") if not f.startswith('.')]

    for date in dates:
        print(f"Processing {date}...")

        quote_path = os.path.join("data/raw/quotes", date)
        trade_path = os.path.join("data/raw/trades", date)

        dfq = pd.read_csv(quote_path)
        dft = pd.read_csv(trade_path)

        df = merge_trade_quote(dfq, dft)
        df = df[df["ticker"].isin(tickers), :] # added this to filter out tickers we don't use: check this
        df = process_single_file(df)
        df_gridded = df.groupby(by="ticker").apply(lambda df: grid_data(df, time_freq), include_groups=False).reset_index()

        save_path = os.path.join("data/processed/tq_data", date)
        df.to_csv(save_path, index=False)

        save_path = os.path.join("data/processed/tq_data_gridded", date)
        df_gridded.to_csv(save_path, index=False)

        print(f"Completed processing for {date}.")


if __name__ == "__main__":
    main()