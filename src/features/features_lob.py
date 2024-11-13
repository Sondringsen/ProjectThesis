import os
import pandas as pd
from config import time_freq
from datetime import time

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


def add_first_hour_indicator(df: pd.DataFrame):
    """
    Adds the first hour of the trading day as an indicator variable. This is done
    since the open is more volatile than the rest of the day.

    Args:
        df (pd.Dataframe): dataframe to add the indicator to.

    Returns:
        pd.DataFrame: dataframe with indicator variable.
    """
    first_hour = time(14, 30)
    df["first_hour_indicator"] = df["sip_timestamp"].dt.time <= first_hour

    return df

def add_last_hour_indicator(df: pd.DataFrame):
    """
    Adds the last hour of the trading day as an indicator variable. This is done
    since the close is more volatile than the rest of the day.

    Args:
        df (pd.Dataframe): dataframe to add the indicator to.

    Returns:
        pd.DataFrame: dataframe with indicator variable.
    """
    last_hour = time(19, 00)
    df["last_hour_indicator"] = df["sip_timestamp"].dt.time >= last_hour

    return df


def add_weighted_mid_price(df: pd.DataFrame):
    """
    Adds a mid price weighted by the sizes on each side. This will be used as the target.
    This metric is in line with the literature.

    Args:
        df (pd.Dataframe): dataframe to add the weighted mid price to.

    Returns:
        pd.DataFrame: dataframe with mid price.
    """
    weights = df["bid_size"] / (df["bid_size"] + df["ask_size"])
    df["weighted_mid_price"] = df["bid_price"] * (1 - weights) + df["ask_price"] * weights

    return df

def filter_time(df: pd.DataFrame):
    """
    Filter a dataframe on time to only include opening hours (9:30-16:00). 

    Args:
        df (pd.DataFrame): dataframe to filter on time.
    
    Returns:
        pd.DataFrame: dataframe with only prices between 9:30 and 16:00
    """ 
    df = df.loc[(df.loc[:, "sip_timestamp"].dt.time > pd.to_datetime("13:30").time()) & 
                            (df.loc[:, "sip_timestamp"].dt.time < pd.to_datetime("20:00").time()), :]
    return df

def grid_data(df: pd.DataFrame, time_freq: int):
    """
    Grids the data for the AR-model to ensure equal distance between timestamps. Uses timestamp of 10 seconds by default.

    Args:
        df (pd.DataFrame): dataframe to grid.
        time_freq (int): the number of seconds between each point in the grid.
    """
    df = df.set_index("sip_timestamp", drop=True)
    df = df.resample(f"{time_freq}s").last()
    df = df.ffill()

    return df

def add_features(df: pd.DataFrame):
    """
    Function to call for adding new features to the data.

    Args: 
        df (pd.DataFrame): dataframe to add features to.

    Returns:
        pd.DataFrame: dataframe with added features
    
    Note:: make sure to add target last
    """
    # df = add_first_hour_indicator(df)
    # df = add_last_hour_indicator(df)
    df = add_mid_price(df)
    # df = add_weighted_mid_price(df)
    return df

def remove_features(df: pd.DataFrame):
    """
    Function to call for removing features of the data.

    Args: 
        df (pd.DataFrame): dataframe to remove features from.

    Returns:
        pd.DataFrame: dataframe with removed features
    """
    features_to_remove = ["ask_price", "bid_price", "ask_size", "bid_size", "last_trade_price", "last_trade_size"]
    df = df.drop(columns=features_to_remove)

    return df
import numpy as np

def return_transformation(df: pd.DataFrame):
    """
    Transform the mid price time series into a log return series. Ensures that no returns span two days.

    Args:
        df (pd.DataFrame): DataFrame with a 'sip_timestamp' datetime column and 'mid_price' column.
    
    Returns:
        pd.DataFrame: DataFrame with 'mid_price_log_return' values.
    """
    df["date"] = df["sip_timestamp"].dt.date
    
    df["mid_price"] = df["mid_price"].replace(0, np.nan)
    df["mid_price"] = df.groupby("date")["mid_price"].ffill()

    df["mid_price_log_return"] = df.groupby("date")["mid_price"].transform(lambda x: np.log(x / x.shift(1)))
    
    df = df.dropna(subset=["mid_price_log_return"]).reset_index(drop=True)
    df = df.drop(columns=["date", "mid_price"])
    
    return df



def process_single_file(df: pd.DataFrame):
    """
    Applies all preprocessing done for a single day.

    Args:
        df (pd.DataFrame): dataframe with trade data from a single day.

    Returns:
        pd.DataFrame: returns a processed dataframe.
    """
    df["sip_timestamp"] = pd.to_datetime(df["sip_timestamp"])
    df = add_features(df)
    df = remove_features(df)
    df = filter_time(df)
    df = df.groupby(by="ticker").apply(lambda df: grid_data(df, time_freq), include_groups=False).reset_index()
    df = return_transformation(df)
    
    return df

def main_already_merged():
    """
    This main function assumes you already have merged quote and trade data. This was only used because the total
    data on the external disk is computationally intensive to download and treat as it is 600GB in size. 
    The merged data was already available under data/processed/tq_data.
    """
    input_dir = "data/processed/tq_data"
    output_dir = "data/processed/tq_data_gridded"

    df_tot = None
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_dir, filename)
            print(f"Treating {input_filepath}")
            
            df = pd.read_csv(input_filepath)
            df = process_single_file(df)
            

            if df_tot is None:
                df_tot = df
            else:
                df_tot = pd.concat([df_tot, df])
    
    df_tot = df_tot.sort_values(by=["ticker", "sip_timestamp"])
            
    output_filename = "df_tot_gridded.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    df_tot.to_csv(output_filepath, index=False)
    print(f"Processed and saved: {output_filepath}")

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
    # main()
    main_already_merged()