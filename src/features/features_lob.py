import pandas as pd

def add_mid_price(df: pd.DataFrame):
    """
    Adds the mid price defined as the average between the bid and offer.

    Args:
        df (pd.Dataframe): dataframe to add the mid price to.

    Returns:
        pd.DataFrame: dataframe with mid price.
    """

    df["mid_price"] = (df["bid_price"] + df["offer_price"]) / 2
    return df

def filter_time(df: pd.DataFrame):
    """
    Filter a dataframe on time to only include opening hours (9:30-16:00). 

    Args:
        df (pd.DataFrame): dataframe to filter on time.
    
    Returns:
        pd.DataFrame: dataframe with only prices between 9:30 and 16:00
    """ 
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.loc[(df.loc[:, "timestamp"].dt.time > pd.to_datetime("9:30").time()) & 
                            (df.loc[:, "timestamp"].dt.time < pd.to_datetime("16:00").time()), :]
    return df

def add_features(df: pd.DataFrame):
    df = add_mid_price(df)
    return df

def main():
    df = pd.read_csv("data/raw/msft.csv", index_col=0)
    df = add_features(df)
    df = filter_time(df)
    df.to_csv("data/processed/msft.csv")


if __name__ == "__main__":
    main()