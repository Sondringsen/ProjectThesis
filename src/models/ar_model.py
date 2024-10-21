import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

def grid_data(df: pd.DataFrame, time_diff: int = 10):
    """
    Grids the data for the AR-model to ensure equal distance between timestamps. Uses timestamp of 10 seconds by default.

    Args:
        df (pd.DataFrame): dataframe to grid.
        time_diff (int): the number of seconds between each point in the grid.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df_resampled = df.copy()
    df_resampled["index"] = df_resampled.index
    df_resampled = df_resampled.set_index("timestamp", drop=True)
    df_resampled = df_resampled.resample(f"{time_diff}s").last()
    df_resampled = df_resampled.ffill()
    df_resampled = df_resampled[["index"]]

    df = pd.merge(df_resampled, df, left_on="index", right_index=True, how="left")
    df["timestamp"] = df.index
    df = df.set_index(df["index"])
    df = df.drop(columns=["index"])
    return df

def ar_model(df: pd.DataFrame, lags: int):
    """
    Trains a AR(p) model on a rolling window basis.

    Args:
        df (pd.DataFrame): dataframe containing the data to train the model.
        lags (int): number of lags in the AR(p) model (p=lags).

    Returns:
        pd.DataFrame: dataframe with predictions.
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('10s')

    time_stamps = []
    predictions = []

    for i in range(df.shape[0] // 100, df.shape[0]):
        df_train = df.iloc[:i, :]
        
        model = AutoReg(df_train['mid_price'], lags=lags).fit()
        prediction = model.predict(start=i, end=i)

        predictions.append(prediction.values[0]) 
        time_stamps.append(df.index[i])

    df_pred = pd.DataFrame({
        'timestamp': time_stamps,
        'prediction': predictions
    })

    df_pred = pd.merge(df_pred, df, on='timestamp', how='inner')
    return df_pred

def main():
    df = pd.read_csv("data/processed/msft.csv", index_col=0)
    df = grid_data(df)

    df_pred = ar_model(df, 5)
    df_pred.to_csv("models/predictions/benchmark_ar.csv")

if __name__ == "__main__":
    main()

