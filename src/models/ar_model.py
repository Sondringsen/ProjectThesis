import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error

def optimal_lag_selection(df: pd.DataFrame, time_freq: int, max_lags: int = 20, criterion: str = "BIC"):
    """
    Find the optimal number of lags for an AR model based on AIC or BIC.
    
    Args:
        df (pd.DataFrame): The time series data.
        time_freq (int): the number of seconds between each point in the grid.
        max_lags (int): Maximum number of lags to test.
        criterion (str): Criterion to minimize, either "AIC" or "BIC".
    
    Returns:
        int: Optimal number of lags for the AR model.
        pd.DataFrame: DataFrame with lag, AIC, and BIC values.
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.asfreq(f'{time_freq}s')

    # Initialize lists to store results
    aic_values = []
    bic_values = []
    lags = range(1, max_lags + 1)
    
    # Fit AR models for each lag
    for p in lags:
        try:
            model = AutoReg(df["mid_price"], lags=p).fit()
            aic_values.append(model.aic)
            bic_values.append(model.bic)
        except ValueError:
            # Some lag values may be too large for the data, skip these
            continue
    
    # Compile results into DataFrame
    results = pd.DataFrame({"Lag": lags, "AIC": aic_values, "BIC": bic_values})
    
    # Find the optimal lag based on the selected criterion
    if criterion == "AIC":
        optimal_lag = results.loc[results["AIC"].idxmin(), "Lag"]
    elif criterion == "BIC":
        optimal_lag = results.loc[results["BIC"].idxmin(), "Lag"]
    else:
        raise ValueError("Criterion must be either 'AIC' or 'BIC'")
    
    return optimal_lag, results


def grid_data(df: pd.DataFrame, time_freq):
    """
    Grids the data for the AR-model to ensure equal distance between timestamps. Uses timestamp of 10 seconds by default.

    Args:
        df (pd.DataFrame): dataframe to grid.
        time_freq (int): the number of seconds between each point in the grid.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df_resampled = df.copy()
    df_resampled["index"] = df_resampled.index
    df_resampled = df_resampled.set_index("timestamp", drop=True)
    df_resampled = df_resampled.resample(f"{time_freq}s").last()
    df_resampled = df_resampled.ffill()
    df_resampled = df_resampled[["index"]]

    df = pd.merge(df_resampled, df, left_on="index", right_index=True, how="left")
    df["timestamp"] = df.index
    df = df.set_index(df["index"])
    df = df.drop(columns=["index"])
    return df

def ar_model(df: pd.DataFrame, lags: int, time_freq: int):
    """
    Trains a AR(p) model on a rolling window basis.

    Args:
        df (pd.DataFrame): dataframe containing the data to train the model.
        lags (int): number of lags in the AR(p) model (p=lags).
        time_freq (int): the number of seconds between each point in the grid.

    Returns:
        pd.DataFrame: dataframe with predictions.
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq(f'{time_freq}s')

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

def create_report(df: pd.DataFrame, df_pred: pd.DataFrame, ):
    """
    Creates a latex table with the MSE and MAE. 

    Args:
        df (pd.DataFrame): the dataframe used for training and prediction.
        df_pred (pd.DataFrame): predictions from the AR(p)-model.

    Returns:
        pd.DataFrame: a dataframe containing the loss.
    """
    columns = ["MSE", "MAE"]
    data = [
        [mean_squared_error(df[df.shape[0] // 100, "mid_price"], df_pred["mid_price"])]
        [mean_absolute_error(df[df.shape[0] // 100, "mid_price"], df_pred["mid_price"])]
    ]
    df_loss = pd.DataFrame(data, columns=columns)
    
    return df_loss



def main():
    # should seperate more between training and test.
    df = pd.read_csv("data/processed/msft.csv", index_col=0)

    time_freq = 10
    df = grid_data(df, time_freq)

    # optimal_lag, _ = optimal_lag_selection(df, time_freq)
    optimal_lag = 7

    df_pred = ar_model(df, optimal_lag, time_freq)
    df_pred.to_csv("models/predictions/benchmark_ar.csv")
    
    df_loss = create_report(df, df_pred)
    print(df_loss)

if __name__ == "__main__":
    main()

