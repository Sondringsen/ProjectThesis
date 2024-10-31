import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utils.create_model import create_lstm
from config import lstm_config

def get_data():
    """
    Reads data from files to be used for training and testing.

    Returns:
        (pd.DataFrame, pd.DataFrame): dataframes used for training and testing.
    """
    df_train = pd.read_csv("data/processed/tq_data_gridded/2024-09-03_gridded.csv")
    df_test = pd.read_csv("data/processed/tq_data_gridded/2024-09-17_gridded.csv")
    return df_train, df_test

def create_sequences(data: np.ndarray, seq_length: int):
    """
    Convert DataFrame into sequences of specified length for LSTM input.

    Args: 
        data (np.array): complete data containing both x and y
        seq_length (int): the length of the sequence considered in the lstm.

    Returns:
        (np.array, np.array): x and y datasets with the sequences. 
    """
    x, y = [], []
    for i in range(len(data) - seq_length):
        # Create sequences of length `seq_length`
        x.append(data[i:i+seq_length, :-1])  # all columns except the last for features
        y.append(data[i+seq_length, -1])     # last column as target
    return np.array(x), np.array(y)

def create_loss_report(df_loss: pd.DataFrame):
    """
    Creates a latex table containing loss metrics for all tickers.
    
    Args:
        df_loss (pd.DataFrame): a dataframe containing different loss metrics for all tickers.
    """
    with open("reports/loss_tot_lstm.tex", "w") as file:
        file.write(df_loss.to_latex(index=False, header=True, float_format="%.4f", caption="Loss Metrics For All Tickers LSTM-model", label="tab:loss_total_lstm"))

def create_plot(ticker: str, ts: pd.Series, y_test: np.ndarray, y_pred: np.ndarray):
    """
    Creates and saves a plot of the true values and the predicted values over the test period.

    Args:
        ticker (str): the symbol/ticker of the stock under analysis.
        ts (pd.Series): a series of timestamps corresponding to the y values.
        y_test (np.ndarray): a numpy array of true values.
        y_pred (np.ndarray): a numpy array of predicted values.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(ts, y_test, label="True")
    plt.plot(ts, y_pred, label="Pred")
    plt.legend()
    plt.title(f"LSTM model prediction for {ticker} (min-max-scaled values)")
    plt.ylabel("Midprice")
    plt.xlabel("Time")

    plt.savefig(f"reports/plots/{ticker}_lstm.png")

def main():
    df_train, df_test = get_data()
    
    tickers = df_train["ticker"].unique()
    scaler = MinMaxScaler() # should change this to avoid data leakage (test should not be normalized with values from train)

    loss_dic = {"ticker": [], "mse": [], "rmse": [], "normalized rmse": [], "mae": []}
    i = 0
    for ticker in tickers:
        if i == 3:
            break
        i += 1

        print(f"Analyzing {ticker}")
        
        ticker_data_train = df_train[df_train["ticker"] == ticker]
        ticker_data_train = ticker_data_train.drop(columns=["ticker", "sip_timestamp"])

        data = scaler.fit_transform(ticker_data_train)
        X_train, y_train = create_sequences(data, lstm_config["seq_length"])

        model = create_lstm(**lstm_config)
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=10, batch_size=32)
        model.summary()

        ticker_data_test = df_test[df_test["ticker"] == ticker]
        ts = pd.to_datetime(ticker_data_test["sip_timestamp"].iloc[lstm_config["seq_length"]:]) # used for plotting
        ticker_data_test = ticker_data_test.drop(columns=["ticker", "sip_timestamp"])
        data = scaler.fit_transform(ticker_data_test)

        X_test, y_test = create_sequences(data, lstm_config["seq_length"])
        y_pred = model.predict(X_test)

        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        normalized_mse = rmse / np.std(y_test)
        mae = np.mean(np.abs(y_test - y_pred))

        loss_dic["ticker"].append(ticker)
        loss_dic["mse"].append(mse)
        loss_dic["rmse"].append(rmse)
        loss_dic["normalized rmse"].append(normalized_mse)
        loss_dic["mae"].append(mae)

        create_plot(ticker, ts, y_test, y_pred)

    df_loss = pd.DataFrame(loss_dic)
    create_loss_report(df_loss)

if __name__ == "__main__":
    main()