import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.create_model import create_lstm
from config import lstm_config

def get_data():
    """
    Reads data from files to be used for training and testing.

    Returns:
        (pd.DataFrame, pd.DataFrame): dataframes used for training and testing.
    """
    df = pd.read_csv("data/processed/tq_data_gridded/df_tot_gridded.csv")
    df["sip_timestamp"] = pd.to_datetime(df["sip_timestamp"])
    return df

def get_dates_for_training_scheme(df: pd.DataFrame):
    """
    This function returns all dates needed to train, validate and test the model. The training
    follows a rolling window type scheme. The data is trained on 4 days and validated on the 5th. 
    Some portion of the end of the dates are used for testing.

    Args:
        df (pd.DataFrame): dataframe with data.

    Returns:
        List[Tuples]: a list of tuples conatining the training and validation/test dates.
    """

    dates = df["sip_timestamp"].dt.day.unique()

    date_set = []

    num_days_training = 4
    num_days_val_test = 1

    for i in range(len(dates) - num_days_training):
        date_set.append((dates[i: i + 4], dates[i + 4: i + 4 + num_days_val_test]))

    return date_set



def train_val_test(df: pd.DataFrame, days: tuple[np.ndarray, np.ndarray]):
    """
    Splits a dataframe into training and val/test. The splitting is done by given dates.

    Args:
        df (pd.DataFrame): dataframe to split into training and test.
        days (tuple[np.ndarray, np.ndarray]): a tuple containing two arrays of train days and val/test days respectively

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: a train and val/test set of the original dataframe
    """
    
    train = df[df["sip_timestamp"].dt.day.isin(days[0])]
    val_test = df[df["sip_timestamp"].dt.day.isin(days[1])]

    return train, val_test


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

def get_config_space():
    """
    Creates a hyperparamater space to search over in tuning the model.
    """
    pass

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


def tuning_part(df: pd.DataFrame, date_scheme: tuple[np.ndarray, np.ndarray]):
    config_space = get_config_space()

    config_loss = []

    for config in config_space:
        y_test_total = []
        y_pred_total = []

        for date_set in date_scheme:
            train, test = train_val_test(df, date_set)
            scaler = StandardScaler()

            train = train.drop(columns=["ticker", "sip_timestamp"])
            test = test.drop(columns=["ticker", "sip_timestamp"])

            train = scaler.fit_transform(train)
            test = scaler.transform(test)

            X_train, y_train = create_sequences(train, lstm_config["seq_length"])
            X_test, y_test = create_sequences(train, lstm_config["seq_length"])

            model = create_lstm(**config)
            model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train, epochs=100, batch_size=32)
            model.summary()

            y_pred = model.predict(X_test)
            y_pred_total.append(y_pred)
            y_test_total.append(y_test)

        y_test_total = np.array(y_test_total).flatten()
        y_pred_total = np.array(y_pred_total).flatten()

        mse = np.mean((y_test_total - y_pred_total)**2)
        config_loss.append(mse)
        
    config_loss = np.array(config_loss)
    best_config = config_space[np.argmin(config_loss)]

    return best_config


def testing_part(df: pd.DataFrame, date_scheme: tuple[np.ndarray, np.ndarray], config: dict, loss_dic: dict):
    y_test_total = []
    y_pred_total = []
    ts_test_total = []

    for date_set in date_scheme:
        train, test = train_val_test(df, date_set)
        scaler = StandardScaler()

        train = train.drop(columns=["ticker", "sip_timestamp"])
        test = test.drop(columns=["ticker", "sip_timestamp"])

        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        X_train, y_train = create_sequences(train, lstm_config["seq_length"])
        X_test, y_test = create_sequences(train, lstm_config["seq_length"])

        model = create_lstm(**config)
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=100, batch_size=32)
        model.summary()

        y_pred = model.predict(X_test)
        y_pred_total.append(y_pred)
        y_test_total.append(y_test)

        ts = pd.to_datetime(test["sip_timestamp"].iloc[lstm_config["seq_length"]:]) # used for plotting
        ts_test_total.append(ts)


    create_plot(ticker, ts_test_total, y_test_total, y_pred_total)

    y_test_total = np.array(y_test_total).flatten()
    y_pred_total = np.array(y_pred_total).flatten()

    mse = np.mean((y_test_total - y_pred_total)**2)
    rmse = np.sqrt(mse)
    normalized_mse = rmse / np.std(y_test_total)
    mae = np.mean(np.abs(y_test_total - y_pred_total))

    loss_dic["ticker"].append(ticker)
    loss_dic["mse"].append(mse)
    loss_dic["rmse"].append(rmse)
    loss_dic["normalized rmse"].append(normalized_mse)
    loss_dic["mae"].append(mae)  

    return loss_dic


def main():
    df = get_data()

    tickers = df["ticker"].unique()
    date_scheme = get_dates_for_training_scheme(df)

    loss_dic = {"ticker": [], "mse": [], "rmse": [], "normalized rmse": [], "mae": []}

    num_days_testing = 6

    i = 0
    for ticker in tickers:
        if i == 1:
            break
        i += 1
        print(f"Analyzing {ticker}")

        ticker_data = df[df["ticker"] == ticker]

        date_scheme_val = date_scheme[:-num_days_testing]
        date_scheme_test = date_scheme[-num_days_testing:]

        best_config = tuning_part(ticker_data, date_scheme_val)
        loss_dic = testing_part(ticker_data, date_scheme_test, best_config, loss_dic)

    df_loss = pd.DataFrame(loss_dic)
    create_loss_report(df_loss)


def main():
    pass

if __name__ == "__main__":
    ticker = "ADSK"
    df = get_data()
    df = df[df["ticker"] == ticker]
    dates = get_dates_for_training_scheme(df)
    train, val = train_val_test(df, dates[0])
    train = train.drop(columns=["sip_timestamp", "ticker"])
    val = val.drop(columns=["sip_timestamp", "ticker"])
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    print(scaler.mean_, scaler.var_)
    val = scaler.transform(val)
    print(scaler.mean_, scaler.var_)
    
    