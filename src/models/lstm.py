import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.create_model import create_lstm

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
    seq_length = np.array([10])
    layers = np.array([3, 5, 7])
    units = np.array([8, 16, 32, 64])
    config_space = np.array(np.meshgrid(seq_length, layers, units)).T.reshape(-1, 3)
    
    config_names = ["seq_length", "n_layers", "units"]
    config_space = {x: {config_names[i]: config_space[x, i] for i in range(len(config_names))} for x in range(config_space.shape[0])}

    return config_space


def create_loss_report(df_loss: pd.DataFrame):
    """
    Creates a latex table containing loss metrics for all tickers.
    
    Args:
        df_loss (pd.DataFrame): a dataframe containing different loss metrics for all tickers.
    """
    with open("reports/loss_tot_lstm.tex", "w") as file:
        file.write(df_loss.to_latex(index=False, header=True, float_format="%.4f", caption="Loss Metrics For All Tickers LSTM-model", label="tab:loss_total_lstm"))


def create_plot(ticker: str, ts: list[pd.Series], y_test: list[np.ndarray], y_pred: list[np.ndarray]):
    """
    Creates and saves a plot of the true values and the predicted values over the test period.

    Args:
        ticker (str): the symbol/ticker of the stock under analysis.
        ts (list[pd.Series]): list of series of timestamps corresponding to the y values.
        y_test (list[np.ndarray]): list of numpy arrays of true values.
        y_pred (list[np.ndarray]): list of numpy arrays of predicted values.
    """
    plt.figure(figsize=(12, 8))
    for ts, test, pred in zip(ts, y_test, y_pred):
        plt.plot(ts, test, color = "red")
        plt.plot(ts, pred, color = "blue")
    plt.plot([], [], color="red", label="True values")
    plt.plot([], [], color="blue", label="Predicted values")
    plt.legend()
    plt.title(f"LSTM model prediction for {ticker} (min-max-scaled values)")
    plt.ylabel("Midprice")
    plt.xlabel("Time")

    plt.savefig(f"reports/plots/{ticker}_lstm.png")


def tuning_part(df: pd.DataFrame, date_scheme: tuple[np.ndarray, np.ndarray]):
    config_space = get_config_space()

    config_loss = []

    for index, config in config_space.items():
        if index == 2:
            break
        y_val_total = np.array([])
        y_pred_total = np.array([])

        # lstm_parameters = ["seq_length", "n_layers", "units"]
        # config_lstm = {lstm_parameter: config[lstm_parameter] for lstm_parameter in lstm_parameters}
        config = {key: int(value) for key, value in config.items()} # the lstm only accepts dtype int, not np.int
        counter = 0
        for date_set in date_scheme:
            if counter == 5:
                break
            counter += 1
            train, val = train_val_test(df, date_set)
            scaler = StandardScaler()

            train = train.drop(columns=["ticker", "sip_timestamp"])
            val = val.drop(columns=["ticker", "sip_timestamp"])

            train = scaler.fit_transform(train)
            val = scaler.transform(val)

            X_train, y_train = create_sequences(train, config["seq_length"])
            X_val, y_val = create_sequences(val, config["seq_length"])

            config["n_features"] = X_train.shape[2]

            model = create_lstm(**config)
            model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train, epochs=20, batch_size=32)
            model.summary()

            y_pred = model.predict(X_val)

            y_pred_total = np.concatenate([y_pred_total, y_pred.flatten()])
            y_val_total = np.concatenate([y_val_total, y_val])

        mse = np.mean((y_val_total - y_pred_total)**2)
        config_loss.append(mse)
        
    config_loss = np.array(config_loss)
    best_config = config_space[np.argmin(config_loss)]

    return best_config


def testing_part(df: pd.DataFrame, date_scheme: tuple[np.ndarray, np.ndarray], config: dict, loss_dic: dict, ticker: str):
    ts_plot = []
    y_pred_plot = []
    y_test_plot = []
    y_test_total = np.array([])
    y_pred_total = np.array([])
    # ts_test_total = np.array([], dtype='datetime64[s]')

    config = {key: int(value) for key, value in config.items()} # the lstm only accepts dtype int, not np.int

    counter = 0
    for date_set in date_scheme:
        if counter == 3:
            break
        counter += 1
        train, test = train_val_test(df, date_set)
        scaler = StandardScaler()

        ts = pd.to_datetime(test["sip_timestamp"].iloc[config["seq_length"]:]) # used for plotting
        # ts_test_total = np.concatenate([ts_test_total, ts])
        ts_plot.append(ts)

        train = train.drop(columns=["ticker", "sip_timestamp"])
        test = test.drop(columns=["ticker", "sip_timestamp"])

        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        X_train, y_train = create_sequences(train, config["seq_length"])
        X_test, y_test = create_sequences(test, config["seq_length"])

        config["n_features"] = X_train.shape[2]

        model = create_lstm(**config)
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=20, batch_size=32)
        model.summary()

        y_pred = model.predict(X_test)

        y_pred_total = np.concatenate([y_pred_total, y_pred.flatten()])
        y_test_total = np.concatenate([y_test_total, y_test])

        y_pred_plot.append(y_pred)
        y_test_plot.append(y_test)


    create_plot(ticker, ts_plot, y_test_plot, y_pred_plot)

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
        if i == 3:
            break
        i += 1
        print(f"Analyzing {ticker}")

        ticker_data = df[df["ticker"] == ticker]

        date_scheme_val = date_scheme[:-num_days_testing]
        date_scheme_test = date_scheme[-num_days_testing:]

        # best_config = tuning_part(ticker_data, date_scheme_val)

        best_config = {'seq_length': 10, 'n_layers': 3, 'units': 8}
        loss_dic = testing_part(ticker_data, date_scheme_test, best_config, loss_dic, ticker)

    df_loss = pd.DataFrame(loss_dic)
    create_loss_report(df_loss)


if __name__ == "__main__":
    main()
    
    