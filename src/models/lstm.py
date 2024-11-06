import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.create_model import create_lstm
from hyperopt import hp, fmin, tpe, space_eval

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
    num_days_val_test = 2

    for i in range(len(dates) - num_days_training):
        date_set.append((dates[i: i + num_days_training], dates[i + num_days_training: i + num_days_training + num_days_val_test]))

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

def create_sequences_modified(data, seq_length):
    """
    Convert DataFrame into sequences of specified length for LSTM input. Makes sure sequences do not 
    contain data from different dates.

    Args: 
        data (np.array): complete data containing both x and y
        seq_length (int): the length of the sequence considered in the lstm.

    Returns:
        (np.array, np.array): x and y datasets with the sequences. 
    """
    X, y = [], []
    for date, group_data in data.groupby("date"):
        group_data = group_data.drop(columns=["date"]).values
        X_date, y_date = create_sequences(group_data, seq_length)
        X.append(X_date)
        y.append(y_date)
    return np.concatenate(X), np.concatenate(y)


def get_config_space():
    """
    Creates a hyperparamater space to search over in tuning the model.

    Note:: Deprecated. Use external library hyperopt.
    """
    seq_length = np.array([5, 10, 20])
    layers = np.array([2, 3, 4])
    units = np.array([4, 8, 16, 32])

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


def tuning_part(df: pd.DataFrame, date_scheme: tuple[np.ndarray, np.ndarray], ticker: str):
    config_space = {}
    config_loss = []

    config_key = 0

    def objective(config):
        nonlocal config_key
        nonlocal config_space
        nonlocal config_loss
        config = {key: int(value) for key, value in config.items()} # the lstm only accepts dtype int, not np.int
        
        config_space[config_key] = config
        config_key += 1
        y_val_total = np.array([])
        y_pred_total = np.array([])

        for date_set in date_scheme:
            train, val = train_val_test(df, date_set)
            scaler = StandardScaler()

            train = train.drop(columns=["ticker", "sip_timestamp"])
            val = val.drop(columns=["ticker", "sip_timestamp"])

            train.loc[:, train.columns != "date"] = scaler.fit_transform(train.loc[:, train.columns != "date"])
            val.loc[:, val.columns != "date"] = scaler.transform(val.loc[:, val.columns != "date"])

            X_train, y_train = create_sequences_modified(train, config["seq_length"])
            X_val, y_val = create_sequences_modified(val, config["seq_length"])

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

        return mse
    
    space = {
        "n_layers": hp.quniform('num_layers', 2, 5, 1),
        "units": hp.quniform("units", 4, 32, 4),
        "seq_length": hp.quniform("seq_length", 5, 30, 5),
    }

    best_config_loss = fmin(objective, space, algo=tpe.suggest, max_evals=20)

    config_loss_df = pd.DataFrame().from_dict(config_space, orient="index")
    config_loss_df = config_loss_df
    config_loss_df["mse"] = config_loss
    config_loss_df["ticker"] = ticker

    best_config_loss = np.array(config_loss)
    best_config = config_space[np.argmin(best_config_loss)]

    return best_config , config_loss_df


def testing_part(df: pd.DataFrame, date_scheme: tuple[np.ndarray, np.ndarray], config: dict, loss_dic: dict, ticker: str):
    ts_plot = []
    y_pred_plot = []
    y_test_plot = []
    y_test_total = np.array([])
    y_pred_total = np.array([])

    config = {key: int(value) for key, value in config.items()} # the lstm only accepts dtype int, not np.int

    for counter, date_set in enumerate(date_scheme):
        if counter == 3:
            break
        train, test = train_val_test(df, date_set)
        scaler = StandardScaler()

        ts = pd.to_datetime(test["sip_timestamp"].iloc[config["seq_length"]:]) # used for plotting
        ts_plot.append(ts)

        train = train.drop(columns=["ticker", "sip_timestamp"])
        test = test.drop(columns=["ticker", "sip_timestamp"])

        train.loc[:, train.columns != "date"] = scaler.fit_transform(train.loc[:, train.columns != "date"])
        test.loc[:, test.columns != "date"] = scaler.transform(test.loc[:, test.columns != "date"])

        X_train, y_train = create_sequences_modified(train, config["seq_length"])
        X_test, y_test = create_sequences_modified(test, config["seq_length"])

        config["n_features"] = X_train.shape[2]

        model = create_lstm(**config)
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=20, batch_size=32)
        model.summary()
        model.save(f"reports/models/{ticker}-{counter}.keras")

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

    num_days_testing = 6
    date_scheme_val = date_scheme[:-num_days_testing]
    date_scheme_test = date_scheme[-num_days_testing:]

    loss_dic = {"ticker": [], "mse": [], "rmse": [], "normalized rmse": [], "mae": []}

    tot_loss_config_df = pd.DataFrame(columns=["ticker", "seq_length", "n_layers", "units", "mse"])

    for ticker in tickers:
        print(f"Analyzing {ticker}")

        ticker_data = df.loc[df["ticker"] == ticker, :].copy()
        ticker_data.loc[:, "date"] = ticker_data["sip_timestamp"].dt.date

        best_config, config_loss_df = tuning_part(ticker_data, date_scheme_val, ticker)
        tot_loss_config_df = pd.concat([tot_loss_config_df, config_loss_df], ignore_index=True)
        tot_loss_config_df.to_csv("reports/config_space_loss.csv")

        loss_dic = testing_part(ticker_data, date_scheme_test, best_config, loss_dic, ticker)

    df_loss = pd.DataFrame(loss_dic)
    create_loss_report(df_loss)



if __name__ == "__main__":
    main()
    
    