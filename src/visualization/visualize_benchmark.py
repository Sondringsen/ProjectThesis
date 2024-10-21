import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_benchmark( df_true: pd.DataFrame, df_pred: pd.DataFrame):
    """
    Plots the true values against the predictions of the AR(p) model and stores it.

    Args:
        df_true (pd.DataFrame): entire dataframe.
        df_pred (pd.DataFrame): dataframe of predictions. In this case it is gridded.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df_true["timestamp"], df_true["mid_price"], label="Mid price")
    plt.plot(df_pred["timestamp"], df_pred["prediction"], label="Predictions")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Mid price")
    plt.title("MSFT mid price and predictions")
    plt.savefig("reports/plots/msft_benchamrk.png")


def main():
    df_pred = pd.read_csv("models/predictions/benchmark_ar.csv")
    df_true = pd.read_csv("data/processed/msft.csv")

    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"])
    df_true["timestamp"] = pd.to_datetime(df_true["timestamp"])
    plot_benchmark(df_true, df_pred)

if __name__ == "__main__":
    main()
