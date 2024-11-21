import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt

def get_data():
    """
    Reads the meta data and the loss metrics into memory.

    Returns:
        tuple[pd.DataFrame, pd.DataFram]: dataframes with all exogenous and endogenous variables respectively.
    """
    X = pd.read_csv("data/raw/meta_data.csv", index_col=1).iloc[:, 1:]
    Y = pd.read_csv("reports/tot_loss_lstm.csv", index_col=0)["normalized_rmse"]

    return X, Y

def clean_data(X: pd.DataFrame, Y: pd.DataFrame, exog: list):
    """
    Cleans the data. Filters out the variabels not used in the regression. Replaces certain 
    nan-values. Also deletes some tickers with nan-values.

    Args:
        X (pd.DataFrame): all exogenous variables.
        Y (pd.DataFrame): all endogenous variable.
        exog (list): list of exogenous variables used in the regression.

    Returns:
        tuple[pd.DataFrame, pd.DataFram]: clean dataframes with exogenous and endogenous variables respectively ready to use for regression.
    """
    X = X[exog]

    tickers = Y.index
    X = X.loc[tickers, :]
    X = add_constant(X)

    # We need to be careful here
    X.loc[:, "dividendYield"] = X["dividendYield"].replace(np.nan, 0)
    X.loc[:, "numberOfAnalystOpinions"] = X["numberOfAnalystOpinions"].replace(np.nan, 0)
    X.loc[:, "overallRisk"] = X["overallRisk"].replace(np.nan, X["overallRisk"].mean())

    # Drop tickers with nan-values (should maybe not do this)
    # tickers_to_drop = X.isna().sum(axis=1).astype(bool)
    # X = X[~tickers_to_drop]
    # Y = Y[~tickers_to_drop]

    return X, Y

def regression(X: pd.DataFrame, Y: pd.DataFrame):
    """
    Performs an OLS regression on X and Y with robust standard errors. Saves the summary
    as a png-file.

    Args:
        X (pd.DataFrame): exogenous variables.
        Y (pd.DataFrame): endogenous variable.
    """
    model = OLS(Y, X)
    result = model.fit(cov_type='HC3')
    result.summary()

    summary_text = result.summary().as_text()

    plt.figure(figsize=(10, 6))
    plt.text(0.01, 0.99, summary_text, fontsize=10, family='monospace', verticalalignment='top')
    plt.axis('off')

    plt.savefig("model_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    X, Y = get_data()
    
    exog = [
        "fullTimeEmployees",
        "overallRisk", 
        "dividendYield", 
        "marketCap", 
        "shortRatio", 
        "heldPercentInsiders", 
        "heldPercentInstitutions", 
        "numberOfAnalystOpinions"
    ]

    X, Y = clean_data(X, Y)

    regression(X, Y)


if __name__ == "__main__":
    main()