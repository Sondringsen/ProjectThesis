import numpy as np
import pandas as pd

def select_random(n: int):
    """
    Selects random companies from all companies there is metadata on.

    Args:
        n (int): number of random companies
    """
    np.random.seed(42)
    meta_data = pd.read_csv("data/raw/meta_data.csv")
    random_meta_data = meta_data.sample(n)
    random_tickers = random_meta_data[["ticker"]]
    random_tickers.to_csv("data/raw/random_tickers.csv")

def main():
    select_random(100)

if __name__ == "__main__":
    main()