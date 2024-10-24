import pandas as pd
import numpy as np

class DataProcessor():
    def create_training_files(self, date, tickers, start_time_of_day, end_time_of_day, num_eval_timestamps):
        path = f"data/raw/{date}.csv"
        df_all = pd.read_csv(path)
        for ticker in tickers:
            df = df_all[df_all.ticker == ticker]
            df.loc[:, "sip_timestamp"] = pd.to_datetime(df.sip_timestamp)
            df = df[(df.sip_timestamp >= start_time_of_day) & (df.sip_timestamp <= end_time_of_day)].sort_values(by="sip_timestamp")
            ts_eval = pd.to_datetime(np.random.uniform(pd.Timestamp(start_time_of_day).value, pd.Timestamp(end_time_of_day).value, num_eval_timestamps))
            ts_eval.sort_values(inplace=True)
            price_after_ts = np.searchsorted(df.sip_timestamp.values, ts_eval.values)
            df_eval = [{"timestamp_eval": t, "price": df.iloc[idx].price if idx < len(df) else np.nan} for t, idx in zip(ts_eval, price_after_ts)]
            df_eval = pd.DataFrame(df_eval)
            save_path = f"data/processed/{ticker}_{date}_eval.csv"
            df_eval.to_csv(save_path, index=False)

            save_path = f"data/processed/{ticker}_{date}_obs.csv"
            df.to_csv(save_path, index=False)





