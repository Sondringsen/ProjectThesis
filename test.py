import pandas as pd

loss_dic = {"ticker": ["AAPL", "AMZN"], "mse": [3.4235, 5.123]}

df = pd.DataFrame(loss_dic)

print(df)