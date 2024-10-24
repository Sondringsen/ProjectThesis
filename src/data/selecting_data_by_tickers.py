import pandas as pd
import numpy as np
import os

# Load tickers
tickers = pd.read_csv("data/raw/random_tickers.csv")
trade_files = os.listdir("data/raw/quotes")
trade_files = [f for f in trade_files if not f.startswith('.')]

# Define the chunk size for reading the large CSV file
chunk_size = 1e8  # You can adjust the chunk size based on your memory limits

for filename in trade_files:
    print(f"Processing {filename}...")
    
    filepath = os.path.join("data/raw/quotes", filename)
    
    # Create an empty list to store filtered chunks
    filtered_chunks = []
    i = 0
    # Read the CSV file in chunks
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        print(i)
        i+=1
        # Filter the chunk for the required tickers and columns
        filtered_chunk = chunk[chunk['ticker'].isin(tickers['ticker'])][["ticker", "sip_timestamp", "ask_price", "ask_size", "bid_price", "bid_size"]]
        
        # Append the filtered chunk to the list
        filtered_chunks.append(filtered_chunk)
    
    # Concatenate all the filtered chunks into a single DataFrame
    df2 = pd.concat(filtered_chunks)
    df2.to_csv(filepath)
    
    