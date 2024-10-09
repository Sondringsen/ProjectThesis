import pandas as pd

class DataReader:
    # Exchange codes mapping
    exchange_codes = {
        1: 'Nasdaq',
        2: 'NYSE',
        3: 'NYSE American',
        4: 'FINRA',
        5: 'Nasdaq OMX',
        6: 'NYSE National',
        7: 'Cboe',
        8: 'NYSE Arca',
        9: 'Investors Exchange',
        10: 'International Securities Exchange',
        11: 'Cboe BATS',
        13: 'Nasdaq Philadelphia'
    }

    # Trade conditions
    trade_conditions = ['A', 'B', 'C', 'D', 'E', 'I', 'L', 'M', 'N', 'O', 'P', 'S', 'T']

    def __init__(self):
        """Initialize the DataReader with no parameters."""
        pass

    def _parse_trade_conditions(self, conditions: str):
        """Parse trade conditions and return a dictionary with flags."""
        if pd.isna(conditions):
            conditions = ""
        
        # Split the trade conditions by "-"
        condition_list = conditions.split('-')

        # Return a dictionary indicating which conditions are present
        return {cond: cond in condition_list for cond in self.trade_conditions}

    def parse(self, file_path: str, file_type: str):
        """Parse the CSV file based on the file type and return a DataFrame."""
        if file_type == "trade":
            return self._parse_trade_data(file_path)
        elif file_type == "quote":
            return self._parse_quote_data(file_path)
        else:
            raise ValueError("Invalid file_type. Use 'trade' or 'quote'.")

    def _parse_trade_data(self, file_path: str):
        """Parse trade data and return a DataFrame."""
        data = pd.read_csv(
            file_path, 
            header=None, 
            names=['timestamp', 'price', 'volume', 'exchange_code', 'trade_conditions']
        )
        
        # Convert exchange code to exchange name
        data['exchange'] = data['exchange_code'].map(self.exchange_codes)
        
        # Ensure trade_conditions is a string and handle missing values
        data['trade_conditions'] = data['trade_conditions'].fillna("").astype(str)
        
        # Separate trade conditions into individual columns
        condition_flags = data['trade_conditions'].apply(self._parse_trade_conditions).apply(pd.Series)
        data = pd.concat([data, condition_flags], axis=1)
        
        # Drop the original exchange code and trade conditions columns
        data.drop(['exchange_code', 'trade_conditions'], axis=1, inplace=True)
        
        return data

    def _parse_quote_data(self, file_path: str):
        """Parse quote data and return a DataFrame."""
        data = pd.read_csv(
            file_path, 
            header=None, 
            names=['timestamp', 'bid_price', 'bid_volume', 'bid_exchange_code', 
                   'offer_price', 'offer_volume', 'offer_exchange_code']
        )
        
        # Convert exchange codes to exchange names
        data['bid_exchange'] = data['bid_exchange_code'].map(self.exchange_codes)
        data['offer_exchange'] = data['offer_exchange_code'].map(self.exchange_codes)
        
        # Drop the original exchange code columns
        data.drop(['bid_exchange_code', 'offer_exchange_code'], axis=1, inplace=True)
        
        return data
