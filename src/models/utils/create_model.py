from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input


def create_lstm(n_layers: int, units: int, seq_length: int, n_features: int):
    """
    Creates a stacked lstm model.

    Args:
        n_layers (int): the number of lstm layers in the model.
        units (int): the number of units per lstm layer.
        seq_length (int): the length of the sequence considered in the lstm.
        n_features (int): number of features in the x set.

    Returns:
        tf.Sequential: a model with lstm layers.
    """

    model = Sequential([
        Input(shape=(seq_length, n_features)),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        Dense(1)
    ])

    return model


