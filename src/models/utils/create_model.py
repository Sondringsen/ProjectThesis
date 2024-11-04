from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization

def create_lstm(n_layers: int, units: int, seq_length: int, n_features: int):
    """
    Creates a stacked LSTM model with a specified number of layers and units.

    Args:
        n_layers (int): the number of LSTM layers in the model.
        units (int): the number of units per LSTM layer.
        seq_length (int): the length of the input sequence.
        n_features (int): the number of features in each input sequence.

    Returns:
        tf.keras.Sequential: A model with the specified number of LSTM layers.
    """
    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))

    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
        model.add(BatchNormalization())

    model.add(Dense(1))

    return model
