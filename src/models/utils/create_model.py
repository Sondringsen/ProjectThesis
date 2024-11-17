from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, SimpleRNN
from tensorflow.keras import regularizers

def create_lstm(n_layers: int, units: int, seq_length: int, n_features: int, l2_reg: float):
    """
    Creates a stacked LSTM model with a specified number of layers and units.

    Args:
        n_layers (int): the number of LSTM layers in the model.
        units (int): the number of units per LSTM layer.
        seq_length (int): the length of the input sequence.
        n_features (int): the number of features in each input sequence.
        l2_reg (float): the L2 regularization coefficient.

    Returns:
        tf.keras.Sequential: A model with the specified number of LSTM layers.
    """
    # something wrong with the keras which make it not accept numpy ints
    n_layers = int(n_layers)
    units = int(units)
    seq_length = int(seq_length)
    n_features = int(n_features)


    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))

    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        model.add(
            LSTM(
                units, 
                activation='relu', 
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.L2(l2_reg), 
                bias_regularizer=regularizers.L2(l2_reg), 
                )
            )
        model.add(BatchNormalization())

    model.add(Dense(1, kernel_regularizer=regularizers.L2(l2_reg), bias_regularizer=regularizers.L2(l2_reg)))

    return model

def create_rnn(n_layers: int, units: int, seq_length: int, n_features: int, l2_reg: float):
    """
    Creates a stacked RNN model with a specified number of layers and units.

    Args:
        n_layers (int): the number of RNN layers in the model.
        units (int): the number of units per RNN layer.
        seq_length (int): the length of the input sequence.
        n_features (int): the number of features in each input sequence.
        l2_reg (float): the L2 regularization coefficient.

    Returns:
        tf.keras.Sequential: A model with the specified number of RNN layers.
    """
    # something wrong with the keras which make it not accept numpy ints
    n_layers = int(n_layers)
    units = int(units)
    seq_length = int(seq_length)
    n_features = int(n_features)
    
    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))

    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        model.add(SimpleRNN(units, activation='relu', return_sequences=return_sequences), regularizers.L2(l2_reg))
        model.add(BatchNormalization())

    model.add(Dense(1, kernel_regularizer=regularizers.L2(l2_reg), bias_regularizer=regularizers.L2(l2_reg)))

    return model