# Importing necessary libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Bidirectional
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import forecast_evaluation  # This is a custom module

def db_net_model(X, y, n_steps_in, n_steps_out, epochs=300, file_name='model_prediction.csv'):
    """
    Constructs and trains a hybrid deep dilated CNN bidirectional LSTM model (db-net) for time series forecasting.

    Args:
        X (array): Input data array of shape [samples, timesteps, features].
        y (array): Target data array of shape [samples, n_steps_out].
        n_steps_in (int): Number of input time steps.
        n_steps_out (int): Number of output time steps.
        epochs (int): Number of epochs for training (default=300).
        file_name (str): Name of the CSV file to save predictions (default='model_prediction.csv').

    Returns:
        None
    """

    # Reshaping input data
    n_features = X.shape[2]
    n_seq = 1
    n_steps = n_steps_in
    X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

    # Building the model architecture
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='same',
                                     activation='relu', kernel_initializer='he_uniform'),
                              input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='same',
                                     activation='relu', kernel_initializer='he_uniform')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding='same')))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(50, return_sequences=True, activation='relu', kernel_initializer='he_uniform')))
    model.add(Bidirectional(LSTM(25, activation='relu', kernel_initializer='he_uniform')))
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(n_steps_out, activation='linear'))

    # Compiling the model
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse')

    # Training the model
    model.fit(x_train, y_train, batch_size=64, epochs=epochs, verbose=2)

    # Timing model testing
    start_time = time.time()
    yhat = model.predict(x_test, verbose=2)
    testing_time = time.time() - start_time
    print("Model testing time: {:.4f} seconds".format(testing_time / len(yhat)))

    # Saving predictions to a CSV file
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': yhat.flatten()})
    df.to_csv(file_name, index=False)
    print("CSV file '{}' created successfully.".format(file_name))

    # Evaluating model predictions
    forecast_evaluation.evaluate_forecasts(y_test, yhat)
