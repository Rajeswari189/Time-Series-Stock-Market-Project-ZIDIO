
# import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping


def run_lstm(df, column='Close', epochs=20):
    # Use only the 'Close' column
    data = df[[column]].values
    print(tf.__version__)
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences of 60 time steps
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i - 60:i, 0])
        y.append(data_scaled[i, 0])

    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)

    # Reshape input to 3D for LSTM [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build the LSTM model
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X.shape[1], 1)
        )
    )
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping
    early_stop = EarlyStopping(monitor='loss', patience=3)

    # Fit the model
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
    )

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot results
    os.makedirs('outputs/forecast_plots', exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_actual, label='Actual')
    plt.plot(predictions, label='LSTM Forecast')
    plt.legend()
    plt.title('LSTM Forecasting - Stock Close Price')
    plt.savefig('outputs/forecast_plots/lstm_forecast.png')
    plt.close()

    return predictions
