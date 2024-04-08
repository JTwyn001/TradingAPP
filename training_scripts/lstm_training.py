import os
import numpy as np
import pandas as pd
import datetime
import MetaTrader5 as mt
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


def download_and_preprocess_data(symbol, start_date, end_date):
    # Initialize MT connection
    if not mt.initialize():
        print("initialize() failed, error code =", mt.last_error())
        
    # Set the symbol and timeframe
    timeframe = mt.TIMEFRAME_D1  # Daily timeframe
    # Fetch historical data
    rates = mt.copy_rates_range(symbol, timeframe, start_date, end_date)
    # Check if data was fetched successfully
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}")
        return None, None
    # Convert to DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)

    # Add indicators
    data['RSI'] = ta.rsi(data['close'], length=15)
    data['EMAF'] = ta.ema(data['close'], length=20)
    data['EMAM'] = ta.ema(data['close'], length=100)
    data['EMAS'] = ta.ema(data['close'], length=150)

    # Target variable
    data['TargetNextClose'] = data['close'].shift(-1)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Select features and target
    features = data[['close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values
    target = data['TargetNextClose'].values

    return features, target


def scale_data(features, target):
    # Initialize scalers
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform features
    features_scaled = feature_scaler.fit_transform(features)

    # Ensure target is a 2D array. It's already a NumPy array, so just reshape it.
    target_reshaped = target.reshape(-1, 1)  # This line ensures target is in the correct shape

    # Fit and transform target
    target_scaled = target_scaler.fit_transform(target_reshaped)

    # Return the scaler objects along with the scaled data
    return features_scaled, target_scaled, feature_scaler, target_scaler


def create_sequences(features_scaled, target_scaled, backcandles=30):
    X, y = [], []
    for i in range(backcandles, len(features_scaled)):
        X.append(features_scaled[i - backcandles:i])
        y.append(target_scaled[i, 0])
    return np.array(X), np.array(y)


def build_and_train_model(X_train, y_train):
    # LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit model
    model.fit(X_train, y_train, epochs=30, batch_size=15, validation_split=0.1, shuffle=True)

    return model


def plot_prediction(y_test, y_pred):
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Actual')
    plt.plot(y_pred, color='green', label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def main():
    symbol = 'DAY.NYSE'
    start_date = '2012-01-01'  # Assuming you want to start from this date
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')  # Today's date in 'YYYY-MM-DD' format
    # Path to a file where the last training timestamp is saved
    timestamp_file = './last_training_timestamp.txt'

    start_date = '2012-01-01'
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            last_date = f.read().strip()
        try:
            datetime.datetime.strptime(last_date, '%Y-%m-%d')
            start_date = last_date
        except ValueError:
            print("Invalid date in timestamp file. Using default start date.")

    print(f"Training model for {symbol}...")
    # Proceed with data downloading and preprocessing using the determined start date
    features, target = download_and_preprocess_data(symbol, start_date, end_date)

    # Check if features and target were successfully retrieved
    if features is None or target is None:
        print("Data retrieval was unsuccessful.")
        return

    features_scaled, target_scaled, feature_scaler, target_scaler = scale_data(features, target)
    X, y = create_sequences(features_scaled, target_scaled)
    X_train, X_test, y_train, y_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):], y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

    model = build_and_train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_prediction(y_test, y_pred)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'./models/lstm_model_{symbol}_{timestamp}.h5'
    feature_scaler_filename = f'./scalers/feature_scaler_{symbol}_{timestamp}.pkl'
    target_scaler_filename = f'./scalers/target_scaler_{symbol}_{timestamp}.pkl'

    model.save(model_filename)
    joblib.dump(feature_scaler, feature_scaler_filename)
    joblib.dump(target_scaler, target_scaler_filename)

    # After training, update the timestamp file with the current date
    with open(timestamp_file, 'w') as f:
        f.write(datetime.datetime.now().strftime('%Y-%m-%d'))

    # Optionally print out the filenames for confirmation
    print(f"Model saved as: {model_filename}")
    print(f"Feature Scaler saved as: {feature_scaler_filename}")
    print(f"Target Scaler saved as: {target_scaler_filename}")


if __name__ == '__main__':
    main()
