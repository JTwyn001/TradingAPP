import os
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


def download_and_preprocess_data(start_date):
    # Current date
    current_date = datetime.datetime.now()
    # Check if today is a weekend day (Saturday=5, Sunday=6)
    if current_date.weekday() >= 5:  # if it's the weekend
        # Roll back to the previous Friday
        end_date = (current_date - datetime.timedelta(days=current_date.weekday() - 4)).strftime('%Y-%m-%d')
    else:
        # Subtract one day from the current date (if it's not a weekend)
        end_date = (current_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    # Download data up to the current date
    data = yf.download(tickers='EURNOK=X', start='2012-02-10', end='2024-03-1')

    print(data.head())  # Add this line to print the first few rows of the DataFrame

    # Add indicators
    data['RSI'] = ta.rsi(data['Close'], length=15)
    data['EMAF'] = ta.ema(data['Close'], length=20)
    data['EMAM'] = ta.ema(data['Close'], length=100)
    data['EMAS'] = ta.ema(data['Close'], length=150)

    # Target variable
    data['TargetNextClose'] = data['Adj Close'].shift(-1)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Select features and target
    features = data[['Adj Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values
    target = data['TargetNextClose']

    return features, target


def scale_data(features, target):
    # Initialize scalers
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform features
    features_scaled = feature_scaler.fit_transform(features)

    # Convert target to numpy array and reshape
    target_array = target.values.reshape(-1, 1)  # Convert to numpy array and reshape

    # Fit and transform target
    target_scaled = target_scaler.fit_transform(target_array)

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

    # Proceed with data downloading and preprocessing using the determined start date
    features, target = download_and_preprocess_data(start_date)

    # Check if features and target were successfully retrieved
    if features is None or target is None:
        print("Data retrieval was unsuccessful. Exiting the script.")
        return  # Exit the script as we cannot proceed without data

    features_scaled, target_scaled, feature_scaler, target_scaler = scale_data(features, target)
    X, y = create_sequences(features_scaled, target_scaled)

    # Splitting data into training and test sets
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = build_and_train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plotting the results
    plot_prediction(y_test, y_pred)

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    ticker_symbol = 'EURNOK'
    # Include the ticker symbol and timestamp in the filenames for clarity
    # Create directories if they don't exist
    # Ensure directories for models and scalers exist
    models_dir = 'C:/Users/tohye/Documents/Capstone/TradingAPP/models'
    scalers_dir = 'C:/Users/tohye/Documents/Capstone/TradingAPP/scalers'

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)

    # Use absolute paths for saving
    model_filename = os.path.join(models_dir, f'lstm_model_{ticker_symbol}_{timestamp}.keras')
    feature_scaler_filename = os.path.join(scalers_dir, f'feature_scaler_{ticker_symbol}_{timestamp}.pkl')
    target_scaler_filename = os.path.join(scalers_dir, f'target_scaler_{ticker_symbol}_{timestamp}.pkl')

    # Your code to save model and scalers...
    model.save(model_filename, save_format='tf')
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
