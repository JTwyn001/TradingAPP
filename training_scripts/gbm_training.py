import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import joblib
import datetime
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Define your parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}


def download_and_preprocess_data(ticker, start_date='2010-01-01', end_date='2024-04-01'):
    data = yf.download(tickers='SPY', start=start_date, end=end_date)

    # RSI
    data['RSI'] = ta.rsi(data['Close'], length=15)

    # Exponential Moving Average
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Moving Average Convergence Divergence
    macd = ta.macd(data['Close'])  # This returns a DataFrame
    data['MACD'] = macd['MACD_12_26_9']

    # Simple Moving Average
    data['SMA50'] = data['Close'].rolling(window=50).mean()

    # Average True Range
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)

    # Target variable for prediction
    data['TargetNextClose'] = data['Close'].shift(-1)

    data.dropna(inplace=True)

    return data[['Close', 'RSI', 'EMA20', 'MACD', 'SMA50', 'ATR']], data['TargetNextClose']



def train_gbm(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def save_model(model, directory, filename):
    joblib.dump(model, f"{directory}/{filename}")


def plot_prediction(y_test, y_pred, title='GBM Predicted vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.reset_index(drop=True), label='Actual')
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Observations')
    plt.ylabel('Next Close Price')
    plt.legend()
    plt.show()


def main():
    ticker = 'SPY'
    features, target = download_and_preprocess_data(ticker)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize your model
    gbm = GradientBoostingRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Retrain on the full training set with the best parameters
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Plotting
    plot_prediction(y_test, y_pred)

    # Save the best model
    model_directory = 'C:/Users/tohye/Documents/Capstone/TradingAPP/gbm_models'
    os.makedirs(model_directory, exist_ok=True)
    model_filename = f'gbm_model_{ticker}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    joblib.dump(best_model, os.path.join(model_directory, model_filename))
    print(f'Model saved to {model_directory}/{model_filename}')


if __name__ == '__main__':
    main()
