import datetime
import json
import logging

from flask import Flask, request, jsonify, session, render_template
# from flask_cors import CORS
import os
import platform
import openai
import MetaTrader5 as mt
from mt5 import (market_order, get_top_10_momentum_stocks, get_top_10_momentum_forex,
                 close_order, get_exposure)
import time
import numpy as np  # The Numpy numerical computing library
import pandas as pd  # The Pandas data science library
import pandas_ta as ta
import requests  # The requests library for HTTP requests in Python
import xlsxwriter  # The XlsxWriter library for
import math  # The Python math module
from scipy import stats  # The SciPy stats module
from pandas.tseries.offsets import BDay
import webbrowser
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from tensorflow.keras.models import load_model
import joblib

mt.initialize()

if not mt.initialize():
    print("initialize() failed, error code =", mt.last_error())
    quit()
else:
    print('Connected to MetaTrader5')

login = 51684010
password = 'd&CISL465!tBzO'
server = 'ICMarketsSC-Demo'

mt.login(login, password, server)

openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize the Flask app and set the template folder
app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)
# Set a secret key for session handling
app.secret_key = 'algotradingproject'


@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have 'index.html' in the templates folder


@app.route('/trading-ai')
def trading_ai():
    sp500_stocks = pd.read_csv('mt5_stock_tickers.csv')
    top_10_stocks = sp500_stocks['Ticker'].head(10).tolist()  # Assuming the column name is 'Ticker'
    return render_template('trading_ai.html', top_10_stocks=top_10_stocks)


@app.route('/trading-ai')
def trading_ai_forex():
    sp500_forex = pd.read_csv('mt5_forex_tickers.csv')
    top_10_forex = sp500_forex['Ticker'].head(10).tolist()  # Assuming the column name is 'Ticker'
    return render_template('trading_ai.html', top_10_forex=top_10_forex)


@app.route('/get_data/<symbol>')
def get_data(symbol):
    try:
        # Ensure MT5 is initialized
        if not mt.initialize():
            print("initialize() failed, error code =", mt.last_error())
            return {'error': 'Failed to initialize MT5'}

        # Fetch the most recent bars from MetaTrader 5
        rates = mt.copy_rates_from_pos(symbol, mt.TIMEFRAME_M1, 0, 1440)  # Last day's data, assuming 1 min timeframe

        if rates is None or len(rates) == 0:
            return {'error': 'No data available for the symbol'}

        # Convert the rates to a list of dictionaries
        response_data = []
        for rate in rates:
            rate_dict = {
                'time': datetime.datetime.fromtimestamp(rate['time']).strftime('%Y-%m-%d %H:%M:%S'),
                'open': rate['open'],
                'high': rate['high'],
                'low': rate['low'],
                'close': rate['close'],
                'volume': rate['tick_volume']  # or just 'volume' if using real volume
            }
            response_data.append(rate_dict)

        return response_data  # Return the list of dictionaries directly
    except Exception as e:
        print(f"Error: {str(e)}")  # Print the actual error message
        return {'error': str(e)}


# Flask route to scan market and get top 10 momentum stocks
@app.route('/scan-market')
def scan_market():
    try:
        top_10_stocks = get_top_10_momentum_stocks()
        print("Top 10 Momentum Stocks:", top_10_stocks)
        return jsonify(top_10_stocks)
    except Exception as e:
        return jsonify({'error': str(e)})


# Flask route to scan forex market and get top 10 momentum forex
@app.route('/scan-forex-market')
def scan_forex_market():
    try:
        top_10_forex = get_top_10_momentum_forex()
        print("Top 10 Momentum Forex:", top_10_forex)
        return jsonify(top_10_forex)
    except Exception as e:
        return jsonify({'error': str(e)})


# Function to fetch recent data
def fetch_recent_data(ticker, window_size=30):
    if not mt.initialize():
        print("MT5 initialize() failed, error code =", mt.last_error())
        return None

    try:
        # Define the date range for data fetching
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=window_size)

        # Convert to POSIX timestamps
        utc_from = int(datetime.datetime.timestamp(start_date))
        utc_to = int(datetime.datetime.timestamp(end_date))

        # Fetch historical data from MT5
        rates = mt.copy_rates_range(ticker, mt.TIMEFRAME_D1, utc_from, utc_to)

        if rates is None or len(rates) == 0:
            print(f"No recent data fetched for {ticker}. Skipping...")
            return None

        # Convert to DataFrame
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data.set_index('time', inplace=True)

        # Add technical indicators
        data['RSI'] = ta.rsi(data['close'], length=15)
        data['EMAF'] = ta.ema(data['close'], length=20)
        data['EMAM'] = ta.ema(data['close'], length=100)
        data['EMAS'] = ta.ema(data['close'], length=150)

        # Clean up any missing data
        data.dropna(inplace=True)

        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


@app.route('/execute_lstm_predictions', methods=['GET'])
def execute_lstm_predictions():
    tickers = ['EURUSD', 'SEKJPY', 'EURNOK', 'MU', 'NRG']
    predictions = {}
    mt5_tick = ['EURUSD', 'SEKJPY', 'EURNOK', 'MU.NAS', 'NRG.NYSE']
    for ticker_symbol in mt5_tick:
        # Load model and scalers for each ticker
        model, feature_scaler, target_scaler = load_latest_model_and_scaler_for_ticker(ticker_symbol)

        # Preprocess data for each ticker
        prepared_data = preprocess_data_for_lstm_mt(ticker_symbol, feature_scaler)

        if prepared_data is not None:
            # Get prediction for each ticker
            predicted_change_scaled = model.predict(prepared_data)
            predicted_change = target_scaler.inverse_transform(predicted_change_scaled.reshape(-1, 1))[0][0]

            # Convert numpy.float32 to Python float for JSON serialization
            predictions[ticker_symbol] = float(predicted_change)

    # Return the predictions for all tickers
    return jsonify(predictions)


def preprocess_data_for_lstm_mt(ticker, feature_scaler):
    if not mt.initialize():
        print("MT5 initialize() failed, error code =", mt.last_error())
        return None

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)  # Example: last 365 days

    utc_from = int(start_date.timestamp())
    utc_to = int(end_date.timestamp())

    rates = mt.copy_rates_range(ticker, mt.TIMEFRAME_D1, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        error_code = mt.last_error()
        print(f"No recent data fetched for {ticker}. Error: {error_code}")
        return None

    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)

    # Same indicators as used during training
    data['RSI'] = ta.rsi(data['close'], length=15)
    data['EMAF'] = ta.ema(data['close'], length=20)
    data['EMAM'] = ta.ema(data['close'], length=100)
    data['EMAS'] = ta.ema(data['close'], length=150)
    data.dropna(inplace=True)

    features = data[['close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values[-30:]  # Last 30 days

    features_scaled = feature_scaler.transform(features)
    features_reshaped = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])

    return features_reshaped


def load_latest_model_and_scaler_for_ticker(ticker_symbol, model_dir='./models', scaler_dir='./scalers'):
    # Adjusted to look for '.keras' files instead of '.h5'
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.keras') and ticker_symbol in f]
    feature_scaler_files = [os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir) if 'feature_scaler' in f and f.endswith('.pkl') and ticker_symbol in f]
    target_scaler_files = [os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir) if 'target_scaler' in f and f.endswith('.pkl') and ticker_symbol in f]

    # Ensure there are files before using max to avoid ValueError
    latest_model_file = max(model_files, key=os.path.getctime) if model_files else None
    latest_feature_scaler_file = max(feature_scaler_files, key=os.path.getctime) if feature_scaler_files else None
    latest_target_scaler_file = max(target_scaler_files, key=os.path.getctime) if target_scaler_files else None

    # Load the latest files if they exist
    model = tf.keras.models.load_model(latest_model_file) if latest_model_file else None
    feature_scaler = joblib.load(latest_feature_scaler_file) if latest_feature_scaler_file else None
    target_scaler = joblib.load(latest_target_scaler_file) if latest_target_scaler_file else None

    return model, feature_scaler, target_scaler


# Function to load the latest model and scaler
def load_latest_model_and_scaler(model_dir='./models', scaler_dir='./scalers'):
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.h5')]
    feature_scaler_files = [os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir) if
                            'feature_scaler' in f and f.endswith('.pkl')]
    target_scaler_files = [os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir) if
                           'target_scaler' in f and f.endswith('.pkl')]

    # Ensure there are files before using max to avoid ValueError
    latest_model_file = max(model_files, key=os.path.getctime) if model_files else None
    latest_feature_scaler_file = max(feature_scaler_files, key=os.path.getctime) if feature_scaler_files else None
    latest_target_scaler_file = max(target_scaler_files, key=os.path.getctime) if target_scaler_files else None

    # Load the latest files if they exist
    model = load_model(latest_model_file) if latest_model_file else None
    feature_scaler = joblib.load(latest_feature_scaler_file) if latest_feature_scaler_file else None
    target_scaler = joblib.load(latest_target_scaler_file) if latest_target_scaler_file else None

    return model, feature_scaler, target_scaler


def preprocess_data_for_lstm(ticker, feature_scaler):
    if not mt.initialize():
        print("MT5 initialize() failed, error code =", mt.last_error())
        return None

    try:
        # Define the date range for data fetching
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * 10)  # Fetching data for the last 10 years

        # Convert to POSIX timestamps
        utc_from = int(datetime.datetime.timestamp(start_date))
        utc_to = int(datetime.datetime.timestamp(end_date))

        # Fetch historical data from MT5
        rates = mt.copy_rates_range(ticker, mt.TIMEFRAME_D1, utc_from, utc_to)

        if rates is None or len(rates) == 0:
            print(f"No recent data fetched for {ticker}. Skipping...")
            return None

        # Convert to DataFrame
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data.set_index('time', inplace=True)

        # Adding indicators
        data['RSI'] = ta.rsi(data['close'], length=15)
        data['EMAF'] = ta.ema(data['close'], length=20)
        data['EMAM'] = ta.ema(data['close'], length=100)
        data['EMAS'] = ta.ema(data['close'], length=150)

        # Ensure all data is complete
        data.dropna(inplace=True)

        # Select relevant features for predictions
        features = data[['close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values

        # Use the passed feature_scaler to scale the features
        features_scaled = feature_scaler.transform(features)
        print("Shape of features_scaled:", features_scaled.shape)  # Print the shape

        # Ensure there's enough data to create a sequence for LSTM
        if features_scaled.shape[0] < 30:
            print(f"Not enough data to create a sequence for {ticker}. Needed 30, got {features_scaled.shape[0]}")
            return None  # Return None to indicate insufficient data

        features_reshaped = features_scaled[-30:].reshape(1, 30, features_scaled.shape[1])

        return features_reshaped

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


# Global dictionary to store prediction values
prediction_values = {}


# Function for LSTM prediction
@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    global prediction_values
    # Extract the selected stocks from the request
    selected_stocks = request.json['selectedStocks']
    # Load the latest model and scaler dynamically
    model, feature_scaler, target_scaler = load_latest_model_and_scaler()  # Assuming you have separate scalers for
    # features and target

    predictions = []
    for ticker in selected_stocks:
        try:
            prepared_data = preprocess_data_for_lstm(ticker, feature_scaler)  # Pass feature_scaler here
            if prepared_data is None:
                print(f"Prepared data for {ticker} is None.")
                continue

            # Get predictions from the model
            predicted_change_scaled = model.predict(prepared_data)
            predicted_change = target_scaler.inverse_transform(predicted_change_scaled.reshape(-1, 1))[0][0]

            # Store the ticker and its prediction
            predictions.append((ticker, predicted_change))

            # Sort the predictions based on the predicted change value
            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

            # Convert to a dictionary for JSON response
            sorted_predictions = {ticker: rank for rank, (ticker, _) in enumerate(sorted_predictions, start=1)}
            print(f"Predicted change for {ticker}: {predicted_change}")
        except Exception as e:
            print(f"Exception processing {ticker}: {e}")

    # Sort the predictions based on the predicted change value
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    ranked_predictions = {ticker: rank for rank, (ticker, _) in enumerate(sorted_predictions, start=1)}

    return jsonify(sorted_predictions)


@app.route('/get_trade_predictions', methods=['POST'])
def get_trade_predictions():
    # Extract the selected stocks from the request
    selected_stocks = request.json['selectedStocks']

    # Load the latest model and scaler dynamically
    model, feature_scaler, target_scaler = load_latest_model_and_scaler()

    trade_predictions = {}
    for ticker in selected_stocks:
        try:
            # Ensure that preprocess_data_for_lstm is fetching and preparing MT5 data correctly
            prepared_data = preprocess_data_for_lstm(ticker, feature_scaler)
            if prepared_data is not None:
                predicted_change_scaled = model.predict(prepared_data)
                predicted_change = target_scaler.inverse_transform(predicted_change_scaled.reshape(-1, 1))[0][0]
                trade_predictions[ticker] = float(predicted_change)
            else:
                print(f"Insufficient or unavailable data for {ticker}. Skipping...")
        except Exception as e:
            print(f"Exception processing {ticker}: {e}")
            # Instead of returning an error string, consider returning a default value or omitting the ticker
            trade_predictions[ticker] = 0  # Default value or use `None` and handle it on the frontend

    return jsonify(trade_predictions)


def allocate_funds(total_funds, prediction_values):
    # Filter out non-numeric and None values from prediction_values
    valid_predictions = {ticker: value for ticker, value in prediction_values.items() if
                         isinstance(value, (int, float))}

    # Normalize prediction values to get allocation percentages
    total_prediction = sum(valid_predictions.values())
    if total_prediction == 0:
        raise ValueError("Total prediction value is 0, unable to allocate funds")

    allocation_percentages = {ticker: (value / total_prediction) for ticker, value in valid_predictions.items()}

    # Allocate funds based on percentages
    allocations = {ticker: total_funds * percentage for ticker, percentage in allocation_percentages.items()}
    return allocations


def find_mt5_symbol(ticker):
    # Define a list of common exchange suffixes used in MT5
    exchange_suffixes = ['.NAS', '.NYSE', '.LSE', '.ASE', '.TSX']  # Add more as needed

    for suffix in exchange_suffixes:
        potential_symbol = f"{ticker}{suffix}"
        print(f"Checking MT5 for symbol: {potential_symbol}")  # Debug print

        if mt.symbol_select(potential_symbol, True):
            symbol_info = mt.symbol_info(potential_symbol)
            if symbol_info is not None:
                print(f"Found and made visible: {potential_symbol}")
                return potential_symbol
            else:
                print(f"Symbol {potential_symbol} could not be made visible.")
        else:
            print(f"Symbol {potential_symbol} does not exist.")

    print(f"No match found for {ticker} in MT5.")  # Debug print
    return None  # Return None if no matching symbol is found


@app.route('/execute_lstm_trades', methods=['POST'])
def execute_lstm_trades():
    try:
        data = request.get_json()
        prediction_values = data.get('predictionValues', {})
        print(f"Received prediction values: {prediction_values}")
        total_funds = 100000  # Fixed total funds for trading

        # Ensure all prediction values are numeric and log them
        for ticker in list(prediction_values.keys()):  # Use list to avoid dictionary size change during iteration
            try:
                prediction_values[ticker] = float(prediction_values[ticker])
            except ValueError as e:
                print(f"Invalid prediction value for {ticker}: {prediction_values[ticker]}")
                return jsonify({'error': f'Invalid prediction value for {ticker}: {prediction_values[ticker]}'}), 400

        # Allocate funds based on LSTM prediction values (you'll need to implement this)
        allocations = allocate_funds(total_funds, prediction_values)
        trade_results = []  # To store the results of each trade

        # Ensure all values are numeric
        for ticker, allocation in allocations.items():
            # Use the find_mt5_symbol function to attempt to find the MT5 symbol
            mt5_symbol = find_mt5_symbol(ticker)
            if mt5_symbol is None:
                print(f"No matching MT5 symbol found for {ticker}. Skipping...")
                continue

            try:
                # Fetch the latest tick data for the ticker Using the matched MT5 symbol
                tick = mt.symbol_info_tick(mt5_symbol)
                if tick is None:
                    print(f"No tick data available for {mt5_symbol}. Skipping...")
                    continue

                # Fetch symbol-specific constraints
                symbol_info = mt.symbol_info(mt5_symbol)
                if symbol_info is None:
                    print(f"No symbol info available for {ticker}. Skipping...")
                    continue

                min_volume = symbol_info.min_volume
                max_volume = symbol_info.max_volume
                volume_step = symbol_info.volume_step

                # Calculate the units based on allocation and price, and adjust according to volume constraints
                units = allocation / tick.ask
                units = max(min_volume, min(units, max_volume))  # Ensure units are within min and max volume
                units = round(units / volume_step) * volume_step  # Adjust units to the nearest volume step

                # Execute trade
                result = market_order(mt5_symbol, units, 'buy', 5, 100, 0, 0)
                print(f"Executed trade for {mt5_symbol}: {result}")
                trade_results.append({mt5_symbol: {'result': result, 'units': units}})
            except Exception as e:
                print(f"Error executing trade for {ticker}: {e}")
        return jsonify({'message': 'Executed LSTM-based trades', 'tradeResults': trade_results})
    except Exception as e:
        # Make sure to return a JSON response even in case of error
        return jsonify({'error': str(e)}), 500


@app.route('/get_account_info')
def get_account_info():
    if not mt.initialize():
        return jsonify({'error': 'Failed to initialize MT5'}), 500

    account_info = mt.account_info()._asdict()
    positions = mt.positions_get()
    positions_data = [{'symbol': pos.symbol, 'profit': pos.profit, 'currency': '£'} for pos in
                      positions]  # Assuming USD for simplicity

    # mt.shutdown()

    return jsonify({
        'balance': account_info['balance'],
        'equity': account_info['equity'],
        'positions': positions_data
    })


def get_open_positions():
    if not mt.initialize():
        print("initialize() failed, error code =", mt.last_error())
        return []

    positions = mt.positions_get()
    if positions is None:
        print("No positions found, error code =", mt.last_error())
        return []

    positions_data = [{'symbol': pos.symbol, 'profit': pos.profit} for pos in positions]
    return positions_data


@app.route('/update_positions', methods=['GET'])
def update_positions():
    positions_data = get_open_positions()
    # You can pass positions_data to your HTML template or return it as JSON
    return jsonify(positions_data)


def calculate_get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)


@app.route('/get_stock_price', methods=['POST'])
def get_stock_price_route():
    data = request.json
    ticker = data['ticker']
    price = calculate_get_stock_price(ticker)
    return jsonify({'price': price})


def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])


@app.route('/calculate_SMA', methods=['POST'])
def SMA_route():
    data = request.json
    ticker = data['ticker']
    window = data['window']
    sma = calculate_SMA(ticker, window)
    return jsonify({'sma': sma})


def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])


@app.route('/calculate_EMA', methods=['POST'])
def EMA_route():
    data = request.json
    ticker = data['ticker']
    window = data['window']
    ema = calculate_EMA(ticker, window)
    return jsonify({'ema': ema})


def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])


@app.route('/calculate_RSI', methods=['POST'])
def RSI_route():
    data = request.json
    ticker = data['ticker']
    rsi = calculate_RSI(ticker)
    return jsonify({'rsi': rsi})


def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'


@app.route('/calculate_MACD', methods=['POST'])
def MACD_route():
    data = request.json
    ticker = data['ticker']
    macd = calculate_MACD(ticker)
    return jsonify({'macd': macd})


def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()


@app.route('/plot_stock_price', methods=['POST'])
def plot_stock_price_route():
    data = request.json
    ticker = data['ticker']
    plot_stock_price(ticker)
    # Assuming 'stock.png' is saved in a static directory
    return jsonify({'image_url': 'stock.png'})


functions = [
    {
        'name': 'get_stock_price',
        'description': 'Gets the latest stock price given the ticker symbol of a company.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company(for example AAPL for apple)'
                }
            },
            'required': ['ticker']
        }
    },
    {
        "name": "calculate_SMA",
        "description": "Calculate the simple moving average for a given stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": 'string',
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "window": {
                    "type": "integer",
                    "description": "The timeframe to consider when calculating the SMA"
                }
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "calculate_EMA",
        "description": "Calculate the exponential moving average for a given stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "window": {
                    "type": "integer",
                    "description": "The timeframe to consider when calculating the EMA"
                }
            },
            "required": ["ticker", "window"],
        }
    },
    {
        "name": "calculate_RSI",
        "description": "Calculate the RSI for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": 'string',
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_MACD",
        "description": "Calculate the MACD for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": 'string',
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_stock_price",
        "description": "Plot the stock price for the last year given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": 'string',
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
            },
            "required": ["ticker"],
        },
    },
]

available_functions = {
    'get_stock_price': calculate_get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price
}


@app.route('/process_user_input', methods=['POST'])
def process_user_input():
    data = request.json
    # debug
    print("Received data:", data)  # Log incoming data
    user_input = data.get('user_input')

    if 'messages' not in session:
        session['messages'] = []

    session['messages'].append({'role': 'user', 'content': user_input})

    try:
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo-0613',
                                                messages=session['messages'],
                                                functions=functions,
                                                function_call='auto')

        response_message = response.choices[0].message

        if hasattr(response_message, 'function_call'):
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)

            args_dict = {}
            for key, value in function_args.items():
                args_dict[key] = value

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**args_dict)

            session['messages'].append(
                {
                    'role': 'function',
                    'name': function_name,
                    'content': function_response
                }
            )

            second_response = openai.ChatCompletion.create(model='gpt-3.5-turbo-0613',
                                                           messages=session['messages'])

            final_response = second_response.choices[0].message.content
            return jsonify({'response': final_response})
        else:
            session['messages'].append({'role': 'assistant', 'content': response_message['content']})
            return jsonify({'response': response_message['content']})

    except Exception as e:
        return jsonify({'error': str(e)})


def open_browser():
    url = "http://localhost:9000/"
    if platform.system() == "Windows":
        os.system(f'start msedge.exe --app="{url}"')
    elif platform.system() == "Darwin":  # macOS
        os.system(f'open -a "Microsoft Edge" "{url}"')
    else:
        # For other platforms, like Linux, adjust as necessary
        webbrowser.open_new(url)


if __name__ == '__main__':
    # Open the browser
    open_browser()
    app.run(debug=True, port=9000)  # Changed Flask to run on por
