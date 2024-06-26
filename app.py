import datetime
import json
import logging

from flask import Flask, request, jsonify, session, render_template
# from flask_cors import CORS
import os
import platform
import openai
import MetaTrader5 as mt
from datetime import timedelta
import datetime
from mt5 import (market_order, get_top_10_momentum_stocks, get_top_10_momentum_forex,
                 close_order, get_exposure)
import time
import numpy as np  # The Numpy numerical computing library
import pandas as pd  # The Pandas data science library
import pandas_ta as ta
from flask import jsonify, abort
from flask import Flask
from flask_cors import CORS
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
CORS(app)
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


def get_current_price(ticker):
    price = mt.symbol_info_tick(ticker).ask  # or use .bid, depending on your buying strategy
    return price


@app.route('/get_current_price/<symbol>')
def get_current_price_route(symbol):
    price = get_current_price(symbol)  # Your existing function to fetch the price
    if price:
        return jsonify({'current_price': price})
    else:
        return jsonify({'error': 'Price not found'}), 404


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


def initialize_mt():
    if not mt.initialize():
        print("Initialize() failed, error code =", mt.last_error())
        return False
    return True


def get_last_day_metrics(ticker_symbol):
    if not mt.initialize():
        print("Failed to initialize MT5 connection")
        return None, None
    symbol_info = mt.symbol_info_tick(ticker_symbol)
    if symbol_info is None:
        print(f"No symbol info found for {ticker_symbol}")
        return None, None

    rates = mt.copy_rates_from_pos(ticker_symbol, mt.TIMEFRAME_D1, 0, 2)
    if rates is None or len(rates) < 2:
        print(f"Not enough data to fetch rates for {ticker_symbol}")
        return None, None

    last_close = rates[-1]['close']
    last_prev_close = rates[-2]['close']
    return last_close, last_prev_close


@app.route('/execute_ml_predictions', methods=['GET'])
def execute_ml_predictions():
    tickers = ['EURUSD', 'SEKJPY', 'EURNOK', 'MU.NAS', 'NRG.NYSE']
    combined_predictions = {}

    for ticker_symbol in tickers:
        price = get_current_price(ticker_symbol)  # Make sure this function returns the current price
        combined_predictions[ticker_symbol] = {
            'lstm': None,
            'gbm': None,
            'avg': None,
            'last_close': None,
            'pct_change': None,
            'current_price': price  # Add current price to the data sent to the frontend
        }

        try:
            # Load models and scalers
            lstm_model, lstm_feature_scaler, lstm_target_scaler = load_latest_model_and_scaler_for_ticker(ticker_symbol)
            gbm_model = load_latest_gbm_model_for_ticker(ticker_symbol)

            # Fetch last day metrics
            last_close, last_prev_close = get_last_day_metrics(
                ticker_symbol)  # Assuming this function exists and returns last close and the previous close

            # Preprocess and predict using LSTM
            if lstm_model and lstm_feature_scaler and lstm_target_scaler:
                lstm_prepared_data = preprocess_data_for_lstm_mt(ticker_symbol, lstm_feature_scaler)
                if lstm_prepared_data is not None:
                    lstm_predicted_change_scaled = lstm_model.predict(lstm_prepared_data)
                    lstm_predicted_change = \
                        lstm_target_scaler.inverse_transform(lstm_predicted_change_scaled.reshape(-1, 1))[0][0]
                    combined_predictions[ticker_symbol]['lstm'] = float(lstm_predicted_change)

            # Preprocess and predict using GBM
            if gbm_model:
                gbm_prepared_data = preprocess_data_for_gbm(ticker_symbol, None)  # No scaler used
                if gbm_prepared_data is not None:
                    gbm_predicted_change = gbm_model.predict(gbm_prepared_data)
                    combined_predictions[ticker_symbol]['gbm'] = float(gbm_predicted_change[0])

            # Calculate average prediction
            if combined_predictions[ticker_symbol]['lstm'] is not None and combined_predictions[ticker_symbol][
                'gbm'] is not None:
                combined_predictions[ticker_symbol]['avg'] = (combined_predictions[ticker_symbol]['lstm'] +
                                                              combined_predictions[ticker_symbol]['gbm']) / 2

            # Store last close and percentage change
            combined_predictions[ticker_symbol]['last_close'] = last_close
            if last_close and last_prev_close:
                combined_predictions[ticker_symbol]['pct_change'] = ((
                                                                             last_close - last_prev_close) / last_prev_close) * 100

        except Exception as e:
            logging.error(f"Error processing {ticker_symbol}: {str(e)}")
            abort(500, description=f"Error processing {ticker_symbol}")

    return jsonify(combined_predictions)


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


def prepare_features(features, feature_scaler):
    if feature_scaler:
        # Scale the features using the provided scaler and return
        features_scaled = feature_scaler.transform(features)
        return features_scaled
    else:
        # Return features as numpy array if no scaler is used
        return features.values


def preprocess_data_for_gbm(ticker, feature_scaler):
    if not mt.initialize():
        print("MT5 initialize() failed, error code =", mt.last_error())
        return None

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)  # Last year
    utc_from = int(start_date.timestamp())
    utc_to = int(end_date.timestamp())

    rates = mt.copy_rates_range(ticker, mt.TIMEFRAME_D1, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        print(f"No recent data fetched for {ticker}. Error: {mt.last_error()}")
        return None

    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)

    # Forward fill to handle missing data, since backward fill might introduce look-ahead bias
    data.fillna(method='ffill', inplace=True)

    # Ensure all necessary columns have no NaN values after filling
    if data[['close', 'high', 'low']].isnull().any().any():
        print("Missing data persists in necessary columns after filling.")
        return None

    # Compute the required indicators based on the new setup
    compute_indicators(data)

    features = data[['close', 'RSI', 'EMA20', 'MACD', 'SMA50', 'ATR']].tail(
        1)  # Ensure this matches the training feature set
    if features.isnull().any().any():
        print("Features contain NaN values after computation.")
        return None

    return prepare_features(features, feature_scaler)


def compute_indicators(data):
    data['RSI'] = ta.rsi(data['close'], length=15)
    data['EMA20'] = ta.ema(data['close'], length=20)
    data['MACD'] = ta.macd(data['close'])['MACD_12_26_9']
    data['SMA50'] = data['close'].rolling(window=50).mean()
    if 'volume' in data.columns:
        data['OBV'] = ta.obv(data['close'], data['volume']).astype(float)
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)


def load_latest_model_and_scaler_for_ticker(ticker_symbol, model_dir='./models', scaler_dir='./scalers'):
    # Adjusted to look for '.keras' files instead of '.h5'
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if
                   f.endswith('.keras') and ticker_symbol in f]
    feature_scaler_files = [os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir) if
                            'feature_scaler' in f and f.endswith('.pkl') and ticker_symbol in f]
    target_scaler_files = [os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir) if
                           'target_scaler' in f and f.endswith('.pkl') and ticker_symbol in f]

    # Ensure there are files before using max to avoid ValueError
    latest_model_file = max(model_files, key=os.path.getctime) if model_files else None
    latest_feature_scaler_file = max(feature_scaler_files, key=os.path.getctime) if feature_scaler_files else None
    latest_target_scaler_file = max(target_scaler_files, key=os.path.getctime) if target_scaler_files else None

    # Load the latest files if they exist
    model = tf.keras.models.load_model(latest_model_file) if latest_model_file else None
    feature_scaler = joblib.load(latest_feature_scaler_file) if latest_feature_scaler_file else None
    target_scaler = joblib.load(latest_target_scaler_file) if latest_target_scaler_file else None

    return model, feature_scaler, target_scaler


def load_latest_gbm_model_for_ticker(ticker_symbol, model_dir='./gbm_models'):
    # Look for joblib files specific to the GBM models
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if
                   f.endswith('.joblib') and ticker_symbol in f]

    # Ensure there are files before using max to avoid ValueError
    latest_model_file = max(model_files, key=os.path.getctime) if model_files else None

    # Load the latest file if it exists
    model = joblib.load(latest_model_file) if latest_model_file else None

    # Return the model (No feature_scaler is used here, based on your GBM setup)
    return model


def load_specific_model_and_scalers(model_dir='./models', scaler_dir='./scalers'):
    model_filename = 'lstm_model_SPY_20240425_011025.keras'
    feature_scaler_filename = 'feature_scaler_SPY_20240425_011025.pkl'
    target_scaler_filename = 'target_scaler_SPY_20240425_011025.pkl'

    model_path = os.path.join(model_dir, model_filename)
    feature_scaler_path = os.path.join(scaler_dir, feature_scaler_filename)
    target_scaler_path = os.path.join(scaler_dir, target_scaler_filename)

    if not os.path.exists(model_path) or not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
        print("Specific model and scaler files not found.")
        return None, None, None

    model = load_model(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

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


def preprocess_data_for_gbm_prediction(ticker):
    # Download historical data for the ticker
    data = yf.download(ticker, start="2010-01-01", end="2024-04-01")

    # Compute RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # Exponential Moving Average
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Moving Average Convergence Divergence
    macd = ta.macd(data['Close'])  # This returns a DataFrame
    data['MACD'] = macd['MACD_12_26_9']

    # Simple Moving Average
    data['SMA50'] = data['Close'].rolling(window=50).mean()

    # Average True Range
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)

    # Ensure all columns needed are calculated before dropping NA values
    data.dropna(inplace=True)

    # Prepare the feature array; ensure the order and columns match the model's training data
    features = data[['Close', 'RSI', 'EMA20', 'MACD', 'SMA50', 'ATR']].values[-1]  # Get the most recent data point as input

    return features.reshape(1, -1)  # Reshape for a single prediction

def load_and_predict_with_gbm(ticker, model_filename):
    # Load the pre-trained GBM model
    try:
        model = joblib.load(model_filename)
    except FileNotFoundError:
        print("Model file not found.")
        return None

    # Preprocess the data for the given ticker
    features = preprocess_data_for_gbm_prediction(ticker)

    if features is not None:
        # Make the prediction
        predicted_change = model.predict(features)
        return predicted_change[0]  # Return the predicted value
    else:
        print(f"No data available to make a prediction for {ticker}.")
        return None


# Global dictionary to store prediction values
prediction_values = {}


@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    data = request.get_json()
    selected_stocks = data['selectedStocks']
    model_type = data['modelType']  # 'LSTM' or 'GBM'

    # Adjust the path for GBM model
    if model_type == 'GBM':
        model_path = 'C:\\Users\\tohye\\Documents\\Capstone\\TradingAPP\\gbm_models\\gbm_model_SPY_20240425_023925.joblib'
        model = joblib.load(model_path)
        preprocess = preprocess_data_for_gbm_prediction  # Assume this function is defined elsewhere
    else:
        model, feature_scaler, target_scaler = load_specific_model_and_scalers()  # Your LSTM setup
        preprocess = lambda ticker: preprocess_data_for_lstm(ticker, feature_scaler)  # Adjust as needed for your LSTM

    predictions = {}
    for ticker in selected_stocks:
        try:
            prepared_data = preprocess(ticker)
            if prepared_data is None:
                print(f"Prepared data for {ticker} is None.")
                continue

            if model_type == 'GBM':
                predicted_change = model.predict(prepared_data)[0]
            else:
                predicted_change_scaled = model.predict(prepared_data)
                predicted_change = target_scaler.inverse_transform(predicted_change_scaled.reshape(-1, 1))[0][0]

            predictions[ticker] = float(predicted_change)  # Convert numpy float32 to regular float for JSON serialization
        except Exception as e:
            print(f"Exception processing {ticker}: {e}")
            predictions[ticker] = None

    return jsonify(predictions)


def check_existing_positions(symbol):
    """Checks if there are open positions for a given symbol."""
    positions = mt.positions_get(symbol=symbol)
    return len(positions) > 0

@app.route('/trade_stocks', methods=['POST'])
def trade_stocks():
    data = request.get_json()
    print("Raw data received:", data)
    if not data:
        logging.error("No data received for trading stocks.")
        print("No data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    logging.info(f"Raw data received for trading: {data}")
    print("Received trade data:", data)  # This will log the received data
    results = {}
    for trade in data:
        symbol = trade.get('symbol')
        volume = float(trade.get('volume', 0))
        direction = trade.get('direction')  # 'buy' or 'sell'
        # Assume defaults for deviation, magic, sl, and tp if not provided in trade data
        deviation = trade.get('deviation', 10)
        magic = trade.get('magic', 123456)
        stoploss = trade.get('sl', 0)  # Set appropriate default or handle it in the market_order function
        takeprofit = trade.get('tp', 0)
        # Check if there are any open positions for this symbol
        if check_existing_positions(symbol):
            results[symbol] = {'status': 'error', 'message': 'Open positions exist, skipping trade'}
            logging.info(f"Skipping trade for {symbol} as open positions exist.")
            continue  # Skip to the next trade if open positions exist

        # Execute the market order
        result = market_order(symbol, volume, direction, deviation, magic, stoploss, takeprofit)
        if result.get('status') == 'error':
            logging.error(f"Trade failed for {symbol}: {result['message']}")
        results[symbol] = result

    return jsonify(results)


def market_order(symbol, volume, order_type, deviation, magic, stoploss, takeprofit):
    if not mt.initialize():
        return {'status': 'error', 'message': 'Failed to initialize MetaTrader'}

    if not mt.symbol_select(symbol, True):
        return {'status': 'error', 'message': 'Symbol not found or market closed'}

    tick = mt.symbol_info_tick(symbol)
    if tick is None:
        return {'status': 'error', 'message': 'No tick data available'}

    price = tick.ask if order_type == 'buy' else tick.bid
    request = {
        "action": mt.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt.ORDER_TYPE_BUY if order_type == 'buy' else mt.ORDER_TYPE_SELL,
        "price": price,
        "deviation": deviation,
        "magic": magic,
        "comment": "Executed via Flask",
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_IOC,
    }
    if stoploss > 0:
        request['sl'] = stoploss
    if takeprofit > 0:
        request['tp'] = takeprofit
    result = mt.order_send(request)
    if result is None or result.retcode != mt.TRADE_RETCODE_DONE:
        last_error = mt.last_error()
        error_message = f"Trade failed with error code {last_error[0]}: {last_error[1]}" if result is None else f"Trade failed with retcode: {result.retcode}"
        return {'status': 'error', 'message': error_message}

    # If result is a namedtuple and contains the '_asdict' method, convert it to a dict
    return result._asdict() if hasattr(result, '_asdict') else result


def close_order(ticket):
    """Closes a trade based on its ticket ID."""
    position = mt.position_get(ticket=ticket)
    if not position:
        logging.error(f"No position found with ticket {ticket}")
        return {'status': 'error', 'message': 'Position not found'}

    # Determine the type of order needed to close the position
    if position.type == mt.ORDER_TYPE_BUY:
        order_type = mt.ORDER_TYPE_SELL
        price = mt.symbol_info_tick(position.symbol).bid
    else:
        order_type = mt.ORDER_TYPE_BUY
        price = mt.symbol_info_tick(position.symbol).ask

    close_request = {
        "action": mt.TRADE_ACTION_DEAL,
        "position": ticket,
        "type": order_type,
        "volume": position.volume,
        "price": price,
        "deviation": 10,  # Set a suitable deviation value
        "comment": "Closing position"
    }

    result = mt.order_send(close_request)
    if result is None or result.retcode != mt.TRADE_RETCODE_DONE:
        error_message = f"Failed to close position {ticket}: {mt.last_error()}"
        logging.error(error_message)
        return {'status': 'error', 'message': error_message}

    return result._asdict() if hasattr(result, '_asdict') else result


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
