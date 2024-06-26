import datetime
import json

from flask import Flask, request, jsonify, session, render_template
# from flask_cors import CORS
import os
import platform
from openai import OpenAI
import numpy as np  # The Numpy numerical computing library
import pandas as pd  # The Pandas data science library
import pandas_ta as ta
import requests  # The requests library for HTTP requests in Python
import xlsxwriter  # The XlsxWriter library for
import math  # The Python math module
from scipy import stats  # The SciPy stats module

import webbrowser
import matplotlib.pyplot as plt
import yfinance as yf

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from tensorflow.keras.models import load_model
import joblib

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize the Flask app and set the template folder
app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)
# Set a secret key for session handling
app.secret_key = 'algotradingproject'  # Generates a random key

# trained model saved as 'lstm_model.h5'
lstm_model = load_model('./models/lstm_model.h5')

scaler = joblib.load('./scalers/lstm_scaler_20240303_054818.pkl')


@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have 'index.html' in the templates folder


@app.route('/trading-ai')
def trading_ai():
    sp500_stocks = pd.read_csv('sp_500_stocks.csv')
    top_10_stocks = sp500_stocks['Ticker'].head(10).tolist()  # Assuming the column name is 'Ticker'
    return render_template('trading_ai.html', top_10_stocks=top_10_stocks)


@app.route('/get_data/<symbol>')
def get_data(symbol):
    try:
        # Fetch historical data for the symbol from Yahoo Finance
        data = yf.download(symbol, period="1d", interval="1m")
        # Convert the data to a list of dictionaries for JSON response
        response_data = data.reset_index().to_dict('records')
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)})


# Load the list of stocks from a CSV file
stocks = pd.read_csv('sp_500_stocks.csv')


def fetch_data_yfinance(tickers):
    data = yf.download(tickers, period="1y")
    return data['Adj Close']  # Adjusted Close Prices over the last year


# Assuming 'tickers' is a list of ticker symbols you want to fetch data for
tickers = ['AAPL', 'MSFT', 'GOOG']  # Example tickers
data = fetch_data_yfinance(tickers)

# Calculate the one-year price return for each stock
one_year_returns = data.pct_change().tail(1)


# Function to split the stock list into chunks of 100
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Split stocks into groups of 100 and create a list of comma-separated strings
symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = [','.join(group) for group in symbol_groups]

# Define columns for the DataFrame
columns = ['Ticker', 'Price', 'One-Year Price Return']

# Create an empty DataFrame
momentum_dataframe = pd.DataFrame(columns=columns)

# Sort the DataFrame by 'One-Year Price Return' and select the top 10
momentum_dataframe.sort_values('One-Year Price Return', ascending=False, inplace=True)
top_10_momentum_stocks = momentum_dataframe.head(10)


# Function to get top 10 momentum stocks
def get_top_10_momentum_stocks():
    sp500_stocks = pd.read_csv('sp_500_stocks.csv')
    tickers = sp500_stocks['Ticker'].tolist()
    historical_data = yf.download(tickers, period="1y")['Adj Close']
    returns = historical_data.pct_change().mean() * 252  # Assuming 252 trading days in a year
    top_10_momentum = returns.nlargest(10)
    return top_10_momentum.index.tolist()


# Flask route to scan market and get top 10 momentum stocks
@app.route('/scan-market')
def scan_market():
    try:
        top_10_stocks = get_top_10_momentum_stocks()
        return jsonify(top_10_stocks)
    except Exception as e:
        return jsonify({'error': str(e)})


# Function to get historical stock data using Yahoo Finance
def get_historical_data(tickers):
    data = yf.download(tickers, period="1y")['Adj Close']
    return data


# Function to calculate momentum scores
def calculate_momentum_scores(data):
    returns = data.pct_change().dropna()
    momentum_scores = returns.mean() * 252 / returns.std()  # Example: Annualized Sharpe-like momentum score
    return momentum_scores


def get_last_price(ticker):
    # Fetch the most recent closing price for the ticker
    data = yf.download(tickers=ticker, period='1d')
    last_price = data['Close'].iloc[-1]
    return last_price


# Function to load the latest model and scaler
def load_latest_model_and_scaler(model_dir='./models', scaler_dir='./scalers'):
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.h5')]
    scaler_files = [os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir) if f.endswith('.pkl')]

    # Find the latest model file
    latest_model_file = max([os.path.join(model_dir, f) for f in os.listdir(model_dir)], key=os.path.getctime)
    # Find the latest scaler file
    latest_scaler_file = max([os.path.join(scaler_dir, f) for f in os.listdir(scaler_dir)], key=os.path.getctime)

    model = load_model(latest_model_file)
    scaler = joblib.load(latest_scaler_file)

    return model, scaler


# Function to fetch recent data
def fetch_recent_data(ticker, window_size=30):
    try:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        # Increase the buffer to account for weekends and holidays
        start_date = (datetime.datetime.now() - datetime.timedelta(days=window_size * 3)).strftime('%Y-%m-%d')

        data = yf.download(ticker, start=start_date, end=end_date)

        # Ensure data was fetched successfully
        if data.empty:
            raise ValueError(f"No data fetched for {ticker}")

        # Add indicators as per model's training
        data['RSI'] = ta.rsi(data['Close'], length=15)
        data['EMAF'] = ta.ema(data['Close'], length=20)
        data['EMAM'] = ta.ema(data['Close'], length=100)
        data['EMAS'] = ta.ema(data['Close'], length=150)

        data.dropna(inplace=True)

        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None  # Return None or handle this case as appropriate in your application


def preprocess_data_for_lstm(ticker):
    # Fetch historical data
    data = yf.download(tickers=ticker, start='2012-03-11', end='2024-02-10')

    # Adding indicators
    data['RSI'] = ta.rsi(data['Close'], length=15)
    data['EMAF'] = ta.ema(data['Close'], length=20)
    data['EMAM'] = ta.ema(data['Close'], length=100)
    data['EMAS'] = ta.ema(data['Close'], length=150)

    # Ensure all data is complete
    data.dropna(inplace=True)

    # Select relevant features for predictions
    features = data[['Adj Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values

    # Load the scaler
    scaler = joblib.load('./scalers/lstm_scaler_20240303_054818.pkl')

    # Scale the features
    features_scaled = scaler.transform(features)

    if features_scaled.shape[0] < 30:
        raise ValueError("Not enough data to create a sequence for LSTM.")

    features_reshaped = features_scaled[-30:].reshape(1, 30, features_scaled.shape[1])

    return features_reshaped


# Function for LSTM prediction
@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    # Extract the selected stocks from the request
    selected_stocks = request.json['selectedStocks']

    # Load the latest model and scaler dynamically
    model, scaler = load_latest_model_and_scaler()

    predictions = {}
    for ticker in selected_stocks:
        try:
            recent_data = fetch_recent_data(ticker, window_size=30)
            prepared_data = preprocess_data_for_lstm(ticker)
            if prepared_data is None or recent_data.empty:
                predictions[ticker] = 'No data available'
                continue

            predicted_change = lstm_model.predict(prepared_data)[0][0]
            recommendation = 'Buy' if predicted_change > 0 else 'Sell'
            predictions[ticker] = recommendation
        except Exception as e:
            predictions[ticker] = f'Error: {str(e)}'

    return jsonify(predictions)


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
        response = client.chat.completions.create(model='gpt-3.5-turbo-0613',
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

            second_response = client.chat.completions.create(model='gpt-3.5-turbo-0613',
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
    app.run(debug=True, port=9000)  # Changed Flask to run on port 8000
