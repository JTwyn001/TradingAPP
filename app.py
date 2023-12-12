from flask import Flask, request, jsonify, session, render_template
import os
import platform
import openai
import webbrowser
import matplotlib.pyplot as plt
import yfinance as yf

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the Flask app and set the template folder
app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have 'index.html' in the templates folder


@app.route('/trading-ai')
def trading_ai():
    return render_template('trading_ai.html')


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


@app.route('/process_chat', methods=['POST'])
def process_chat():
    try:
        data = request.json
        user_input = data['user_input']

        # Logic to handle the chat
        # Replace the Streamlit session state with a server-side storage, e.g., a simple list
        if 'messages' not in session:
            session['messages'] = []

        session['messages'].append({'role': 'user', 'content': user_input})

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=session['messages'],
            functions=functions,
            function_call='auto'
        )

        # Process the response and return it
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


def open_browser():
    url = "http://localhost:5000/index.html"
    if platform.system() == "Windows":
        os.system(f'start msedge.exe --app="{url}"')
    elif platform.system() == "Darwin":  # macOS
        os.system(f'open -a "Microsoft Edge" "{url}"')
    else:
        webbrowser.open(url)


# Open the browser

if __name__ == '__main__':
    open_browser()
    app.run(debug=True, port=5000)  # Changed Flask to run on port 5000
