import pandas as pd
import MetaTrader5 as mt
import time
import numpy as np
from datetime import datetime, timedelta




# Sending Market Order Crossover Strategy *1
def market_order(symbol, volume, order_type, deviation, magic, stoploss, takeprofit):
    tick = mt.symbol_info_tick(symbol)

    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}

    request = {
        "action": mt.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "deviation": deviation,
        "magic": magic,
        "sl": stoploss,
        "tp": takeprofit,
        "comment": "python market order",
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_IOC,
    }

    order_result = mt.order_send(request)
    print(order_result)

    return order_result


# Closing an order from the ticket ID
def close_order(ticket):
    positions = mt.positions_get()

    for pos in positions:
        tick = mt.symbol_info_tick(pos.symbol)
        type_dict = {0: 1, 1: 0}  # 0 for buy, 1 for sale. so inverting to close pos
        price_dict = {0: tick.ask, 1: tick.bid}

        if pos.ticket == ticket:
            request = {
                "action": mt.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "deviation": DEVIATION,
                "magic": 100,
                "sl": 2,
                "tp": 3,
                "comment": "python close order",
                "type_time": mt.ORDER_TIME_GTC,
                "type_filling": mt.ORDER_FILLING_IOC,
            }

        order_result = mt.order_send(request)
        print(order_result)
        return order_result

    return 'Ticket does not exist'


# function for symbol exposure
def get_exposure(symbol):
    positions = mt.positions_get(symbol=symbol)
    if positions:
        pos_df = pd.DataFrame(positions, columns=positions[0]._asdict().keys())
        exposure = pos_df['volume'].sum()

        return exposure


# --------------------------Start of signals--------------------------------------

# function looking for trading signals (Bollinger Bands)
def get_signal():
    # bar data
    bars = mt.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, SMA_PERIOD)
    # bars = mt.copy_rates_from_pos(symbol, timeframe, 1, sma_period)

    # Convert bars to DataFrame
    df = pd.DataFrame(bars)

    # Simple Moving Average
    sma = df['close'].mean()

    sd = df['close'].std()

    lower_band = sma - STANDARD_DEVIATIONS * sd

    upper_band = sma + STANDARD_DEVIATIONS * sd

    last_close_price = df.iloc[-1]['close']

    print(last_close_price, lower_band, upper_band)
    # finding signals
    bollsignal = 'flat'
    if last_close_price < lower_band:
        bollsignal = 'buy'
    elif last_close_price > upper_band:
        bollsignal = 'sell'

    return sd, bollsignal


# function looking for trading signals (Crossover)
def cross_signal(symbol, timeframe, sma_period):
    bars = mt.copy_rates_from_pos(symbol, timeframe, 1, sma_period)
    df = pd.DataFrame(bars)

    last_close = df.iloc[-1].close
    sma = df.close.mean()

    direction = 'flat'
    if last_close > sma:
        direction = 'buy'  # long
    elif last_close < sma:
        direction = 'sell'  # short

    return last_close, sma, direction


def find_crossover(symbol, timeframe, sma_periods):
    # Assuming sma_periods is a tuple or list with two elements: (fast_sma_period, slow_sma_period)
    fast_sma_period, slow_sma_period = sma_periods

    # Copy the last 'slow_sma_period' + 1 bars to calculate the fast and slow SMAs
    rates = mt.copy_rates_from_pos(symbol, timeframe, 0, slow_sma_period + 1)
    if rates is None or len(rates) < slow_sma_period + 1:
        return None, None  # Not enough data to calculate SMAs

    # Convert the rates to a DataFrame
    df = pd.DataFrame(rates)

    # Calculate the fast and slow SMAs
    df['fast_sma'] = df['close'].rolling(fast_sma_period).mean()
    df['slow_sma'] = df['close'].rolling(slow_sma_period).mean()

    # Check for crossover in the last two periods
    last_fast_sma = df.iloc[-1]['fast_sma']
    prev_fast_sma = df.iloc[-2]['fast_sma']
    last_slow_sma = df.iloc[-1]['slow_sma']
    prev_slow_sma = df.iloc[-2]['slow_sma']

    crossignal = 'flat'
    # Detect bullish and bearish crossovers
    if last_fast_sma > last_slow_sma and prev_fast_sma < prev_slow_sma:
        crossignal = 'buy'
    elif last_fast_sma < last_slow_sma and prev_fast_sma > prev_slow_sma:
        crossignal = 'sell'

    return crossignal, last_fast_sma


def calculate_rsi(symbol, timeframe, rsi_period=14):
    """
    Calculate the Relative Strength Index (RSI) for given data.

    :param symbol: BTCUSD symbol
    :param timeframe:
    :param rsi_period: Period for RSI calculation
    :return: DataFrame with an additional 'rsi' column
    """
    bars = mt.copy_rates_from_pos(symbol, timeframe, 0, rsi_period + 1)
    if bars is None or len(bars) < rsi_period + 1:
        return None

    df = pd.DataFrame(bars)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=rsi_period, min_periods=rsi_period).mean()

    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def rsi_signal(data, overbought_level=70, oversold_level=30):
    """
    Generate RSI based buy/sell signals.

    :param data: DataFrame with RSI values.
    :param overbought_level: RSI level to indicate overbought conditions.
    :param oversold_level: RSI level to indicate oversold conditions.
    :return: Signal ('buy', 'sell', or 'flat')
    """
    latest_rsi = data['rsi'].iloc[-1]

    if latest_rsi > overbought_level:
        return 'sell'
    elif latest_rsi < oversold_level:
        return 'buy'
    else:
        return 'flat'


# --------------------------End of signals--------------------------------------


if __name__ == '__main__':
    # strategy params
    SYMBOL = "BTCUSD"
    TIMEFRAME = mt.TIMEFRAME_D1  # TIMEFRAME_D1, TIMEFRAME
    VOLUME = 1.0  # FLOAT
    DEVIATION = 5  # INTEGER
    MAGIC = 10
    SMA_PERIOD = 10  # INTEGER
    OVERBOUGHT = 70
    OVERSOLD = 30
    STANDARD_DEVIATIONS = 1  # number of Deviations for calculation of Bollinger Bands
    TP_SD = 2  # number of deviations for take profit
    SL_SD = 3  # number of deviations for stop loss
    fsma_period = 5
    ssma_period = 30

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

    account_info = mt.account_info()
    print(account_info)

    symbol_info = mt.symbol_info("BTCUSD")._asdict()
    print(symbol_info)

    symbol_price = mt.symbol_info_tick("BTCUSD")._asdict()
    print(symbol_price)

    while True:
        # calculating account exposure
        #    if mt.positions_total() == 0:
        exposure = get_exposure(SYMBOL)
        tick = mt.symbol_info_tick(SYMBOL)
        sd, bollsignal = get_signal()
        df = calculate_rsi(SYMBOL, TIMEFRAME)  # Fetch and calculate RSI for SYMBOL
        if df is not None:  # checks that df is not null and the rsi value calculated is valid
            rsisignal = rsi_signal(df)
        # calculating last candle close and sma and checking trading signal
        last_close, sma, direction = cross_signal(SYMBOL, TIMEFRAME, SMA_PERIOD)
        crossignal, last_fast_sma = find_crossover(SYMBOL, TIMEFRAME, (fsma_period, ssma_period))

        # trading logic
        # if direction == 'buy' and bollsignal == 'buy' and rsisignal == 'buy':
        if rsisignal == 'buy':
            # if a BUY signal is detected, close all short orders
            for pos in mt.positions_get():
                if pos.type == 1:  # pos.type == 1 means a sell order
                    close_order(pos.ticket)
            # if there are no open positions, open a new long position
            if not mt.positions_total():
                market_order(SYMBOL, VOLUME, 'buy', DEVIATION, MAGIC, tick.bid - SL_SD * sd,
                             tick.bid + TP_SD * sd)

        # elif direction == 'sell' and bollsignal == 'sell' and rsisignal == 'sell':
        elif rsisignal == 'sell':
            # if a SELL signal is detected, close all short orders
            for pos in mt.positions_get():
                if pos.type == 0:  # pos.type == 0 means a buy order
                    close_order(pos.ticket)
            if not mt.positions_total():
                market_order(SYMBOL, VOLUME, 'sell', DEVIATION, MAGIC, tick.bid + SL_SD * sd,
                             tick.bid - TP_SD * sd)

        print('time: ', datetime.now())
        print('exposure: ', exposure)
        print('last_close: ', last_close)
        print('sma: ', sma)
        print('crossover signal: ', direction)
        print('Bollinger Signal: ', bollsignal)
        print('SMA signal: ', crossignal)
        print('RSI signal: ', rsisignal)
        print('-------\n')

        # update ever 1s
        time.sleep(1)

num_orders = mt.orders_total()
num_orders

orders = mt.orders_get()
orders

num_order_history = mt.history_orders_total(datetime(2023, 10, 1), datetime(2021, 10, 13))
num_order_history

order_history = mt.history_orders_get(datetime(2023, 10, 1), datetime(2023, 10, 13))
order_history

num_deal_history = mt.history_deals_total(datetime(2023, 10, 1), datetime(2023, 10, 13))
num_deal_history

deal_history = mt.history_deals_get(datetime(2023, 10, 1), datetime(2023, 10, 13))
deal_history
