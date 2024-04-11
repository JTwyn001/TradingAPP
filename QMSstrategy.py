import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from scipy import stats

def fetch_historical_data(ticker, start_date, end_date):
    # Append '=X' to the ticker symbol for forex data
    forex_ticker = ticker + '=X'
    data = yf.download(forex_ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if data.empty:
        print(f"No data for {forex_ticker}")
        return None
    return data

def get_top_10_momentum_forex():
    forex_tickers = pd.read_csv('mt5_forex_tickers.csv')
    tickers = forex_tickers['Ticker'].tolist()[:62]  # Adjust as needed

    end_date = pd.Timestamp.now()
    periods = {
        'One-Year': end_date - BDay(252),
        'Six-Month': end_date - BDay(126),
        'Three-Month': end_date - BDay(63),
        'One-Month': end_date - BDay(21),
    }

    hqm_dataframe = pd.DataFrame(index=tickers)

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        for period_name, start_date in periods.items():
            data = fetch_historical_data(ticker, start_date, end_date)
            if data is not None:
                price_return = data['Close'].pct_change().dropna().mean() * 252
                hqm_dataframe.loc[ticker, period_name + ' Return'] = price_return

    for period_name in periods.keys():
        hqm_dataframe[period_name + ' Return Percentile'] = hqm_dataframe[period_name + ' Return'].rank(pct=True)

    hqm_dataframe['HQM Score'] = hqm_dataframe[[period + ' Return Percentile' for period in periods.keys()]].mean(axis=1)

    top_10_hqm_forex = hqm_dataframe.sort_values(by='HQM Score', ascending=False).head(10)

    # Explicitly print all data for the top 10 momentum forex
    print(top_10_hqm_forex.to_string())

    return top_10_hqm_forex.index.tolist()

# Running the function and printing the top 10 momentum forex
top_10_momentum_forex = get_top_10_momentum_forex()
print("Top 10 Momentum Forex:", top_10_momentum_forex)
