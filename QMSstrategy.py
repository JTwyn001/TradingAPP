import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from scipy import stats

def fetch_historical_data_yf(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if data.empty:
        print(f"No data for {ticker}")
        return None

    data.reset_index(inplace=True)
    data.set_index('Date', inplace=True)
    return data

def get_top_10_momentum_stocks():
    sp500_stocks = pd.read_csv('sp_500_stocks.csv')
    tickers = sp500_stocks['Ticker'].tolist()  # Limiting for quick testing

    end_date = pd.Timestamp.now()
    periods = {
        'One-Year': end_date - BDay(252),
        'Six-Month': end_date - BDay(126),
        'Three-Month': end_date - BDay(63),
        'One-Month': end_date - BDay(21),
    }

    hqm_dataframe = pd.DataFrame(index=tickers)

    for ticker in tickers:  # Limiting the number of tickers for quicker testing
        print(f"Fetching data for {ticker}...")
        for period_name, start_date in periods.items():
            data = fetch_historical_data_yf(ticker, start_date, end_date)
            if data is not None:
                price_return = data['Close'].pct_change().dropna().mean() * 252  # Annualized return
                hqm_dataframe.loc[ticker, period_name + ' Return'] = price_return

    for period_name in periods.keys():
        hqm_dataframe[period_name + ' Return Percentile'] = hqm_dataframe[period_name + ' Return'].rank(pct=True)

    hqm_dataframe['HQM Score'] = hqm_dataframe[[period + ' Return Percentile' for period in periods.keys()]].mean(
        axis=1)

    top_10_hqm_stocks = hqm_dataframe.sort_values(by='HQM Score', ascending=False).head(10)

    # Print top 10 stocks with their returns, percentiles, and HQM Scores for clarity
    print(top_10_hqm_stocks[[period + ' Return' for period in periods.keys()] +
                            [period + ' Return Percentile' for period in periods.keys()] +
                            ['HQM Score']])

    return top_10_hqm_stocks.index.tolist()
# top_10_momentum_stocks = get_top_10_momentum_stocks()
# print("Top 10 Momentum Stocks:", top_10_momentum_stocks)