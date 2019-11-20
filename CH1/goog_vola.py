import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

import matplotlib.pyplot as plt

# goog = web.DataReader("GOOG", data_source='google', start='3/14/2019' ,end='4/14/2014')
# goog.tail()

goog = yf.download(  # or pdr.get_data_yahoo(...
    # tickers list or string as well
    tickers="GOOG",

    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    period="10y",

    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')
    interval="1d",

    # group by ticker (to access via data['SPY'])
    # (optional, default is 'column')
    group_by='ticker',

    # adjust all OHLC automatically
    # (optional, default is False)
    auto_adjust=True,

    # download pre/post regular market hours data
    # (optional, default is False)
    prepost=True,

    # use threads for mass downloading? (True/False/Integer)
    # (optional, default is True)
    threads=True,

    # proxy URL scheme use use when downloading?
    # (optional, default is None)
    proxy=None
)

goog['Log_Ret'] = np.log(goog['Close'] / goog['Close'].shift(1))
goog['Volatility'] = goog['Log_Ret'].rolling(252).std()

goog[['Close', 'Volatility']].plot(subplots=True)
plt.show()