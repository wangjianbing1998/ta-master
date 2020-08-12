# coding=gbk
import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)
import numpy as np
import os
import tushare as ts
from tqdm import tqdm

ts.set_token(token='b405a182568b48e1efdd49b944c592286125a7e72fdee77ea47a071b')
pro = ts.pro_api()


def stock_price_intraday(ticker, folder):
    os.makedirs(folder, exist_ok=True)
    symbol = ticker + ('.SH' if ticker.startswith('6') else '.SZ')
    intraday_bar = ts.pro_bar(symbol, adj='qfq', adjfactor=True)
    intraday_daily = pro.daily_basic(ts_code=symbol)
    del intraday_daily['close']

    intraday = pd.merge(intraday_bar, intraday_daily, on=['ts_code', 'trade_date'])

    path = os.path.join(folder, f'{ticker}.csv')
    # if os.path.exists(path):
    #     history = pd.read_csv(path, index_col=0)
    #     intraday.append(history)
    intraday.reset_index(drop=True, inplace=True)
    intraday.sort_values(by='trade_date', inplace=True)
    intraday.to_csv(path, index=False)


members = pd.read_excel('CSI800.xlsx')
res = 'res/t0625-get_CSI800_unique_history/'

unique_tickers = []
for col in members.columns.to_list():
    ticker_list = members[col].values.tolist()
    ticker_list = [c.split()[0] for c in ticker_list if c is not np.nan]
    unique_tickers.extend(ticker_list)

unique_tickers = list(set(unique_tickers))
unique_tickers.sort()
bar = tqdm(unique_tickers)
for ticker in bar:
    try:
        bar.set_description(f'Get ticker {ticker}')
        stock_price_intraday(ticker, 'res/t0625-get_CSI800_unique_daily_history/')
        # break
    except Exception as e:
        print(e)
