# -*- coding: utf-8 -*-

import mpl_finance as mpf
import numpy as np
import pandas as pd

SHOW_DATE = 60
import matplotlib.pyplot as plt


def dropna(df):
    """Drop rows with "Nans" values
    """
    # df = df[df < math.exp(709)]  # big number
    # df = df[df != 0.0]
    df = df.dropna()
    return df


def _fillna(series, value=0):
    return series.replace([np.inf, -np.inf], np.nan).fillna(value)


def highest(series, perid=10, fillna=False, name='_highest'):
    res = pd.Series(data=np.nan, index=series.index, name=name)

    for i in range(len(series)):
        res.iloc[i] = series.iloc[max(i - perid, 0):i].max()

    if fillna:
        res = _fillna(res)
    return res


def lowest(series, perid=10, fillna=False, name='_lowest'):
    res = pd.Series(data=np.nan, index=series.index, name=name)

    for i in range(len(series)):
        res.iloc[i] = series.iloc[max(i - perid, 0):i].min()

    if fillna:
        res = _fillna(res)
    return res


def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()


def get_min_max(x1, x2, f='min'):
    if not np.isnan(x1) and not np.isnan(x2):
        if f == 'max':
            return max(x1, x2)
        elif f == 'min':
            return min(x1, x2)
        else:
            raise ValueError('"f" variable value should be "min" or "max"')
    else:
        return np.nan


def plot_k(stock, show_date=SHOW_DATE):
    # plt.xticks(rotation=45)
    plt.xlabel('time')
    plt.ylabel('price')
    mpf.candlestick_ohlc(plt.gca(), stock[['Date', 'Open', 'High', 'Low', 'Close']].iloc[:show_date].values, width=0.7,
                         colorup='r', colordown='green')  # 上涨为红色K线，下跌为绿色，K线宽度为0.7
    plt.grid(True)
    plt.gca().xaxis_date()


def plot_(data, show_date=SHOW_DATE):
    show_date = min(show_date, len(data))
    date_ = data.iloc[:show_date]
    date_.plot(ax=plt.gca())


def get_result(func, *args, show=True, show_date=20, **kwargs):
    from ..data.get_data import get_yahoo_data
    data = get_yahoo_data()
    args = [data[t] for t in args]
    res = func(*args, **kwargs)
    if show:
        import matplotlib.pyplot as plt
        if show_date is not None:
            res[:show_date].plot()
        else:
            res[:].plot()
        plt.ylabel('price')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    return res


def after_return(*bargs, target=None, **bkwargs):
    '''
    do target after called, but before return.
    @after_return(lambda x,y:print(x,y),'after')
    def t2(a,):
        print(f'a={a}')
        return 2
    Input t2(5)
    OUTPUT: a=5
            2,after
    :param target:
    :param bargs:
    :param bkwargs:
    :return:
    '''

    def warpFunc(func):
        def _warp(*args, **kwargs):
            res = func(*args, **kwargs)

            def fillna_target(value, name=None):
                if kwargs['fillna']:
                    value = _fillna(value)
                name = name or func.__name__
                return pd.Series(value, index=args[0].index, name=name)

            t = target or fillna_target

            return t(res, *bargs, **bkwargs)

        return _warp

    return warpFunc


def eval_function_by_name(fn_name_args):
    return eval(fn_name_args)


def show_funcion(file_path):
    lines = []
    for line in open(file_path):
        if line.strip().startswith('def'):
            line = line.strip()
            lines.append(line)

    lines.sort()
    for line in lines:
        print(line)


if __name__ == '__main__':
    # df = get_yahoo_data()
    # plot_k(df)
    # plt.show()
    show_funcion('all.py')
    # pass
