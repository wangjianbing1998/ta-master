# -*- coding: utf-8 -*-
"""
.. module:: momentum
   :synopsis: Momentum Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)


def rsi(close, n=14, fillna=False):
def money_flow_index(high, low, close, volume, n=14, fillna=False):
def tsi(close, r=25, s=13, fillna=False):
def uo(high, low, close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
def stoch(high, low, close, n=14, fillna=False):
def stoch_signal(high, low, close, n=14, d_n=3, fillna=False):
def wr(high, low, close, lbp=14, fillna=False):
def ao(high, low, s=5, len=34, fillna=False):
def kama(close, n=10, pow1=2, pow2=30, fillna=False):

"""

from .utils import *


def rsi(close, n=14, fillna=False):
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    diff = close.diff(1)
    which_dn = diff < 0

    up, dn = diff, diff * 0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, n, fillna)
    emadn = ema(dn, n, fillna)

    rsi = 100 * emaup / (emaup + emadn)
    if fillna:
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rsi, name='rsi')


def money_flow_index(high, low, close, volume, n=14, fillna=False):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """

    # 0 Prepare dataframe to work
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0

    # 2 up or down column
    df['Up_or_Down'] = 0
    df.loc[(tp > tp.shift(1)), 'Up_or_Down'] = 1
    df.loc[(tp < tp.shift(1)), 'Up_or_Down'] = -1

    # 3 money flow
    mf = tp * df['Volume'] * df['Up_or_Down']

    # 4 positive and negative money flow with n periods
    n_positive_mf = mf.rolling(n).apply(
        lambda x: np.sum(np.where(x >= 0.0, x, 0.0)),
        raw=True)
    n_negative_mf = abs(mf.rolling(n).apply(
        lambda x: np.sum(np.where(x < 0.0, x, 0.0)),
        raw=True))

    # 5 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))

    if fillna:
        mr = mr.replace([np.inf, -np.inf], np.nan).fillna(50)

    return pd.Series(mr, name='mfi_' + str(n))


def tsi(close, r=25, s=13, fillna=False):
    """True strength index (TSI)

    Shows both trend direction and overbought/oversold conditions.

    https://en.wikipedia.org/wiki/True_strength_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        r(int): high period.
        s(int): low period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    m = close - close.shift(1, fill_value=close.mean())
    m1 = m.ewm(r).mean().ewm(s).mean()
    m2 = abs(m).ewm(r).mean().ewm(s).mean()
    tsi = m1 / m2
    tsi *= 100
    if fillna:
        tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(tsi, name='tsi')


def uo(high, low, close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
       fillna=False):
    """Ultimate Oscillator

    Larry Williams' (1976) signal, a momentum oscillator designed to capture
    momentum across three different timeframes.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator

    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)

    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        s(int): short period
        m(int): medium period
        len(int): long period
        ws(float): weight of short BP average for UO
        wm(float): weight of medium BP average for UO
        wl(float): weight of long BP average for UO
        fillna(bool): if True, fill nan values with 50.

    Returns:
        pandas.Series: New feature generated.

    """
    min_l_or_pc = close.shift(1, fill_value=close.mean()).combine(low, min)
    max_h_or_pc = close.shift(1, fill_value=close.mean()).combine(high, max)

    bp = close - min_l_or_pc
    tr = max_h_or_pc - min_l_or_pc

    avg_s = bp.rolling(s, min_periods=0).sum() / tr.rolling(s, min_periods=0).sum()
    avg_m = bp.rolling(m, min_periods=0).sum() / tr.rolling(m, min_periods=0).sum()
    avg_l = bp.rolling(len, min_periods=0).sum() / tr.rolling(len, min_periods=0).sum()

    uo = 100.0 * ((ws * avg_s) + (wm * avg_m) + (wl * avg_l)) / (ws + wm + wl)
    if fillna:
        uo = uo.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(uo, name='uo')


def stoch(high, low, close, n=14, fillna=False):
    """Stochastic Oscillator

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    smin = low.rolling(n, min_periods=0).min()
    smax = high.rolling(n, min_periods=0).max()
    stoch_k = 100 * (close - smin) / (smax - smin)

    if fillna:
        stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(stoch_k, name='stoch_k')


def stoch_signal(high, low, close, n=14, d_n=3, fillna=False):
    """Stochastic Oscillator Signal

    Shows SMA of Stochastic Oscillator. Typically a 3 day SMA.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        d_n(int): sma period over stoch_k
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    stoch_k = stoch(high, low, close, n, fillna=fillna)
    stoch_d = stoch_k.rolling(d_n, min_periods=0).mean()

    if fillna:
        stoch_d = stoch_d.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(stoch_d, name='stoch_d')


def wr(high, low, close, lbp=14, fillna=False):
    """Williams %R

    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r

    Developed by Larry Williams, Williams %R is a momentum indicator that is
    the inverse of the Fast Stochastic Oscillator. Also referred to as %R,
    Williams %R reflects the level of the close relative to the highest high
    for the look-back period. In contrast, the Stochastic Oscillator reflects
    the level of the close relative to the lowest low. %R corrects for the
    inversion by multiplying the raw value by -100. As a result, the Fast
    Stochastic Oscillator and Williams %R produce the exact same lines, only
    the scaling is different. Williams %R oscillates from 0 to -100.

    Readings from 0 to -20 are considered overbought. Readings from -80 to -100
    are considered oversold.

    Unsurprisingly, signals derived from the Stochastic Oscillator are also
    applicable to Williams %R.

    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.

    From: https://www.investopedia.com/terms/w/williamsr.asp
    The Williams %R oscillates from 0 to -100. When the indicator produces
    readings from 0 to -20, this indicates overbought market conditions. When
    readings are -80 to -100, it indicates oversold market conditions.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """

    hh = high.rolling(lbp, min_periods=0).max()  # highest high over lookback period lbp
    ll = low.rolling(lbp, min_periods=0).min()  # lowest low over lookback period lbp

    wr = -100 * (hh - close) / (hh - ll)

    if fillna:
        wr = wr.replace([np.inf, -np.inf], np.nan).fillna(-50)
    return pd.Series(wr, name='wr')


def ao(high, low, s=5, len=34, fillna=False):
    """Awesome Oscillator

    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)

    The Awesome Oscillator is an indicator used to measure market momentum. AO
    calculates the difference of a 34 Period and 5 Period Simple Moving
    Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used
    to affirm trends or to anticipate possible reversals.

    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

    Awesome Oscillator is a 34-period simple moving average, plotted through
    the central points of the bars (H+L)/2, and subtracted from the 5-period
    simple moving average, graphed across the central points of the bars
    (H+L)/2.

    MEDIAN PRICE = (HIGH+LOW)/2

    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where

    SMA — Simple Moving Average.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        s(int): short period
        len(int): long period
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """

    mp = 0.5 * (high + low)
    ao = mp.rolling(s, min_periods=0).mean() - mp.rolling(len, min_periods=0).mean()

    if fillna:
        ao = ao.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ao, name='ao')


def kama(close, n=10, pow1=2, pow2=30, fillna=False):
    """Kaufman's Adaptive Moving Average (KAMA)

    Moving average designed to account for market noise or volatility. KAMA
    will closely follow prices when the price swings are relatively small and
    the noise is low. KAMA will adjust when the price swings widen and follow
    prices from a greater distance. This trend-following indicator can be
    used to identify the overall trend, time turning points and filter price
    movements.

    https://www.tradingview.com/ideas/kama/

    Args:
        close(pandas.Series): dataset 'Close' column
        n(int): n number of periods for the efficiency ratio
        pow1(int): number of periods for the fastest EMA constant
        pow2(int): number of periods for the slowest EMA constant

    Returns:
        pandas.Series: New feature generated.
    """
    close_values = close.values
    vol = pd.Series(abs(close - np.roll(close, 1)))

    ER_num = abs(close_values - np.roll(close_values, n))
    ER_den = vol.rolling(n).sum()
    ER = ER_num / ER_den

    sc = ((ER * (2.0 / (pow1 + 1) - 2.0 / (pow2 + 1.0)) + 2 / (pow2 + 1.0)) ** 2.0).values

    kama = np.zeros(sc.size)
    N = len(kama)
    first_value = True

    for i in range(N):
        if np.isnan(sc[i]):
            kama[i] = np.nan
        else:
            if first_value:
                kama[i] = close_values[i]
                first_value = False
            else:
                kama[i] = kama[i - 1] + sc[i] * (close_values[i] - kama[i - 1])

    kama = pd.Series(kama, name='kama', index=close.index)

    if fillna:
        kama = kama.replace([np.inf, -np.inf], np.nan).fillna(close)

    return kama
