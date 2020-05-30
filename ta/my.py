# -*- coding: utf-8 -*-
"""
.. module:: my
   :synopsis: My Indicators.

.. moduleauthor:: JianBing Wang(Hust)


def sma(close, n, fillna=False):
def negative_volume_index(close, volume, fillna=False):
def moving_average_convergence_divergence(close, n1_shoter=12, n2_longer=26, fillna=False):
def signal_line(close, n=9, n1=12, n2=26, fillna=False):
def stochastic_oscillator_k(high, low, close, n=14, fillna=False):
def stochastic_oscillator_d(high, low, close, n_k=14, n_d=3, fillna=False):
def alma(close, window=9, sigma=6, offset=0.85, fillna=False):
def aroon_oscillator(close, n=25, fillna=False):
def average_true_range(high, low, close, n=14, fillna=False):
def compare(t1, t2, tr):
def average_true_range_percent(high, low, close, n=14, fillna=False):
def smoothed_moving_average(close, n=30, shift=1, fillna=False):
def bill_willams_alligator_jaw(high, low, oscillator='jaw', n1=13, n2=8, fillna=False):
def chaikin_volatility_indicator(high, low, n=10, fillna=False):
def chande_kroll_stop(high, low, close, stop_='short', n=9, x=1.5, q=14, fillna=False):
def chande_momentum_oscillator(close, n=10, fillna=False):
def chopping_index(high, low, close, n=10, fillna=False):
def commodity_channel_indicator(high, low, close, n=10, c=.015, fillna=False):
def momentum(close, n=10, fillna=False):
def bollinger_band_middle(close,n=14,fillna=False):
def bollinger_band_lower(close,n=14,a=0.02,fillna=False):
def bollinger_band_upper(close,n=14,a=0.02,fillna=False):
def directional_movement_index(high, low, close, n=14, fillna=False):
def range_of_change(close, n=10, fillna=False):
def range_of_change_ratio(close, n=10, fillna=False):
def connors_rsi(close, n1=3, n2=2, n3=100, fillna=False):
def sma(close, n=10, fillna=False):
def wma(close, n=10, fillna=False):
def coppock_curve(close, n1=10, n2=14, n3=11, fillna=False):
def detrended_price_oscillator(close, n=10, fillna=False):
def directional_movement(close, ):
def donchian_channels(high, low, n, fillna=False):
def dema(close, n, fillna=False):
def eom(high, low, volume, c=0.0001, fillna=False):
def market_facilitation_index(high, low, volume, fillna=False):
def adxr(high, low, close, n=14, fillna=False):
def elder_ray_index(high, n=13, fillna=False):
def faster_stochastic_oscillator(high, low, close, type='k', n1=14, n2=3, fillna=False):
def hlb(*args, fillna=False):
def volatility_ratio(high, low, close, fillna=False):
def hma(close, n=14, fillna=False):
def intreday_intensity_index(high, low, close, volume, fillna=False):
def intreday_intensity_percent(high, low, close, volume, n=21, fillna=False):
def stochastic_oscillator(high, low, close, n=3, fillna=False):
def linear_regression(close, n=14, fillna=False):
def ppo(close, n1=12, n2=26, fillna=False):
def pvi(close, volume, fillna=False):
def pivot_points_high(high, n=5, fillna=False):
def pivot_points_low(low, n=5, fillna=False):
def pvt(close, volume, fillna=False):
def standard_deviation_channel(close, n=14, fillna=False):
def standard_error(close, n=14, fillna=False):
def stochastic_rsi(close, n=14, fillna=False):
def triangular_moving_average(close, n=14, fillna=False):
def trix(close,n=14,fillna=False):
def typical_price(high,low,close,fillna=False):
def ultimate_oscillator(high,low,close,n=6,m=10,p=14,fillna=False):
def variable_horizontal_filter(close,n=14,fillna=False):
def vidya(close,n=6,m=14,a=0.2,fillna=False):
def volume_ema(volume,n=14,fillna=False):
def volume_wma(volume,n=14,fillna=False):
def weighted_close(high,low,close,fillna=False):
"""
import talib

from .utils import *
from .utils import _fillna


def sma(close, n, fillna=False):
    sma = close.rolling(n, min_periods=0).mean()
    if fillna:
        nvi = sma.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(sma, name='sma')


def negative_volume_index(close, volume, fillna=False):
    """Negative Volume Index (NVI)

    It is used to identify primary market(stock, index, ETF and etc) movements
    and reversals. NVI is based on the running cumulative total of the percentage
    price change for the days with volume lover than for the previous day's volume.
     Basically, NVI reflects periods of declining volume.
      Positive Volume Index (PVI) on the other side reflect days
      when volume moved up (advancing volume).

    https://www.marketvolume.com/technicalanalysis/negativevolumeindex.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    nvi = pd.Series(data=np.nan, index=close.index, name='nvi')
    nvi.iloc[0] = 1000
    price_change = close.pct_change()
    c = volume >= volume.shift(1)

    for i in range(1, len(close)):
        nvi.iloc[i] = nvi.iloc[i - 1]
        if not c.iloc[i]:
            nvi.iloc[i] *= (1 + price_change.iloc[i])
    if fillna:
        nvi = nvi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(nvi, name='nvi')


def moving_average_convergence_divergence(close, n1_shoter=12, n2_longer=26, fillna=False):
    '''
    MACD Line is a result of taking a longer term EMA and subtracting it from a shorter term EMA.
    The most commonly used values are 26 days for the longer term EMA and 12 days for the shorter term EMA, but it is the trader's choice.

    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

    :param close:
    :param n1_shoter: the shorter period of macd_line,
    :param n2_longer: the longer period of macd_line,
    :param fillna:
    :return: pandas.Series: New feature generated.

    '''
    macd_line = ema(close, n1_shoter) - ema(close, n2_longer)
    if fillna:
        macd_line = _fillna(macd_line)
    return pd.Series(macd_line, name='macd')


def signal_line(close, n=9, n1=12, n2=26, fillna=False):
    '''
    Similar to above macd_line, The Signal Line is an EMA of the MACD Line described in Component 1.
The trader can choose what period length EMA to use for the Signal Line however 9 is the most common.
    :param close:
    :param n:
    :param n1:
    :param n2:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''
    macd_line = moving_average_convergence_divergence(close, n1, n2, fillna)

    signal = ema(macd_line, n)
    if fillna:
        signal = _fillna(signal)
    return pd.Series(signal, name='signal')


def stochastic_oscillator_k(high, low, close, n=14, fillna=False):
    '''
        %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100

        https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''
    _K = pd.Series(data=np.nan, index=close.index, name='_K')

    for i in range(len(close)):
        lowest = low.iloc[max(0, i - n):i].min()
        highest = high.iloc[max(0, i - n):i].max()
        _K.iloc[i] = (close.iloc[i] - lowest) / (highest - lowest) * 100

    if fillna:
        _K = _fillna(_K)
    return pd.Series(_K, name='so_k')


def stochastic_oscillator_d(high, low, close, n_k=14, n_d=3, fillna=False):
    '''
    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %K is multiplied by 100 to move the decimal point two places

    The default setting for the Stochastic Oscillator is 14 periods,
    which can be days, weeks, months or an intraday timeframe.
    A 14-period %K would use the most recent close, the highest high over the last 14 periods and
    the lowest low over the last 14 periods. %D is a 3-day simple moving average of %K.


    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''
    _K = stochastic_oscillator_k(high, low, close, n_k, fillna)
    _D = sma(_K, n_d)
    if fillna:
        _D = _fillna(_D)
    return pd.Series(_D, name='so_d')


def alma(close, window=9, sigma=6, offset=0.85, fillna=False):
    '''
    Arnaud Legoux Moving Average


    https://www.prorealcode.com/prorealtime-indicators/alma-arnaud-legoux-moving-average/
    :param close:
    :param window:
    :param sigma:
    :param offset:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''

    price = close
    m = (offset * (window - 1))
    s = window / sigma
    WtdSum = 0
    CumWt = 0

    alma = pd.Series(data=np.nan, index=close.index, name='alma')

    for k in range(window - 1):
        Wtd = math.exp(-(((k - m) ** 2) / (2 * (s ** 2))))
        WtdSum += Wtd * price[window - 1 - k]
        CumWt += Wtd

        alma.iloc[k] = WtdSum / CumWt

    if fillna:
        alma = _fillna(alma)
    return pd.Series(alma, name='alma')


def aroon_oscillator(close, n=25, fillna=False):
    '''
    Aroon_Oscillator

    Aroon Oscillator=Aroon Up−Aroon Down
    Aroon Up=100*(25-Periods since 25-period High)/25
    Aroon Down=100*(25-Periods since 25-period Low)/25

​
    https://www.investopedia.com/terms/a/aroonoscillator.asp

    :param close:
    :param n:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''
    _aro = pd.Series(data=np.nan, index=close.index)
    for index, t in enumerate(close.index):
        new_data = close[str((pd.to_datetime(t) - pd.to_timedelta(n, 'D')).date()):t]
        high_time = (pd.to_datetime(t) - pd.to_datetime(new_data.idxmax())).days
        low_time = (pd.to_datetime(t) - pd.to_datetime(new_data.idxmin())).days
        au = (n - high_time) / n * 100
        ad = (n - low_time) / n * 100
        _aro.iloc[index] = au - ad

    if fillna:
        _aro = _fillna(_aro)
    return pd.Series(_aro, name='aro')


def average_true_range(high, low, close, n=14, fillna=False):
    '''
    Average_true_range
    TR=max(high-low,abs(high-close_prev),abs(low-close_prev))
    ATR_0=mean(TR)
    ATR_i=(ATR_(i-1)*(n-1)+TR_i)/n

    https://en.wikipedia.org/wiki/Average_true_range


    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''
    t1 = high - low
    t2 = abs(high - close.shift(1))
    t3 = abs(low - close.shift(1))

    tr = pd.Series(data=np.nan, index=close.index, name='tr')

    def compare(t1, t2, tr):
        c1 = (t1 > t2)
        c2 = (t1 < t2)
        if c1.any():
            tr.loc[c1] = t1
        if c2.any():
            tr.loc[c2] = t2

    compare(t1, t2, tr)
    compare(t1, t3, tr)
    compare(t2, t3, tr)

    atr = pd.Series(data=np.nan, index=close.index, name='atr')
    atr.iloc[0] = tr.mean()
    for i in range(1, len(close)):
        atr.iloc[i] = (atr.iloc[i - 1] * (n - 1) + tr.iloc[i]) / n
    if fillna:
        atr = _fillna(atr)
    return pd.Series(atr, name='atr')


def average_true_range_percent(high, low, close, n=14, fillna=False):
    '''
        https://en.wikipedia.org/wiki/Average_true_range

    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''
    atr = average_true_range(high, low, close, n, fillna)
    atrp = atr / close * 100

    if fillna:
        atrp = _fillna(atrp)
    return pd.Series(atrp, name='atrp')


def smoothed_moving_average(close, n=30, shift=1, fillna=False):
    sum_1 = close.iloc[:n].sum()
    smma = pd.Series(data=np.nan, index=close.index, name='smma')

    smma.iloc[0] = sum_1 / n
    for i in range(1, len(close)):
        smma.iloc[i] = ((n - shift) * smma.iloc[i - 1] + close.iloc[i]) / n

    if fillna:
        smma = _fillna(smma)

    return pd.Series(smma, name='smma')


def bill_willams_alligator_jaw(high, low, oscillator='jaw', n1=13, n2=8, fillna=False):
    '''
    Bill_Willams_Alligator_Oscillator

    MEDIAN PRICE = (HIGH + LOW) / 2

    ALLIGATORS JAW = SMMA (MEDIAN PRICE, 13, 8)
    ALLIGATORS TEETH = SMMA (MEDIAN PRICE, 8, 5)
    ALLIGATORS LIPS = SMMA (MEDIAN PRICE, 5, 3)

    https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/go

    :param high:
    :param low:
    :param n1:
    :param n2:
    :param fillna:
    :return:pandas.Series: New feature generated.
    '''
    oscillator = oscillator.lower()
    assert oscillator in ['jaw', 'teeth', 'lips'], "oscillator must be in ['jaw','teeth','lips']"

    if n1 is None or n2 is None:
        if oscillator == 'jaw':
            n1 = 13
            n2 = 8
        elif oscillator == 'teeth':
            n1 = 8
            n2 = 5
        elif oscillator == 'lips':
            n1 = 5
            n2 = 3

    median_price = (high + low) / 2
    alligators_oscillator = smoothed_moving_average(median_price, n1, n2)

    if fillna:
        alligators_oscillator = _fillna(alligators_oscillator)
    return pd.Series(alligators_oscillator, name='alligators_{}'.format(oscillator))


def chaikin_volatility_indicator(high, low, n=10, fillna=False):
    '''
    Chaikin Volatility Indicator

    https://www.metastock.com/customer/resources/taaz/?p=120


    :param high:
    :param low:
    :param n:
    :param fillna:
    :return:
    '''
    hla = ema(high - low, n)
    cvi = 100 * hla.pct_change(periods=n)
    if fillna:
        cvi = _fillna(cvi)

    return pd.Series(cvi, index=high.index, name='cvi')


def chande_kroll_stop(high, low, close, stop_='short', n=9, x=1.5, q=14, fillna=False):
    '''
    Chande Kroll Stop

    the stop type must be in ['short' or 'long']


    https://www.tradingview.com/script/vMAD5SZT/


    :param high:
    :param low:
    :param close:
    :param stop_:
    :param n:
    :param x:
    :param q:
    :param fillna:
    :return:
    '''
    stop_ = stop_.lower()

    assert stop_ in ['short', 'long'], "stop_ must be in ['short', 'long']"

    first_high_stop = pd.Series(data=np.nan, index=close.index)
    first_low_stop = pd.Series(data=np.nan, index=close.index)
    for i in range(len(close)):
        first_high_stop.iloc[i] = high.iloc[max(i - n, 0):i].max()
        first_low_stop.iloc[i] = high.iloc[max(i - n, 0):i].min()
    first_high_stop -= average_true_range(high, low, close, n=n, fillna=fillna).apply(lambda t: t * x)
    first_low_stop += average_true_range(high, low, close, n=n, fillna=fillna).apply(lambda t: t * x)

    if stop_ == 'short':
        res = highest(first_high_stop, q)
    else:
        res = lowest(first_low_stop, q)

    if fillna:
        res = _fillna(res)
    return pd.Series(res, index=high.index, name='cks_{}'.format(stop_))


def chande_momentum_oscillator(close, n=10, fillna=False):
    '''
    Chande_Momentum_Oscillator

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo

    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    diff = close.diff(1)
    which_dn = diff < 0

    up, dn = diff, diff * 0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    s_up = up.rolling(n, min_periods=0).sum()
    s_down = up.rolling(n, min_periods=0).sum()

    cmo = 100 * (s_up - s_down) / (s_up + s_down)
    if fillna:
        cmo = _fillna(cmo)
    return pd.Series(cmo, index=close.index, name='cmo')


def chopping_index(high, low, close, n=10, fillna=False):
    '''
    Chopping Index:

    https://www.tradingview.com/wiki/Choppiness_Index_(CHOP)


    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    LOG_10 = np.log10
    atr = average_true_range(high, low, close, 1, fillna)
    sum_atr = atr.rolling(n, min_periods=0).sum()
    max_high = high.rolling(n, min_periods=0).max()
    min_low = low.rolling(n, min_periods=0).min()

    chop = 100 * LOG_10(sum_atr / (max_high - min_low)) / LOG_10(n)
    if fillna:
        chop = _fillna(chop)
    return pd.Series(chop, index=close.index, name='chop')


def commodity_channel_indicator(high, low, close, n=10, c=.015, fillna=False):
    '''
    Commodity Channel Indicator

    https://tulipindicators.org/cci


    :param high:
    :param low:
    :param close:
    :param n:
    :param c:
    :param fillna:
    :return:
    '''
    typeprice = (high + low + close) / 3
    atp = typeprice.rolling(n, min_periods=0).mean()

    md = typeprice.rolling(n, min_periods=0).std()

    cci = (typeprice - atp) / (c * md)
    if fillna:
        cci = _fillna(cci)
    return pd.Series(cci, index=close.index, name='cci')


def momentum(close, n=10, fillna=False):
    '''
    Momentum

    https://tulipindicators.org/mom

    Momentum
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    mom = close - close.shift(n, fill_value=close.mean())
    if fillna:
        mom = _fillna(mom)

    return pd.Series(mom, index=close.index, name='mom')


@after_return()
def bollinger_band_middle(close,n=14,fillna=False):
    '''

    Bollinger Band Middle
    https://tulipindicators.org/bbands


    :param close:
    :param n:
    :param fillna:
    :return:
    '''

    return close.rolling(n,min_periods=0).mean()



@after_return()
def bollinger_band_lower(close,n=14,a=0.02,fillna=False):
    '''

    Bollinger Band Middle
    https://tulipindicators.org/bbands


    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    middle=bollinger_band_middle(close,n,fillna)

    return middle-a*middle.rolling(n,min_periods=0).std()



@after_return()
def bollinger_band_upper(close,n=14,a=0.02,fillna=False):
    '''

    Bollinger Band Middle
    https://tulipindicators.org/bbands


    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    middle=bollinger_band_middle(close,n,fillna)

    return middle+a*middle.rolling(n,min_periods=0).std()

@after_return()
def directional_movement_index(high, low, close, n=14, fillna=False):
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'max'))
    pdn = low.combine(cs, lambda x1, x2: get_min_max(x1, x2, 'min'))
    tr = pdm - pdn

    trs_initial = np.zeros(n - 1)
    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs) - 1):
        trs[i] = trs[i - 1] - (trs[i - 1] / float(n)) + tr[n + i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(close) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio) - 1):
        dip_mio[i] = dip_mio[i - 1] - (dip_mio[i - 1] / float(n)) + pos[n + i]

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio) - 1):
        din_mio[i] = din_mio[i - 1] - (din_mio[i - 1] / float(n)) + neg[n + i]

    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        dip[i] = 100 * (dip_mio[i] / trs[i])

    din = np.zeros(len(trs))
    for i in range(len(trs)):
        din[i] = 100 * (din_mio[i] / trs[i])

    dx = 100 * np.abs((dip - din) / (dip + din))

    adx = np.zeros(len(trs))
    adx[n] = dx[0:n].mean()

    for i in range(n + 1, len(adx)):
        adx[i] = ((adx[i - 1] * (n - 1)) + dx[i - 1]) / float(n)

    adx = np.concatenate((trs_initial, adx), axis=0)

    adx=ema(ema(ema(adx,n),n),n)

    return adx
def range_of_change(close, n=10, fillna=False):
    '''
    Range Of Change

    https://tulipindicators.org/roc


    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    roc = (close - close.shift(n, fill_value=close.mean())) / close.shift(n)
    if fillna:
        roc = _fillna(roc)
    return pd.Series(roc, index=close.index, name='roc')


def range_of_change_ratio(close, n=10, fillna=False):
    '''
    Range Of Change Ratio

    https://tulipindicators.org/rocr


    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    rocr = close / close.shift(n, fill_value=close.mean())
    if fillna:
        rocr = _fillna(rocr)

    return pd.Series(rocr, index=close.index, name='rocr')


def connors_rsi(close, n1=3, n2=2, n3=100, fillna=False):
    '''
    Connors RSI

    https://www.tradingview.com/wiki/Connors_RSI_(CRSI)


    :param close:
    :param n1:
    :param n2:
    :param n3:
    :param fillna:
    :return:
    '''
    from .momentum import rsi
    rsi_n1 = rsi(close, n1, fillna)
    diff = (close.diff(1) != 0)
    updown_length = diff.rolling(n2, min_periods=0).sum()
    rsi_n2 = rsi(updown_length, n2, fillna)
    roc = range_of_change(close, n3)
    crsi = (rsi_n1 + rsi_n2 + roc) / 3
    if fillna:
        crsi = _fillna(crsi)
    return pd.Series(crsi, index=close.index, name='crsi')


def sma(close, n=10, fillna=False):
    sma = close.rolling(n, min_periods=0).mean()
    if fillna:
        sma = _fillna(sma)

    return pd.Series(sma, index=close.index, name='sma')


def wma(close, n=10, fillna=False):
    '''
    Weighted Moving Average


    https://tulipindicators.org/wma

    wma=2*(n-1)/(n+1)*SMA
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    sma_n = sma(close, n)
    wma = 2 * (n - 1) / (n + 1) * sma_n
    if fillna:
        wma = _fillna(wma)
    return pd.Series(wma, index=close.index, name='wma')


def coppock_curve(close, n1=10, n2=14, n3=11, fillna=False):
    '''
    Coppock Curve

    https://school.stockcharts.com/doku.php?id=technical_indicators:coppock_curve


    :param close:
    :param n1:
    :param n2:
    :param n3:
    :param fillna:
    :return:
    '''

    roc_ = range_of_change(close, n2, fillna) + range_of_change(close, n3, fillna)
    cc = wma(roc_, n1, fillna)
    if fillna:
        cc = _fillna(cc)
    return pd.Series(cc, index=close.index, name='coppock_curve')


def detrended_price_oscillator(close, n=10, fillna=False):
    '''
    Detrended Price Oscillator

    https://tulipindicators.org/dpo


    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    m = int(n / 2 + 1)
    dpo = close.shift(m) - sma(close, n)
    if fillna:
        dpo = _fillna(dpo)
    return pd.Series(dpo, index=close.index, name='dpo')


def directional_movement(close, ):
    pass


def donchian_channels(high, low, n, fillna=False):
    '''
    Donchian Channels Indicator

    https://www.investopedia.com/terms/d/donchianchannels.asp


    :param high:
    :param low:
    :param n:
    :param fillna:
    :return:
    '''
    uc = high.rolling(n, min_periods=0).max()
    lc = low.rolling(n, min_periods=0).min()

    middle_channel = (uc + lc) / 2

    if fillna:
        middle_channel = _fillna(middle_channel)

    return pd.Series(middle_channel, index=high.index, name='dc')


def dema(close, n, fillna=False):
    '''

    Double Exponetial Moving Average


    https://www.investopedia.com/terms/d/double-exponential-moving-average.asp

    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    dema = 2 * ema(close, n, fillna) - ema(ema(close, n, fillna), n, fillna)

    if fillna:
        dema = _fillna(dema)
    return pd.Series(dema, index=close.index, name='dema')


def eom(high, low, volume, c=0.0001, fillna=False):
    '''
    Ease of Movement


    https://tulipindicators.org/emv


    :param high:
    :param low:
    :param volume:
    :param c:
    :param fillna:
    :return:
    '''
    hl = (high + low) / 2
    br = volume * c / (high - low)

    emv = (hl - hl.shift(1, fill_value=hl.mean())) / br

    if fillna:
        emv = _fillna(emv)

    return pd.Series(emv, index=high.index, name='eom')


def market_facilitation_index(high, low, volume, fillna=False):
    '''
    Market Facilitation Index
    https://tulipindicators.org/marketfi
    :param high:
    :param low:
    :param volume:
    :param fillna:
    :return:
    '''

    marketfi = (high - low) / volume
    if fillna:
        marketfi = _fillna(marketfi)

    return pd.Series(marketfi, index=high.index, name='marketfi')


def adxr(high, low, close, n=14, fillna=False):
    '''

    Average Directional Movement Rating


    https://tulipindicators.org/adxr

    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    from .trend import adx
    adx = adx(high, low, close, n, fillna)
    adxr = (adx + adx.shift(n - 1)) / 2

    if fillna:
        adxr = _fillna(adxr)
    return pd.Series(adxr, index=adx.index, name='adxr')


def elder_ray_index(high, n=13, fillna=False):
    '''
    Elder Ray Index
    https://www.investopedia.com/terms/e/elderray.asp

    :param high:
    :param n:
    :param fillna:
    :return:
    '''
    _ema = ema(high, n, fillna)
    eri = high - _ema
    if fillna:
        eri = _fillna(eri)
    return pd.Series(eri, index=high.index, name='er')


def faster_stochastic_oscillator(high, low, close, type='k', n1=14, n2=3, fillna=False):
    '''
    Fast Stochastic Oscillator

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/fast-stochastic


    :param high:
    :param low:
    :param close:
    :param type:
    :param n1:
    :param n2:
    :param fillna:
    :return:
    '''
    type = type.lower()
    assert type in ['k', 'd'], 'type must be in ["k","d"]'

    Ln = low.rolling(n1, min_periods=0).min()
    Hn = high.rolling(n1, min_periods=0).max()

    _K = 100 * ((close - Ln) / Hn - Ln)

    if type == 'd':

        _D = sma(_K, n2, fillna)
        if fillna:
            _D = _fillna(_D)
        return pd.Series(_D, index=close.index, name='faster_stochastic_d')

    if fillna:
        _K = _fillna(_K)
    return pd.Series(_K, index=close.index, name='faster_stochastic_k')


def hlb(*args, fillna=False):
    '''
    High Low Bands


    https://library.tradingtechnologies.com/trade/chrt-ti-high-low-bands.html
    :param args:
    :param fillna:
    :return:
    '''
    hlb = 0
    for a in args:
        hlb += a

    hlb /= len(args)
    if fillna:
        hlb = _fillna(hlb)

    return pd.Series(hlb, index=args[0].index, name='hlb')


def volatility_ratio(high, low, close, fillna=False):
    '''

    Historical Volatility Ratio

    https://www.investopedia.com/ask/answers/031115/what-volatility-ratio-formula-and-how-it-calculated.asp


    :param high:
    :param low:
    :param close:
    :param fillna:
    :return:
    '''
    maximum = (high + close.shift(1, fill_value=close.mean())) / 2

    minimum = (low + close.shift(1, fill_value=close.mean())) / 2

    current_true_range = maximum - minimum

    ptr = high - low

    vr = current_true_range / ptr

    if fillna:
        vr = _fillna(vr)
    return pd.Series(vr, index=close.index, name='vr')


def hma(close, n=14, fillna=False):
    '''
    Hull Moving Average

    https://www.investopedia.com/ask/answers/031115/what-volatility-ratio-formula-and-how-it-calculated.asp


    :param close:
    :param n:
    :param fillna:
    :return:
    '''

    wma_1 = wma(close, n // 2, fillna).apply(lambda x: 2 * x)
    wma_2 = wma_1 - wma(close, n)
    hma = wma(wma_2, int(np.sqrt(n)))

    if fillna:
        hma = _fillna(hma)
    return pd.Series(hma, index=close.index, name='hma')


def intreday_intensity_index(high, low, close, volume, fillna=False):
    '''
    Intraday Intensity Index


    https://www.investopedia.com/terms/i/intradayintensityindex.asp

    :param high:
    :param low:
    :param close:
    :param volume:
    :param fillna:
    :return:
    '''
    iii = (close.apply(lambda x: 2 * x) - high - low) / ((high - low) * volume)
    if fillna:
        iii = _fillna(iii)
    return pd.Series(iii, index=close.index, name='iii')


def intreday_intensity_percent(high, low, close, volume, n=21, fillna=False):
    iii = intreday_intensity_index(high, low, close, volume, fillna)
    vs = volume.rolling(n, min_periods=0).sum()

    iip = iii / vs * 100
    if fillna:
        iip = _fillna(iip)
    return pd.Series(iip, index=close.index, name='iip')


def stochastic_oscillator(high, low, close, n=3, fillna=False):
    '''

    Lane's Stochastic Oscillator

    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full


    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    lowest = low.rolling(n, fillna).min()
    highest = high.rolling(n, fillna).max()

    so = (close - lowest) / (highest - lowest) * 100
    if fillna:
        kst = _fillna(so)
    return pd.Series(so, index=close.index, name='so')


def linear_regression(close, n=14, fillna=False):
    '''

    Linear Regression

    https://tulipindicators.org/linreg


    :param close:
    :param n:
    :param fillna:
    :return:
    '''

    x_ = (n + 1) / 2
    y_ = ema(close, n)

    belta = pd.Series(np.nan, index=close.index, name='Belta')
    alpha = pd.Series(np.nan, index=close.index, name='Alpha')

    for t in range(len(close)):
        # belta.iloc[t]=
        pass


def ppo(close, n1=12, n2=26, fillna=False):
    '''
    Percentage Price Oscillator

    https://tulipindicators.org/ppo


    :param close:
    :param n1:
    :param n2:
    :param fillna:
    :return:
    '''
    short = ema(close, n1)
    long = ema(close, n2)

    ppo = (short - long) / long * 100

    if fillna:
        ppo = _fillna(ppo)
    return pd.Series(ppo, index=close.index, name='ppo')


def pvi(close, volume, fillna=False):
    """Positive Volume Index (PVI)
        https://tulipindicators.org/pvi

        The Pegative Volume Index (PVI) is a cumulative indicator that uses the
        change in volume to decide when the smart money is active. Paul Dysart
        first developed this indicator in the 1930s. [...] Dysart's Negative Volume
        Index works under the assumption that the smart money is active on days
        when volume decreases and the not-so-smart money is active on days when
        volume increases.

        The cumulative PVI line was unchanged when volume increased from one
        period to the other. In other words, nothing was done. Norman Fosback, of
        Stock Market Logic, adjusted the indicator by substituting the percentage
        price change for Net Advances.

        This implementation is the Fosback version.

        If today's volume is greater than yesterday's volume then:
            nvi(t) = nvi(t-1) * ( 1 + (close(t) - close(t-1)) / close(t-1) )
        Else
            nvi(t) = nvi(t-1)

        Please note: the "stockcharts.com" example calculation just adds the
        percentange change of price to previous NVI when volumes decline; other
        sources indicate that the same percentage of the previous NVI value should
        be added, which is what is implemented here.

        Args:
            close(pandas.Series): dataset 'Close' column.
            volume(pandas.Series): dataset 'Volume' column.
            fillna(bool): if True, fill nan values with 1000.

        Returns:
            pandas.Series: New feature generated.

        See also:
        https://en.wikipedia.org/wiki/Positive_volume_index
        """

    price_change = close.pct_change()
    vol_decrease = (volume.shift(1) < volume)

    pvi = pd.Series(data=np.nan, index=close.index, dtype='float64', name='pvi')

    pvi.iloc[0] = 1000
    for i in range(1, len(pvi)):
        if vol_decrease.iloc[i]:
            pvi.iloc[i] = pvi.iloc[i - 1] * (1.0 + price_change.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]

    if fillna:
        pvi = pvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

    return pd.Series(pvi, name='pvi')


def pivot_points_high(high, n=5, fillna=False):
    '''
    Pivot Points High/Low


    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/pivot-points-high-low
    :param high:
    :param n:
    :param fillna:
    :return:
    '''
    pivot_points = pd.Series(np.ones(len(high)), index=high.index, name='pivot_points_high')
    for i in range(n):
        c = high.shift(i) < high
        pivot_points.loc[c] += 1
    if fillna:
        pivot_points = pivot_points.replace([np.inf, -np.inf], np.nan).fillna(1000)

    return pd.Series(pivot_points, name='pivot_points_high')


def pivot_points_low(low, n=5, fillna=False):
    '''

     Pivot Points High/Low


    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/pivot-points-high-low


    :param low:
    :param n:
    :param fillna:
    :return:
    '''
    pivot_points = pd.Series(np.ones(len(low)), index=low.index, name='pivot_points_low')
    for i in range(n):
        c = low.shift(i) > low
        pivot_points.loc[c] += 1
    if fillna:
        pivot_points = pivot_points.replace([np.inf, -np.inf], np.nan).fillna(1000)

    return pd.Series(pivot_points, name='pivot_points_low')


def pvt(close, volume, fillna=False):
    '''
    Price Volume Trend Indicator


    https://www.tradingview.com/wiki/Price_Volume_Trend_(PVT)


    :param close:
    :param volume:
    :param fillna:
    :return:
    '''
    close_percent = close.pct_change()
    pvt = close_percent * volume
    pvt = pvt.cumsum()

    if fillna:
        pvt = _fillna(pvt)

    return pd.Series(pvt, index=close.index, name='pvt')


def standard_deviation_channel(close, n=14, fillna=False):
    '''
    Standard Deviation Channel
    https://tulipindicators.org/stddev

    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    std = close.rolling(n, min_periods=0).std()

    std = std.applay(lambda x: np.sqrt(x))
    if fillna:
        std = _fillna(std)

    return pd.Series(std, index=close.index, name='sdc')


def standard_error(close, n=14, fillna=False):
    '''
    Standard Error
    https://tulipindicators.org/stderr
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    std = close.rolling(n, min_periods=0).std()

    stde = std.applay(lambda x: np.sqrt(x) / np.sqrt(n))
    if fillna:
        stde = _fillna(stde)

    return pd.Series(stde, index=close.index, name='se')


@after_return()
def stochastic_rsi(close, n=14, fillna=False):
    from .momentum import rsi
    rsi = rsi(close, n, fillna)

    max_rsi = rsi.rolling(n, min_periods=0).max()
    min_rsi = rsi.rolling(n, min_periods=0).min()

    srsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    return srsi


@after_return()
def triangular_moving_average(close, n=14, fillna=False):

    '''
    Triangular Moving Average
    https://tulipindicators.org/trima
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    if n % 2 == 0:
        tma = sma(sma(close, n // 2), n // 2 + 1)
    else:
        tma = sma(sma(close, (n + 1) // 2), (n + 1) // 2)

    return tma


def trix(close,n=14,fillna=False):
    '''

    https://tulipindicators.org/trix

    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    m=ema(ema(ema(close,n),n),n)
    trix=100*m.pct_change()

    return trix

@after_return()
def typical_price(high,low,close,fillna=False):
    '''
    Typical Price
    https://tulipindicators.org/typprice
    :param high:
    :param low:
    :param close:
    :param fillna:
    :return:
    '''
    tp=(high+low+close)/3
    return tp

@after_return()
def ultimate_oscillator(high,low,close,n=6,m=10,p=14,fillna=False):
    '''

    Ultimate Oscillator
    https://tulipindicators.org/ultosc
    :param high:
    :param low:
    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    c1=low<close.shift(1)
    c2=high>close.shift(1)

    tl=pd.Series(close.shift(1),index=close.index,name='tl')
    th=pd.Series(close.shift(1),index=close.index,name='th')

    if c1.any():
        tl.loc[c1]=low
    if c2.any():
        th.loc[c2]=high

    bp=close-tl
    r=th-tl

    t1=bp.rolling(n,min_periods = 0).sum()/r.rolling(n,min_periods = 0).sum()
    t2=bp.rolling(m,min_periods = 0).sum()/r.rolling(m,min_periods = 0).sum()
    t3=bp.rolling(p,min_periods = 0).sum()/r.rolling(p,min_periods = 0).sum()

    ultosc=100/7*(4*t1+2*t2+t3)

    return ultosc

@after_return()
def variable_horizontal_filter(close,n=14,fillna=False):
    hcp=close.rolling(n,min_periods =0).max()
    lcp=close.rolling(n,min_periods =0).min()

    diff=close.diff(1)
    which=diff<0
    _abs=diff
    _abs[which]=-_abs[which]

    vhf=abs(hcp-lcp)/_abs.cumsum()
    return vhf


@after_return()
def vidya(close,n=6,m=14,a=0.2,fillna=False):
    '''

    VIDYA

    https://tulipindicators.org/vidya

    :param close:
    :param n:
    :param m:
    :param a:
    :param fillna:
    :return:
    '''
    short_std=close.rolling(n, min_periods = 0).std()
    long_std=close.rolling(n, min_periods = 0).std()

    s=a*short_std/long_std
    vidya=ema(s,n)
    return vidya

@after_return()
def volume_ema(volume,n=14,fillna=False):
    return ema(volume,n,fillna)

@after_return()
def volume_wma(volume,n=14,fillna=False):
    return wma(volume,n,fillna)

@after_return()
def weighted_close(high,low,close,fillna=False):

    '''

    https://tulipindicators.org/wcprice
    :param high:
    :param low:
    :param close:
    :param fillna:
    :return:
    '''
    wc=(high+low+2*close)/4
    return wc