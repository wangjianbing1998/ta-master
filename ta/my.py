# -*- coding: utf-8 -*-
"""
.. module:: my
   :synopsis: My Indicators.

.. moduleauthor:: JianBing Wang(Hust)

"""

import math

from .momentum import rsi, stoch_signal, stoch, uo, wr
from .utils import *
from .utils import _fillna
from .volatility import average_true_range


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
    return pd.Series(alma, index=close.index, name='alma')


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
        start = str((pd.to_datetime(t) - pd.to_timedelta(n, 'D')).date())
        new_data = close[start:t]
        high_time = (pd.to_datetime(t) - pd.to_datetime(new_data.idxmax())).days
        low_time = (pd.to_datetime(t) - pd.to_datetime(new_data.idxmin())).days
        au = (n - high_time) / n * 100
        ad = (n - low_time) / n * 100
        _aro.iloc[index] = au - ad

    if fillna:
        _aro = _fillna(_aro)
    return pd.Series(_aro, name='aro')


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
    cvi.name = 'cvi'
    return cvi


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

    cmo = 100 * (s_up) / (s_up + s_down)
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


def _momentum(close, n=10, fillna=False):
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
def bollinger_band_middle(close, n=14, fillna=False):
    '''

    Bollinger Band Middle
    https://tulipindicators.org/bbands


    :param close:
    :param n:
    :param fillna:
    :return:
    '''

    return close.rolling(n, min_periods=0).mean()


def bollinger_band_lower(close, n=14, a=0.02, fillna=False):
    '''

    Bollinger Band Middle
    https://tulipindicators.org/bbands


    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    middle = bollinger_band_middle(close, n=n, fillna=fillna)

    value = middle - a * middle.rolling(n, min_periods=0).std()
    if fillna:
        value = value.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(value, name='bollinger_band_lower')


def bollinger_band_upper(close, n=14, a=0.02, fillna=False):
    '''

    Bollinger Band Middle
    https://tulipindicators.org/bbands


    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    middle = bollinger_band_middle(close, n=n, fillna=fillna)

    value = middle + a * middle.rolling(n, min_periods=0).std()
    if fillna:
        value = value.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(value, name='bollinger_band_upper')


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


def donchian_channels(high, low, n=14, fillna=False):
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


def dema(close, n=14, fillna=False):
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
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()

    so = (close - lowest) / (highest - lowest) * 100
    if fillna:
        kst = _fillna(so)
    return pd.Series(so, index=close.index, name='so')


@after_return()
def linear_regression(close, n=14, fillna=False):
    '''

    Linear Regression

    https://tulipindicators.org/linreg

    https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/

    :param close:
    :param n:
    :param fillna:
    :return:
    '''

    from scipy.stats import linregress
    def momentum(closes):
        returns = np.log(closes)
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        return ((1 + slope) ** 252) * (rvalue ** 2)  # annualize slope and multiply by R^2

    lineareg = close.rolling(n).apply(momentum, raw=False)
    if fillna:
        lineareg = _fillna(lineareg)
    return pd.Series(lineareg, index=close.index, name='linear_regression')


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

    std = std.apply(lambda x: np.sqrt(x))
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

    stde = std.apply(lambda x: np.sqrt(x) / np.sqrt(n))
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


def trix(close, n=14, fillna=False):
    '''

    https://tulipindicators.org/trix

    :param close:
    :param n:
    :param fillna:
    :return:
    '''
    m = ema(ema(ema(close, n), n), n)
    trix = 100 * m.pct_change()

    return trix


@after_return()
def typical_price(high, low, close, fillna=False):
    '''
    Typical Price
    https://tulipindicators.org/typprice
    :param high:
    :param low:
    :param close:
    :param fillna:
    :return:
    '''
    tp = (high + low + close) / 3
    return tp


@after_return()
def ultimate_oscillator(high, low, close, n=6, m=10, p=14, fillna=False):
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
    c1 = low < close.shift(1)
    c2 = high > close.shift(1)

    tl = pd.Series(close.shift(1), index=close.index, name='tl')
    th = pd.Series(close.shift(1), index=close.index, name='th')

    if c1.any():
        tl.loc[c1] = low
    if c2.any():
        th.loc[c2] = high

    bp = close - tl
    r = th - tl

    t1 = bp.rolling(n, min_periods=0).sum() / r.rolling(n, min_periods=0).sum()
    t2 = bp.rolling(m, min_periods=0).sum() / r.rolling(m, min_periods=0).sum()
    t3 = bp.rolling(p, min_periods=0).sum() / r.rolling(p, min_periods=0).sum()

    ultosc = 100 / 7 * (4 * t1 + 2 * t2 + t3)

    return ultosc


@after_return()
def variable_horizontal_filter(close, n=14, fillna=False):
    hcp = close.rolling(n, min_periods=0).max()
    lcp = close.rolling(n, min_periods=0).min()

    diff = close.diff(1)
    which = diff < 0
    _abs = diff
    _abs[which] = -_abs[which]

    vhf = abs(hcp - lcp) / _abs.cumsum()
    return vhf


@after_return()
def vidya(close, n=6, m=14, a=0.2, fillna=False):
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
    short_std = close.rolling(n, min_periods=0).std()
    long_std = close.rolling(m, min_periods=0).std()

    s = a * short_std / long_std
    vidya = ema(s, n)
    return vidya


@after_return()
def volume_ema(volume, n=14, fillna=False):
    return ema(volume, n, fillna)


@after_return()
def volume_wma(volume, n=14, fillna=False):
    return wma(volume, n, fillna)


@after_return()
def weighted_close(high, low, close, fillna=False):
    '''

    https://tulipindicators.org/wcprice
    :param high:
    :param low:
    :param close:
    :param fillna:
    :return:
    '''
    wc = (high + low + 2 * close) / 4
    return wc


@after_return()
def rmi(close, x=1, period=14, fillna=False):
    diff = close.diff(x)
    which_dn = diff < 0

    up, dn = diff, diff * 0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, period, fillna)
    emadn = ema(dn, period, fillna)

    rmi_ = 100 * emaup / (emaup + emadn)
    if fillna:
        rmi_ = rmi_.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rmi_, name='rmi')


@after_return()
def rmi_expansion(close, high=None, low=None, period=14, n=14, variation=2, fillna=False):
    if variation == 1:
        ma = close - sma(close, period)
        which_dn = ma < 0

        up, dn = ma, ma * 0
        up[which_dn], dn[which_dn] = 0, -up[which_dn]

    elif variation == 2:
        diff = close.diff(1)
        which_dn = diff < 0

        up, dn = diff, diff * 0
        up[which_dn], dn[which_dn] = 0, -up[which_dn]

    elif variation == 3:
        diff = close.diff(1)
        which_dn = diff < 0

        up, dn = diff, diff * 0
        up[which_dn], dn[which_dn] = 0, 1
        up[diff > 0], dn[diff > 0] = 1, 0
    elif variation == 4:
        ma = close - sma(low, period)
        which_dn = close > sma(high, period)

        up, dn = ma, ma * 0
        up[which_dn], dn[which_dn] = 0, abs(close - sma(high, period))
    elif variation == 5:
        ma = close - sma(low, period)

        up, dn = ma, ma * 0
        up[close > sma(high, period)], dn[close > sma(low, period)] = 0, 1
        up[close < sma(high, period)], dn[close < sma(high, period)] = 1, 0
    else:
        raise ValueError(f'Unexpected variation ={variation}, Expecting 1,2,3,4,5')
    emaup = ema(up, n, fillna)
    emadn = ema(dn, n, fillna)

    rsi = 100 * emaup / (emaup + emadn)
    if fillna:
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rsi, name=f'rsi_variation_{variation}')


@after_return()
def spearman(close, period=9, fillna=False):
    '''

https://zh.wikipedia.org/wiki/%E6%96%AF%E7%9A%AE%E5%B0%94%E6%9B%BC%E7%AD%89%E7%BA%A7%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0

    :param close:
    :param period:
    :param fillna:
    :return:
    '''

    close_1 = close.shift(-period)
    square_ = (close - close_1) ** 2
    sum_square = square_.rolling(period).sum()
    p = 1 - 6 * sum_square / (period * (period ** 2 - 1))
    return p


@after_return()
def fisher_transform(close, fillna=False):
    '''

    https://en.wikipedia.org/wiki/Fisher_transformation

    https://www.daytrading.com/fisher-transform


    Fisher Transform

    :param close:
    :param fillna:
    :return:
    '''
    x = (close - close.min()) / (close.max() - close.min()) * 2 - 1
    ft = pd.Series([10 * math.log((1 + x_) / (1 - x_ + 1e-3) + 1e-3) for x_ in x.values], index=close.index)
    return ft


@after_return()
def td_count(close, x=4, period=9, variation=1, fillna=False):
    if variation == 1:
        diff = close.diff(x)
        which_dn = diff > 0
        up = 0 * diff
        up[which_dn] = 1
    elif variation == 2:
        pass

    else:
        raise ValueError(f'Unexpected variation ={variation}, Expecting 1,2 ')
    td = sma(up, period, fillna)
    if fillna:
        td = td.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(td, name='td_count')


@after_return()
def trend_stall(close, period=9, fillna=False):
    roc = rate_of_change(close, period, fillna)
    roc_diff1 = roc.diff(1)
    roc_diff2 = roc.diff(2)

    signal = roc * 0
    signal[(roc_diff1 > roc_diff2) & (roc < roc_diff1)] = 1
    if fillna:
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(signal, name='trend_stall')


@after_return()
def atm_stochastic_oscillator_signal(high, low, close, n=14, d_n=3, fillna=False):
    sos = stoch_signal(high, low, close, n, d_n=d_n, fillna=fillna)
    sos_sma = sma(sos, n)
    wnr_ = wr(high, low, sos_sma, n)

    wrn_ = sma(wnr_, n)

    return wrn_


@after_return()
def atm_stochastic_oscillator(high, low, close, n=14, fillna=False):
    sos = stoch(high, low, close, n, fillna=fillna)
    sos_sma = sma(sos, n)
    wnr_ = wr(high, low, sos_sma, n)
    wrn_ = sma(wnr_, n)

    return wrn_


@after_return()
def atm_ultimate_oscillator(high, low, close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
                            fillna=False):
    sos = uo(high, low, close, s=s, m=m, len=len, ws=ws, wm=wm, wl=wl,
             fillna=fillna)
    sos_sma = sma(sos, m)
    wnr_ = wr(high, low, sos_sma, m)
    wrn_ = sma(wnr_, m)

    return wrn_


@after_return()
def rate_of_change(close, n=14, fillna=False):
    '''

    ROC(Rate Of Change).
    ROC=(Close-Close.shift(-n))/(Close.shift(-n))*100%

    https://www.investopedia.com/terms/p/pricerateofchange.asp

    :return:
    '''

    roc = (close - close.shift(-n)) / (close.shift(-n)) * 100.0
    return roc


@after_return()
def atr(close, n=1, period=10, fillna=False):
    close_ = close / close.shift(-1) - 1
    atr_ = sma(close_, period)
    return atr_


@after_return()
def sharp_ratio_volatility(close, volatility=None, period=14, fillna=False):
    if volatility is None:
        volatility = close
    srv = close / sma(close, period) / volatility
    return srv


@after_return()
def sharp_ratio_atr(close, period=14, fillna=False):
    atr_ = atr(close, period)
    sra = close / sma(close, period) / atr_
    return sra


@after_return()
def divergence(close, high, low, period=9, fillna=False):
    A1 = rsi(close, 14, fillna)
    A2 = rate_of_change(close, 14, fillna)

    A3 = wr(high, low, A1, period)
    A4 = wr(high, low, A2, period)

    A5 = A3 - A4

    return A5


@after_return()
def moving_count(close, period=14, fillna=False):
    sma_ = sma(close, period)
    which_dn = close > sma_
    data = close * 0
    data[which_dn] = 1
    mc = sma(data, period)
    return mc
