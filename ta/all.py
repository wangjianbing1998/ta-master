'''
def acc_dist_index(high, low, close, volume, fillna=False):
def adx(high, low, close, n=14, fillna=False):
def adx_neg(high, low, close, n=14, fillna=False):
def adx_pos(high, low, close, n=14, fillna=False):
def adxr(high, low, close, n=14, fillna=False):
def alma(close, window=9, sigma=6, offset=0.85, fillna=False):
def ao(high, low, s=5, len=34, fillna=False):
def aroon_down(close, n=25, fillna=False):
def aroon_oscillator(close, n=25, fillna=False):
def aroon_up(close, n=25, fillna=False):
def atm_stochastic_oscillator(high, low, close, n=14, fillna=False):
def atm_stochastic_oscillator_signal(high, low, close, n=14, d_n=3, fillna=False):
def atm_ultimate_oscillator(high, low, close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
def atr(close, n=1, period=10, fillna=False):
def average_true_range(high, low, close, n=14, fillna=False):
def average_true_range_percent(high, low, close, n=14, fillna=False):
def bill_willams_alligator_jaw(high, low, oscillator='jaw', n1=13, n2=8, fillna=False):
def bollinger_band_lower(close, n=14, a=0.02, fillna=False):
def bollinger_band_middle(close, n=14, fillna=False):
def bollinger_band_upper(close, n=14, a=0.02, fillna=False):
def bollinger_hband(close, n=20, ndev=2, fillna=False):
def bollinger_hband_indicator(close, n=20, ndev=2, fillna=False):
def bollinger_lband(close, n=20, ndev=2, fillna=False):
def bollinger_lband_indicator(close, n=20, ndev=2, fillna=False):
def bollinger_mavg(close, n=20, fillna=False):
def cci(high, low, close, n=20, c=0.015, fillna=False):
def chaikin_money_flow(high, low, close, volume, n=20, fillna=False):
def chaikin_volatility_indicator(high, low, n=10, fillna=False):
def chande_kroll_stop(high, low, close, stop_='short', n=9, x=1.5, q=14, fillna=False):
def chande_momentum_oscillator(close, n=10, fillna=False):
def chopping_index(high, low, close, n=10, fillna=False):
def commodity_channel_indicator(high, low, close, n=10, c=.015, fillna=False):
def connors_rsi(close, n1=3, n2=2, n3=100, fillna=False):
def coppock_curve(close, n1=10, n2=14, n3=11, fillna=False):
def cumulative_return(close, fillna=False):
def daily_log_return(close, fillna=False):
def daily_return(close, fillna=False):
def dema(close, n=14, fillna=False):
def detrended_price_oscillator(close, n=10, fillna=False):
def directional_movement(close, ):
def directional_movement_index(high, low, close, n=14, fillna=False):
def divergence(close, high, low, period=9, fillna=False):
def donchian_channel_hband(close, n=20, fillna=False):
def donchian_channel_hband_indicator(close, n=20, fillna=False):
def donchian_channel_lband(close, n=20, fillna=False):
def donchian_channel_lband_indicator(close, n=20, fillna=False):
def donchian_channels(high, low, n=14, fillna=False):
def dpo(close, n=20, fillna=False):
def ease_of_movement(high, low, close, volume, n=20, fillna=False):
def elder_ray_index(high, n=13, fillna=False):
def ema_indicator(close, n=12, fillna=False):
def eom(high, low, volume, c=0.0001, fillna=False):
def faster_stochastic_oscillator(high, low, close, type='k', n1=14, n2=3, fillna=False):
def fisher_transform(close, fillna=False):
def force_index(close, volume, n=2, fillna=False):
def hlb(*args, fillna=False):
def hma(close, n=14, fillna=False):
def ichimoku_a(high, low, n1=9, n2=26, visual=False, fillna=False):
def ichimoku_b(high, low, n2=26, n3=52, visual=False, fillna=False):
def intreday_intensity_index(high, low, close, volume, fillna=False):
def intreday_intensity_percent(high, low, close, volume, n=21, fillna=False):
def kama(close, n=10, pow1=2, pow2=30, fillna=False):
def keltner_channel_central(high, low, close, n=10, fillna=False):
def keltner_channel_hband(high, low, close, n=10, fillna=False):
def keltner_channel_hband_indicator(high, low, close, n=10, fillna=False):
def keltner_channel_lband(high, low, close, n=10, fillna=False):
def keltner_channel_lband_indicator(high, low, close, n=10, fillna=False):
def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False):
def kst_sig(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False):
def linear_regression(close, n=14, fillna=False):
def macd(close, n_fast=12, n_slow=26, fillna=False):
def macd_diff(close, n_fast=12, n_slow=26, n_sign=9, fillna=False):
def macd_signal(close, n_fast=12, n_slow=26, n_sign=9, fillna=False):
def market_facilitation_index(high, low, volume, fillna=False):
def mass_index(high, low, n=9, n2=25, fillna=False):
def momentum(close, n=10, fillna=False):
def momentum(closes):
def money_flow_index(high, low, close, volume, n=14, fillna=False):
def moving_average_convergence_divergence(close, n1_shoter=12, n2_longer=26, fillna=False):
def moving_count(close, period=14, fillna=False):
def negative_volume_index(close, volume, fillna=False):
def on_balance_volume(close, volume, fillna=False):
def pivot_points_high(high, n=5, fillna=False):
def pivot_points_low(low, n=5, fillna=False):
def ppo(close, n1=12, n2=26, fillna=False):
def put_call_ratio():
def pvi(close, volume, fillna=False):
def pvt(close, volume, fillna=False):
def range_of_change(close, n=10, fillna=False):
def range_of_change_ratio(close, n=10, fillna=False):
def rate_of_change(close, n=14, fillna=False):
def rmi(close, x=1, period=14, fillna=False):
def rmi_expansion(close, high=None, low=None, period=14, n=14, variation=2, fillna=False):
def rsi(close, n=14, fillna=False):
def sharp_ratio_atr(close, period=14, fillna=False):
def sharp_ratio_volatility(close, volatility=None, period=14, fillna=False):
def signal_line(close, n=9, n1=12, n2=26, fillna=False):
def sma(close, n=10, fillna=False):
def smoothed_moving_average(close, n=30, shift=1, fillna=False):
def spearman(close, period=9, fillna=False):
def standard_deviation_channel(close, n=14, fillna=False):
def standard_error(close, n=14, fillna=False):
def stoch(high, low, close, n=14, fillna=False):
def stoch_signal(high, low, close, n=14, d_n=3, fillna=False):
def stochastic_oscillator(high, low, close, n=3, fillna=False):
def stochastic_oscillator_d(high, low, close, n_k=14, n_d=3, fillna=False):
def stochastic_oscillator_k(high, low, close, n=14, fillna=False):
def stochastic_rsi(close, n=14, fillna=False):
def td_count(close, x=4, period=9, variation=1, fillna=False):
def trend_stall(close, period=9, fillna=False):
def triangular_moving_average(close, n=14, fillna=False):
def trix(close, n=14, fillna=False):
def trix(close, n=15, fillna=False):
def tsi(close, r=25, s=13, fillna=False):
def typical_price(high, low, close, fillna=False):
def ultimate_oscillator(high, low, close, n=6, m=10, p=14, fillna=False):
def uo(high, low, close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
def variable_horizontal_filter(close, n=14, fillna=False):
def vidya(close, n=6, m=14, a=0.2, fillna=False):
def volatility_ratio(high, low, close, fillna=False):
def volume_ema(volume, n=14, fillna=False):
def volume_price_trend(close, volume, fillna=False):
def volume_wma(volume, n=14, fillna=False):
def vortex_indicator_neg(high, low, close, n=14, fillna=False):
def vortex_indicator_pos(high, low, close, n=14, fillna=False):
def weighted_close(high, low, close, fillna=False):
def wma(close, n=10, fillna=False):
def wr(high, low, close, lbp=14, fillna=False):




'''
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


import math

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
    long_std = close.rolling(n, min_periods=0).std()

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


def daily_return(close, fillna=False):
    """Daily Return (DR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    dr = (close / close.shift(1, fill_value=close.mean())) - 1
    dr *= 100
    if fillna:
        dr = dr.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(dr, name='d_ret')


def daily_log_return(close, fillna=False):
    """Daily Log Return (DLR)

    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    dr = np.log(close).diff()
    dr *= 100
    if fillna:
        dr = dr.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(dr, name='d_logret')


def cumulative_return(close, fillna=False):
    """Cumulative Return (CR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    cr = (close / close.iloc[0]) - 1
    cr *= 100
    if fillna:
        cr = cr.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(cr, name='cum_ret')


def macd(close, n_fast=12, n_slow=26, fillna=False):
    """Moving Average Convergence Divergence (MACD)

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = ema(close, n_fast, fillna)
    emaslow = ema(close, n_slow, fillna)
    macd = emafast - emaslow
    if fillna:
        macd = macd.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(macd, name='MACD_%d_%d' % (n_fast, n_slow))


def macd_signal(close, n_fast=12, n_slow=26, n_sign=9, fillna=False):
    """Moving Average Convergence Divergence (MACD Signal)

    Shows EMA of MACD.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = ema(close, n_fast, fillna)
    emaslow = ema(close, n_slow, fillna)
    macd = emafast - emaslow
    macd_signal = ema(macd, n_sign, fillna)
    if fillna:
        macd_signal = macd_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(macd_signal, name='MACD_sign')


def macd_diff(close, n_fast=12, n_slow=26, n_sign=9, fillna=False):
    """Moving Average Convergence Divergence (MACD Diff)

    Shows the relationship between MACD and MACD Signal.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = ema(close, n_fast, fillna)
    emaslow = ema(close, n_slow, fillna)
    macd = emafast - emaslow
    macdsign = ema(macd, n_sign, fillna)
    macd_diff = macd - macdsign
    if fillna:
        macd_diff = macd_diff.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(macd_diff, name='MACD_diff')


def ema_indicator(close, n=12, fillna=False):
    """EMA

    Exponential Moving Average via Pandas

    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    ema_ = ema(close, n, fillna)
    return pd.Series(ema_, name='ema')


def adx(high, low, close, n=14, fillna=False):
    """Average Directional Movement Index (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
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
    adx = pd.Series(data=adx, index=close.index)

    if fillna:
        adx = adx.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(adx, name='adx')


def adx_pos(high, low, close, n=14, fillna=False):
    """Average Directional Movement Index Positive (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
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

    dip = np.zeros(len(close))
    for i in range(1, len(trs) - 1):
        dip[i + n] = 100 * (dip_mio[i] / trs[i])

    dip = pd.Series(data=dip, index=close.index)

    if fillna:
        dip = dip.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(dip, name='adx_pos')


def adx_neg(high, low, close, n=14, fillna=False):
    """Average Directional Movement Index Negative (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
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

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio) - 1):
        din_mio[i] = din_mio[i - 1] - (din_mio[i - 1] / float(n)) + neg[n + i]

    din = np.zeros(len(close))
    for i in range(1, len(trs) - 1):
        din[i + n] = 100 * (din_mio[i] / float(trs[i]))

    din = pd.Series(data=din, index=close.index)

    if fillna:
        din = din.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(din, name='adx_neg')


def vortex_indicator_pos(high, low, close, n=14, fillna=False):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    tr = (high.combine(close.shift(1, fill_value=close.mean()), max)
          - low.combine(close.shift(1, fill_value=close.mean()), min))
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1, fill_value=low.mean()))
    vmm = np.abs(low - high.shift(1, fill_value=high.mean()))

    vip = vmp.rolling(n, min_periods=0).sum() / trn
    if fillna:
        vip = vip.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vip, name='vip')


def vortex_indicator_neg(high, low, close, n=14, fillna=False):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bearish signal triggers when the negative trend indicator
    crosses above the positive trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vin = vmm.rolling(n).sum() / trn
    if fillna:
        vin = vin.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vin, name='vin')


def trix(close, n=15, fillna=False):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    ema1 = ema(close, n, fillna)
    ema2 = ema(ema1, n, fillna)
    ema3 = ema(ema2, n, fillna)
    trix = (ema3 - ema3.shift(1, fill_value=ema3.mean())) / ema3.shift(1, fill_value=ema3.mean())
    trix *= 100
    if fillna:
        trix = trix.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(trix, name='trix_' + str(n))


def mass_index(high, low, n=9, n2=25, fillna=False):
    """Mass Index (MI)

    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of
    the current trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n(int): n low period.
        n2(int): n high period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    amplitude = high - low
    ema1 = ema(amplitude, n, fillna)
    ema2 = ema(ema1, n, fillna)
    mass = ema1 / ema2
    mass = mass.rolling(n2, min_periods=0).sum()
    if fillna:
        mass = mass.replace([np.inf, -np.inf], np.nan).fillna(n2)
    return pd.Series(mass, name='mass_index_' + str(n))


def cci(high, low, close, n=20, c=0.015, fillna=False):
    """Commodity Channel Index (CCI)

    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        c(int): constant.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    pp = (high + low + close) / 3.0
    cci = (pp - pp.rolling(n, min_periods=0).mean()) / (c * pp.rolling(n, min_periods=0).std())
    if fillna:
        cci = cci.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cci, name='cci')


def dpo(close, n=20, fillna=False):
    """Detrended Price Oscillator (DPO)

    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    dpo = close.shift(int((0.5 * n) + 1), fill_value=close.mean()) - close.rolling(n, min_periods=0).mean()
    if fillna:
        dpo = dpo.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(dpo, name='dpo_' + str(n))


def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False):
    """KST Oscillator (KST)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    https://en.wikipedia.org/wiki/KST_oscillator

    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = ((close - close.shift(r1, fill_value=close.mean()))
              / close.shift(r1, fill_value=close.mean())).rolling(n1, min_periods=0).mean()
    rocma2 = ((close - close.shift(r2, fill_value=close.mean()))
              / close.shift(r2, fill_value=close.mean())).rolling(n2, min_periods=0).mean()
    rocma3 = ((close - close.shift(r3, fill_value=close.mean()))
              / close.shift(r3, fill_value=close.mean())).rolling(n3, min_periods=0).mean()
    rocma4 = ((close - close.shift(r4, fill_value=close.mean()))
              / close.shift(r4, fill_value=close.mean())).rolling(n4, min_periods=0).mean()
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    if fillna:
        kst = kst.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(kst, name='kst')


def kst_sig(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False):
    """KST Oscillator (KST Signal)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst

    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.
        nsig(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = ((close - close.shift(r1, fill_value=close.mean()))
              / close.shift(r1, fill_value=close.mean())).rolling(n1, min_periods=0).mean()
    rocma2 = ((close - close.shift(r2, fill_value=close.mean()))
              / close.shift(r2, fill_value=close.mean())).rolling(n2, min_periods=0).mean()
    rocma3 = ((close - close.shift(r3, fill_value=close.mean()))
              / close.shift(r3, fill_value=close.mean())).rolling(n3, min_periods=0).mean()
    rocma4 = ((close - close.shift(r4, fill_value=close.mean()))
              / close.shift(r4, fill_value=close.mean())).rolling(n4, min_periods=0).mean()
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst_sig = kst.rolling(nsig, min_periods=0).mean()
    if fillna:
        kst_sig = kst_sig.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(kst_sig, name='kst_sig')


def ichimoku_a(high, low, n1=9, n2=26, visual=False, fillna=False):
    """Ichimoku Kinkō Hyō (Ichimoku)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n1(int): n1 low period.
        n2(int): n2 medium period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    conv = 0.5 * (high.rolling(n1, min_periods=0).max() + low.rolling(n1, min_periods=0).min())
    base = 0.5 * (high.rolling(n2, min_periods=0).max() + low.rolling(n2, min_periods=0).min())

    spana = 0.5 * (conv + base)

    if visual:
        spana = spana.shift(n2, fill_value=spana.mean())

    if fillna:
        spana = spana.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spana, name='ichimoku_a_' + str(n2))


def ichimoku_b(high, low, n2=26, n3=52, visual=False, fillna=False):
    """Ichimoku Kinkō Hyō (Ichimoku)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n2(int): n2 medium period.
        n3(int): n3 high period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    spanb = 0.5 * (high.rolling(n3, min_periods=0).max() + low.rolling(n3, min_periods=0).min())

    if visual:
        spanb = spanb.shift(n2, fill_value=spanb.mean())

    if fillna:
        spanb = spanb.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spanb, name='ichimoku_b_' + str(n2))


def aroon_up(close, n=25, fillna=False):
    """Aroon Indicator (AI)

    Identify when trends are likely to change direction (uptrend).

    Aroon Up - ((N - Days Since N-day High) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    aroon_up = close.rolling(n, min_periods=0).apply(lambda x: float(np.argmax(x) + 1) / n * 100, raw=True)
    if fillna:
        aroon_up = aroon_up.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(aroon_up, name='aroon_up' + str(n))


def aroon_down(close, n=25, fillna=False):
    """Aroon Indicator (AI)

    Identify when trends are likely to change direction (downtrend).

    Aroon Down - ((N - Days Since N-day Low) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    aroon_down = close.rolling(n, min_periods=0).apply(lambda x: float(np.argmin(x) + 1) / n * 100, raw=True)
    if fillna:
        aroon_down = aroon_down.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(aroon_down, name='aroon_down' + str(n))


def average_true_range(high, low, close, n=14, fillna=False):
    """Average True Range (ATR)

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)

    atr = np.zeros(len(close))
    atr[0] = tr[1::].mean()
    for i in range(1, len(atr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr.iloc[i]) / float(n)

    atr = pd.Series(data=atr, index=tr.index)

    if fillna:
        atr = atr.replace([np.inf, -np.inf], np.nan).fillna(0)

    return pd.Series(atr, name='atr')


def bollinger_mavg(close, n=20, fillna=False):
    """Bollinger Bands (BB)

    N-period simple moving average (MA).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n, min_periods=0).mean()
    if fillna:
        mavg = mavg.replace(
            [np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(mavg, name='mavg')


def bollinger_hband(close, n=20, ndev=2, fillna=False):
    """Bollinger Bands (BB)

    Upper band at K times an N-period standard deviation above the moving
    average (MA + Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n, min_periods=0).mean()
    mstd = close.rolling(n, min_periods=0).std()
    hband = mavg + ndev * mstd
    if fillna:
        hband = hband.replace(
            [np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(hband, name='hband')


def bollinger_lband(close, n=20, ndev=2, fillna=False):
    """Bollinger Bands (BB)

    Lower band at K times an N-period standard deviation below the moving
    average (MA − Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n, min_periods=0).mean()
    mstd = close.rolling(n, min_periods=0).std()
    lband = mavg - ndev * mstd
    if fillna:
        lband = lband.replace(
            [np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(lband, name='lband')


def bollinger_hband_indicator(close, n=20, ndev=2, fillna=False):
    """Bollinger High Band Indicator

    Returns 1, if close is higher than bollinger high band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev * mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    hband = df['hband']
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='bbihband')


def bollinger_lband_indicator(close, n=20, ndev=2, fillna=False):
    """Bollinger Low Band Indicator

    Returns 1, if close is lower than bollinger low band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev * mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    lband = df['lband']
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='bbilband')


def keltner_channel_central(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (central) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = (high + low + close) / 3.0
    tp = tp.rolling(n, min_periods=0).mean()
    if fillna:
        tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='kc_central')


def keltner_channel_hband(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (high) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((4 * high) - (2 * low) + close) / 3.0
    tp = tp.rolling(n, min_periods=0).mean()
    if fillna:
        tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='kc_hband')


def keltner_channel_lband(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (low) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((-2 * high) + (4 * low) + close) / 3.0
    tp = tp.rolling(n, min_periods=0).mean()
    if fillna:
        tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='kc_lband')


def keltner_channel_hband_indicator(high, low, close, n=10, fillna=False):
    """Keltner Channel High Band Indicator (KC)

    Returns 1, if close is higher than keltner high band channel. Else,
    return 0.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = ((4 * high) - (2 * low) + close) / 3.0
    df.loc[close > hband, 'hband'] = 1.0
    hband = df['hband']
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='kci_hband')


def keltner_channel_lband_indicator(high, low, close, n=10, fillna=False):
    """Keltner Channel Low Band Indicator (KC)

    Returns 1, if close is lower than keltner low band channel. Else, return 0.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = ((-2 * high) + (4 * low) + close) / 3.0
    df.loc[close < lband, 'lband'] = 1.0
    lband = df['lband']
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='kci_lband')


def donchian_channel_hband(close, n=20, fillna=False):
    """Donchian channel (DC)

    The upper band marks the highest price of an issue for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    hband = close.rolling(n, min_periods=0).max()
    if fillna:
        hband = hband.replace(
            [np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(hband, name='dchband')


def donchian_channel_lband(close, n=20, fillna=False):
    """Donchian channel (DC)

    The lower band marks the lowest price for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    lband = close.rolling(n, min_periods=0).min()
    if fillna:
        lband = lband.replace(
            [np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(lband, name='dclband')


def donchian_channel_hband_indicator(close, n=20, fillna=False):
    """Donchian High Band Indicator

    Returns 1, if close is higher than donchian high band channel. Else,
    return 0.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close >= hband, 'hband'] = 1.0
    hband = df['hband']
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='dcihband')


def donchian_channel_lband_indicator(close, n=20, fillna=False):
    """Donchian Low Band Indicator

    Returns 1, if close is lower than donchian low band channel. Else,
    return 0.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close <= lband, 'lband'] = 1.0
    lband = df['lband']
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='dcilband')


def acc_dist_index(high, low, close, volume, fillna=False):
    """Accumulation/Distribution Index (ADI)

    Acting as leading indicator of price movements.

    https://en.wikipedia.org/wiki/Accumulation/distribution_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0)  # float division by zero
    ad = clv * volume
    ad = ad + ad.shift(1, fill_value=ad.mean())
    if fillna:
        ad = ad.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ad, name='adi')


def on_balance_volume(close, volume, fillna=False):
    """On-balance volume (OBV)

    It relates price and volume in the stock market. OBV is based on a
    cumulative total volume.

    https://en.wikipedia.org/wiki/On-balance_volume

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = np.nan
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV'].cumsum()
    if fillna:
        obv = _fillna(obv)
    return pd.Series(obv, name='obv')


def chaikin_money_flow(high, low, close, volume, n=20, fillna=False):
    """Chaikin Money Flow (CMF)

    It measures the amount of Money Flow Volume over a specific period.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

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
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= volume
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / volume.rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cmf, name='cmf')


def force_index(close, volume, n=2, fillna=False):
    """Force Index (FI)

    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    fi = close.diff(n) * volume.diff(n)
    if fillna:
        fi = fi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(fi, name='fi_' + str(n))


def ease_of_movement(high, low, close, volume, n=20, fillna=False):
    """Ease of movement (EoM, EMV)

    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.

    https://en.wikipedia.org/wiki/Ease_of_movement

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
    emv = (high.diff(1) + low.diff(1)) * (high - low) / (2 * volume)
    emv = emv.rolling(n, min_periods=0).mean()
    if fillna:
        emv = emv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(emv, name='eom_' + str(n))


def volume_price_trend(close, volume, fillna=False):
    """Volume-price trend (VPT)

    Is based on a running cumulative volume that adds or substracts a multiple
    of the percentage change in share price trend and current volume, depending
    upon the investment's upward or downward movements.

    https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    vpt = volume * ((close - close.shift(1, fill_value=close.mean())) / close.shift(1, fill_value=close.mean()))
    vpt = vpt.shift(1, fill_value=vpt.mean()) + vpt
    if fillna:
        vpt = vpt.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vpt, name='vpt')


def negative_volume_index(close, volume, fillna=False):
    """Negative Volume Index (NVI)
    https://tulipindicators.org/nvi
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

    The Negative Volume Index (NVI) is a cumulative indicator that uses the
    change in volume to decide when the smart money is active. Paul Dysart
    first developed this indicator in the 1930s. [...] Dysart's Negative Volume
    Index works under the assumption that the smart money is active on days
    when volume decreases and the not-so-smart money is active on days when
    volume increases.

    The cumulative NVI line was unchanged when volume increased from one
    period to the other. In other words, nothing was done. Norman Fosback, of
    Stock Market Logic, adjusted the indicator by substituting the percentage
    price change for Net Advances.

    This implementation is the Fosback version.

    If today's volume is less than yesterday's volume then:
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
    https://en.wikipedia.org/wiki/Negative_volume_index
    """
    price_change = close.pct_change()
    vol_decrease = (volume.shift(1) > volume)

    nvi = pd.Series(data=np.nan, index=close.index, dtype='float64', name='nvi')

    nvi.iloc[0] = 1000
    for i in range(1, len(nvi)):
        if vol_decrease.iloc[i]:
            nvi.iloc[i] = nvi.iloc[i - 1] * (1.0 + price_change.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]

    if fillna:
        # IDEA: There shouldn't be any na; might be better to throw exception
        nvi = nvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

    return pd.Series(nvi, name='nvi')


# TODO
def put_call_ratio():
    # will need options volumes for this put/call ratio

    """Put/Call ratio (PCR)
    https://en.wikipedia.org/wiki/Put/call_ratio
    """
    # TODO
    return


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
