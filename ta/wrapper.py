# -*- coding: utf-8 -*-
from .momentum import *
from .my import *
from .others import *
from .trend import *
from .volatility import *
from .volume import *


def add_volume_ta(df, high=None, low=None, close=None, volume=None, fillna=False, colprefix=""):
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    if high and low and close and volume:
        df['{}volume_adi'.format(colprefix)] = acc_dist_index(df[high],
                                                              df[low],
                                                              df[close],
                                                              df[volume],
                                                              fillna=fillna)
        df['{}volume_obv'.format(colprefix)] = on_balance_volume(df[close],
                                                                 df[volume],
                                                                 fillna=fillna)
        df['{}volume_cmf'.format(colprefix)] = chaikin_money_flow(df[high],
                                                                  df[low],
                                                                  df[close],
                                                                  df[volume],
                                                                  fillna=fillna)
        df['{}volume_fi'.format(colprefix)] = force_index(df[close],
                                                          df[volume],
                                                          fillna=fillna)
        df['{}volume_em'.format(colprefix)] = ease_of_movement(df[high],
                                                               df[low],
                                                               df[close],
                                                               df[volume],
                                                               n=14,
                                                               fillna=fillna)
        df['{}volume_vpt'.format(colprefix)] = volume_price_trend(df[close],
                                                                  df[volume],
                                                                  fillna=fillna)
        df['{}volume_nvi'.format(colprefix)] = negative_volume_index(df[close],
                                                                     df[volume],
                                                                     fillna=fillna)
    return df


def add_volatility_ta(df, high=None, low=None, close=None, fillna=False, colprefix=""):
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    if close:
        df['{}volatility_bbh'.format(colprefix)] = bollinger_hband(df[close],
                                                                   n=20,
                                                                   ndev=2,
                                                                   fillna=fillna)

        df['{}volatility_bbl'.format(colprefix)] = bollinger_lband(df[close],
                                                                   n=20,
                                                                   ndev=2,
                                                                   fillna=fillna)
        df['{}volatility_bbm'.format(colprefix)] = bollinger_mavg(df[close],
                                                                  n=20,
                                                                  fillna=fillna)
        df['{}volatility_bbhi'.format(colprefix)] = bollinger_hband_indicator(
            df[close],
            n=20,
            ndev=2,
            fillna=fillna)
        df['{}volatility_bbli'.format(colprefix)] = bollinger_lband_indicator(
            df[close],
            n=20,
            ndev=2,
            fillna=fillna)
        df['{}volatility_dch'.format(colprefix)] = donchian_channel_hband(
            df[close],
            n=20,
            fillna=fillna)
        df['{}volatility_dcl'.format(colprefix)] = donchian_channel_lband(
            df[close],
            n=20,
            fillna=fillna)
        df['{}volatility_dchi'.format(colprefix)] = \
            donchian_channel_hband_indicator(df[close],
                                             n=20,
                                             fillna=fillna)

    if high and low and close:
        df['{}volatility_atr'.format(colprefix)] = average_true_range(
            df[high],
            df[low],
            df[close],
            n=14,
            fillna=fillna)

        df['{}volatility_kcc'.format(colprefix)] = keltner_channel_central(
            df[high],
            df[low],
            df[close],
            n=10,
            fillna=fillna)
        df['{}volatility_kch'.format(colprefix)] = keltner_channel_hband(
            df[high],
            df[low],
            df[close],
            n=10,
            fillna=fillna)
        df['{}volatility_kcl'.format(colprefix)] = keltner_channel_lband(
            df[high],
            df[low],
            df[close],
            n=10,
            fillna=fillna)
        df['{}volatility_kchi'.format(colprefix)] = \
            keltner_channel_hband_indicator(df[high],
                                            df[low],
                                            df[close],
                                            n=10,
                                            fillna=fillna)
        df['{}volatility_kcli'.format(colprefix)] = \
            keltner_channel_lband_indicator(df[high],
                                            df[low],
                                            df[close],
                                            n=10,
                                            fillna=fillna)

    return df


def add_trend_ta(df, high, low, close, fillna=False, colprefix=""):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    if close:
        df['{}trend_macd'.format(colprefix)] = macd(df[close],
                                                    n_fast=12,
                                                    n_slow=26,
                                                    fillna=fillna)
        df['{}trend_macd_signal'.format(colprefix)] = macd_signal(df[close],
                                                                  n_fast=12,
                                                                  n_slow=26,
                                                                  n_sign=9,
                                                                  fillna=fillna)
        df['{}trend_macd_diff'.format(colprefix)] = macd_diff(df[close],
                                                              n_fast=12,
                                                              n_slow=26,
                                                              n_sign=9,
                                                              fillna=fillna)
        df['{}trend_ema_fast'.format(colprefix)] = ema_indicator(df[close],
                                                                 n=12,
                                                                 fillna=fillna)
        df['{}trend_ema_slow'.format(colprefix)] = ema_indicator(df[close],
                                                                 n=26,
                                                                 fillna=fillna)
        df['{}trend_trix'.format(colprefix)] = trix(df[close], n=15, fillna=fillna)
        df['{}trend_dpo'.format(colprefix)] = dpo(df[close], n=20, fillna=fillna)
        df['{}trend_kst'.format(colprefix)] = kst(df[close], r1=10, r2=15, r3=20,
                                                  r4=30, n1=10, n2=10, n3=10,
                                                  n4=15, fillna=fillna)
        df['{}trend_kst_sig'.format(colprefix)] = kst_sig(df[close], r1=10, r2=15,
                                                          r3=20, r4=30, n1=10,
                                                          n2=10, n3=10, n4=15,
                                                          nsig=9, fillna=fillna)
        df['{}trend_kst_diff'.format(colprefix)] = (
                df['{}trend_kst'.format(colprefix)] -
                df['{}trend_kst_sig'.format(colprefix)])
        df['{}trend_aroon_up'.format(colprefix)] = aroon_up(df[close], n=25,
                                                            fillna=fillna)
        df['{}trend_aroon_down'.format(colprefix)] = aroon_down(df[close], n=25,
                                                                fillna=fillna)
        df['{}trend_aroon_ind'.format(colprefix)] = (
                df['{}trend_aroon_up'.format(colprefix)] -
                df['{}trend_aroon_down'.format(colprefix)]
        )

    if high and low and close:
        df['{}trend_adx'.format(colprefix)] = adx(df[high],
                                                  df[low],
                                                  df[close],
                                                  n=14,
                                                  fillna=fillna)
        df['{}trend_adx_pos'.format(colprefix)] = adx_pos(df[high],
                                                          df[low],
                                                          df[close],
                                                          n=14,
                                                          fillna=fillna)
        df['{}trend_adx_neg'.format(colprefix)] = adx_neg(df[high],
                                                          df[low],
                                                          df[close],
                                                          n=14,
                                                          fillna=fillna)
        df['{}trend_vortex_ind_pos'.format(colprefix)] = vortex_indicator_pos(
            df[high],
            df[low],
            df[close],
            n=14,
            fillna=fillna)
        df['{}trend_vortex_ind_neg'.format(colprefix)] = vortex_indicator_neg(
            df[high],
            df[low],
            df[close],
            n=14,
            fillna=fillna)
        df['{}trend_vortex_diff'.format(colprefix)] = abs(
            df['{}trend_vortex_ind_pos'.format(colprefix)] -
            df['{}trend_vortex_ind_neg'.format(colprefix)])
        df['{}trend_mass_index'.format(colprefix)] = mass_index(df[high],
                                                                df[low],
                                                                n=9,
                                                                n2=25,
                                                                fillna=fillna)
        df['{}trend_cci'.format(colprefix)] = cci(df[high],
                                                  df[low],
                                                  df[close],
                                                  n=20,
                                                  c=0.015,
                                                  fillna=fillna)

        df['{}trend_ichimoku_a'.format(colprefix)] = ichimoku_a(df[high], df[low],
                                                                n1=9, n2=26,
                                                                fillna=fillna)
        df['{}trend_ichimoku_b'.format(colprefix)] = ichimoku_b(df[high], df[low],
                                                                n2=26, n3=52,
                                                                fillna=fillna)
        df['{}trend_visual_ichimoku_a'.format(colprefix)] = ichimoku_a(
            df[high],
            df[low],
            n1=9,
            n2=26,
            visual=True,
            fillna=fillna)
        df['{}trend_visual_ichimoku_b'.format(colprefix)] = ichimoku_b(
            df[high],
            df[low],
            n2=26,
            n3=52,
            visual=True,
            fillna=fillna)

    return df


def add_momentum_ta(df, high=None, low=None, close=None, volume=None, fillna=False, colprefix=""):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    if close:
        df['{}momentum_rsi'.format(colprefix)] = rsi(df[close], n=14,
                                                     fillna=fillna)

        df['{}momentum_tsi'.format(colprefix)] = tsi(df[close], r=25, s=13,
                                                     fillna=fillna)
        df['{}momentum_kama'.format(colprefix)] = kama(df[close], fillna=fillna)

    if high and low and close and volume:
        df['{}momentum_mfi'.format(colprefix)] = money_flow_index(df[high],
                                                                  df[low],
                                                                  df[close],
                                                                  df[volume],
                                                                  n=14,
                                                                  fillna=fillna)

        df['{}momentum_uo'.format(colprefix)] = uo(df[high], df[low], df[close],
                                                   fillna=fillna)
        df['{}momentum_stoch'.format(colprefix)] = stoch(df[high], df[low],
                                                         df[close], fillna=fillna)
        df['{}momentum_stoch_signal'.format(colprefix)] = stoch_signal(
            df[high],
            df[low],
            df[close],
            fillna=fillna)
        df['{}momentum_wr'.format(colprefix)] = wr(df[high], df[low], df[close],
                                                   fillna=fillna)
        df['{}momentum_ao'.format(colprefix)] = ao(
            df[high], df[low], fillna=fillna)

    return df


def add_others_ta(df, close=None, fillna=False, colprefix=""):
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    if close:
        df['{}others_dr'.format(colprefix)] = daily_return(df[close],
                                                           fillna=fillna)
        df['{}others_dlr'.format(colprefix)] = daily_log_return(df[close],
                                                                fillna=fillna)
        df['{}others_cr'.format(colprefix)] = cumulative_return(df[close],
                                                                fillna=fillna)
    return df


def add_all_ta(df, open=None, high=None, low=None, close=None, volume=None, fillna=False,
               colprefix=""):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(df, high, low, close, volume, fillna=fillna,
                       colprefix=colprefix)
    df = add_volatility_ta(df, high, low, close, fillna=fillna,
                           colprefix=colprefix)
    df = add_trend_ta(df, high, low, close, fillna=fillna, colprefix=colprefix)
    df = add_momentum_ta(df, high, low, close, volume, fillna=fillna,
                         colprefix=colprefix)
    df = add_others_ta(df, close, fillna=fillna, colprefix=colprefix)
    df = add_my_ta(df, high, low, close, volume, fillna=fillna, colprefix=colprefix)

    return df


def add_my_ta(df, high=None, low=None, close=None, volume=None, fillna=False, colprefix=""):
    """Add my technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    df[colprefix + 'my_alma'] = alma(df[close], window=9, sigma=6, offset=0.85, fillna=fillna)
    df[colprefix + 'my_aroon_oscillator'] = aroon_oscillator(df[close], n=25, fillna=fillna)
    df[colprefix + 'my_atr'] = atr(df[close], n=1, period=10, fillna=fillna)
    df[colprefix + 'my_bollinger_band_middle'] = bollinger_band_middle(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_bollinger_band_upper'] = bollinger_band_upper(df[close], n=14, a=0.02, fillna=fillna)
    df[colprefix + 'my_chande_momentum_oscillator'] = chande_momentum_oscillator(df[close], n=10, fillna=fillna)
    df[colprefix + 'my_connors_rsi'] = connors_rsi(df[close], n1=3, n2=2, n3=100, fillna=fillna)
    df[colprefix + 'my_coppock_curve'] = coppock_curve(df[close], n1=10, n2=14, n3=11, fillna=fillna)
    df[colprefix + 'my_dema'] = dema(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_detrended_price_oscillator'] = detrended_price_oscillator(df[close], n=10, fillna=fillna)
    df[colprefix + 'my_directional_movement'] = directional_movement(df[close], )
    df[colprefix + 'my_divergence'] = divergence(df[close], df[high], df[low], period=9, fillna=fillna)
    df[colprefix + 'my_fisher_transform'] = fisher_transform(df[close], fillna=fillna)
    df[colprefix + 'my_hlb'] = hlb(df[close], fillna=fillna)
    df[colprefix + 'my_hma'] = hma(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_linear_regression'] = linear_regression(df[close], n=14, fillna=fillna)
    # df[colprefix + 'my_momentum'] = _momentum(df[close], n=10, fillna=fillna)
    df[colprefix + 'my_moving_average_convergence_divergence'] = moving_average_convergence_divergence(df[close],
                                                                                                       n1_shoter=12,
                                                                                                       n2_longer=26,
                                                                                                       fillna=fillna)
    df[colprefix + 'my_moving_count'] = moving_count(df[close], period=14, fillna=fillna)
    df[colprefix + 'my_ppo'] = ppo(df[close], n1=12, n2=26, fillna=fillna)
    df[colprefix + 'my_range_of_change'] = range_of_change(df[close], n=10, fillna=fillna)
    df[colprefix + 'my_range_of_change_ratio'] = range_of_change_ratio(df[close], n=10, fillna=fillna)
    df[colprefix + 'my_rate_of_change'] = rate_of_change(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_rmi'] = rmi(df[close], x=1, period=14, fillna=fillna)
    df[colprefix + 'my_sharp_ratio_atr'] = sharp_ratio_atr(df[close], period=14, fillna=fillna)
    df[colprefix + 'my_sharp_ratio_volatility'] = sharp_ratio_volatility(df[close], volatility=df[close], period=14,
                                                                         fillna=fillna)
    df[colprefix + 'my_signal_line'] = signal_line(df[close], n=9, n1=12, n2=26, fillna=fillna)
    df[colprefix + 'my_sma'] = sma(df[close], n=10, fillna=fillna)
    df[colprefix + 'my_smoothed_moving_average'] = smoothed_moving_average(df[close], n=30, shift=1, fillna=fillna)
    df[colprefix + 'my_spearman'] = spearman(df[close], period=9, fillna=fillna)
    df[colprefix + 'my_standard_deviation_channel'] = standard_deviation_channel(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_standard_error'] = standard_error(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_stochastic_rsi'] = stochastic_rsi(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_td_count'] = td_count(df[close], x=4, period=9, variation=1, fillna=fillna)
    df[colprefix + 'my_trend_stall'] = trend_stall(df[close], period=9, fillna=fillna)
    df[colprefix + 'my_triangular_moving_average'] = triangular_moving_average(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_trix'] = trix(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_variable_horizontal_filter'] = variable_horizontal_filter(df[close], n=14, fillna=fillna)
    df[colprefix + 'my_vidya'] = vidya(df[close], n=6, m=14, a=0.2, fillna=fillna)
    df[colprefix + 'my_wma'] = wma(df[close], n=10, fillna=fillna)
    df[colprefix + 'my_adxr'] = adxr(df[high], df[low], df[close], n=14, fillna=fillna)
    df[colprefix + 'my_atm_stochastic_oscillator'] = atm_stochastic_oscillator(df[high], df[low], df[close], n=14,
                                                                               fillna=fillna)
    df[colprefix + 'my_atm_stochastic_oscillator_signal'] = atm_stochastic_oscillator_signal(df[high], df[low],
                                                                                             df[close], n=14, d_n=3,
                                                                                             fillna=fillna)
    df[colprefix + 'my_atm_ultimate_oscillator'] = atm_ultimate_oscillator(df[high], df[low], df[close], s=7, m=14,
                                                                           len=28, ws=4.0, wm=2.0, wl=1.0,
                                                                           fillna=False)
    df[colprefix + 'my_average_true_range_percent'] = average_true_range_percent(df[high], df[low], df[close], n=14,
                                                                                 fillna=fillna)
    df[colprefix + 'my_bill_willams_alligator_jaw'] = bill_willams_alligator_jaw(df[high], df[low], oscillator='jaw',
                                                                                 n1=13, n2=8, fillna=fillna)
    df[colprefix + 'my_bollinger_band_lower'] = bollinger_band_lower(df[close], n=14, a=0.02, fillna=fillna)
    df[colprefix + 'my_chaikin_volatility_indicator'] = chaikin_volatility_indicator(df[high], df[low], n=10,
                                                                                     fillna=fillna)
    df[colprefix + 'my_chande_kroll_stop'] = chande_kroll_stop(df[high], df[low], df[close], stop_='short', n=9, x=1.5,
                                                               q=14, fillna=fillna)
    df[colprefix + 'my_chopping_index'] = chopping_index(df[high], df[low], df[close], n=10, fillna=fillna)
    df[colprefix + 'my_commodity_channel_indicator'] = commodity_channel_indicator(df[high], df[low], df[close], n=10,
                                                                                   c=.015, fillna=fillna)
    df[colprefix + 'my_donchian_channels'] = donchian_channels(df[high], df[low], n=14, fillna=fillna)
    df[colprefix + 'my_elder_ray_index'] = elder_ray_index(df[high], n=13, fillna=fillna)
    df[colprefix + 'my_eom'] = eom(df[high], df[low], df[volume], c=0.0001, fillna=fillna)
    df[colprefix + 'my_faster_stochastic_oscillator'] = faster_stochastic_oscillator(df[high], df[low], df[close],
                                                                                     type='k', n1=14, n2=3,
                                                                                     fillna=fillna)
    df[colprefix + 'my_intreday_intensity_index'] = intreday_intensity_index(df[high], df[low], df[close], df[volume],
                                                                             fillna=fillna)
    df[colprefix + 'my_intreday_intensity_percent'] = intreday_intensity_percent(df[high], df[low], df[close],
                                                                                 df[volume], n=21, fillna=fillna)
    df[colprefix + 'my_market_facilitation_index'] = market_facilitation_index(df[high], df[low], df[volume],
                                                                               fillna=fillna)
    df[colprefix + 'my_pivot_points_high'] = pivot_points_high(df[high], n=5, fillna=fillna)
    df[colprefix + 'my_pivot_points_low'] = pivot_points_low(df[low], n=5, fillna=fillna)
    df[colprefix + 'my_pvi'] = pvi(df[close], df[volume], fillna=fillna)
    df[colprefix + 'my_pvt'] = pvt(df[close], df[volume], fillna=fillna)
    df[colprefix + 'my_rmi_expansion'] = rmi_expansion(df[close], df[high], df[low], period=14, n=14, variation=2,
                                                       fillna=fillna)
    df[colprefix + 'my_stochastic_oscillator'] = stochastic_oscillator(df[high], df[low], df[close], n=3, fillna=fillna)
    df[colprefix + 'my_stochastic_oscillator_d'] = stochastic_oscillator_d(df[high], df[low], df[close], n_k=14, n_d=3,
                                                                           fillna=fillna)
    df[colprefix + 'my_stochastic_oscillator_k'] = stochastic_oscillator_k(df[high], df[low], df[close], n=14,
                                                                           fillna=fillna)
    df[colprefix + 'my_typical_price'] = typical_price(df[high], df[low], df[close], fillna=fillna)
    df[colprefix + 'my_ultimate_oscillator'] = ultimate_oscillator(df[high], df[low], df[close], n=6, m=10, p=14,
                                                                   fillna=fillna)
    df[colprefix + 'my_volatility_ratio'] = volatility_ratio(df[high], df[low], df[close], fillna=fillna)
    df[colprefix + 'my_volume_ema'] = volume_ema(df[volume], n=14, fillna=fillna)
    df[colprefix + 'my_volume_wma'] = volume_wma(df[volume], n=14, fillna=fillna)
    df[colprefix + 'my_weighted_close'] = weighted_close(df[high], df[low], df[close], fillna=fillna)

    return df


def add_custom_ta(df, indicator_fn_args, indicator_name=None):
    '''
    Add my custom analysis features to dataframe.
    :param df:
    :param indicator_name: the output column name for the inserted indicator
    :param indicator: the function that will be called to generate the indicator you want
    :param args: any args
    :param kwargs: any kwargs
    :return: dataframe that contained the inserted indicator
    '''
    indicator_name = indicator_name or indicator_fn_args.split('.')[-1].split('(')[0]
    package_name = indicator_fn_args.split('.')[0]
    function_name = indicator_fn_args.split('.')[1].split('(')[0]
    df_args = indicator_fn_args.split('.')[1][:-1].split('(')[1].split(',')
    for index in range(len(df_args)):
        if df_args[index] in ['close', 'high', 'low', 'open']:
            run = f'{df_args[index]}=df[df_args[index]]'
            exec(run)

    run_ = f'{function_name}({indicator_fn_args.split(".")[1][:-1].split("(")[1]})'
    print(run_)
    exec(f'from .{package_name} import {function_name}')
    # exec(f'df[indicator_name]={run_}')
    df[indicator_name] = eval(run_)
    return df
