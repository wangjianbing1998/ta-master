# -*- coding: utf-8 -*-

import importlib

from .momentum import *
from .my import *
from .others import *
from .trend import *
from .volatility import *
from .volume import *


def add_volume_ta(df, high, low, close, volume, fillna=False, colprefix=""):
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


def add_volatility_ta(df, high, low, close, fillna=False, colprefix=""):
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
    df['{}volatility_atr'.format(colprefix)] = average_true_range(
        df[high],
        df[low],
        df[close],
        n=14,
        fillna=fillna)
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
    df['{}volatility_dcli'.format(colprefix)] = \
        donchian_channel_lband_indicator(df[close],
                                         n=20,
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
    df['{}trend_trix'.format(colprefix)] = trix(df[close], n=15, fillna=fillna)
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
    df['{}trend_aroon_up'.format(colprefix)] = aroon_up(df[close], n=25,
                                                        fillna=fillna)
    df['{}trend_aroon_down'.format(colprefix)] = aroon_down(df[close], n=25,
                                                            fillna=fillna)
    df['{}trend_aroon_ind'.format(colprefix)] = (
            df['{}trend_aroon_up'.format(colprefix)] -
            df['{}trend_aroon_down'.format(colprefix)]
    )
    return df


def add_momentum_ta(df, high, low, close, volume, fillna=False, colprefix=""):
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
    df['{}momentum_rsi'.format(colprefix)] = rsi(df[close], n=14,
                                                 fillna=fillna)
    df['{}momentum_mfi'.format(colprefix)] = money_flow_index(df[high],
                                                              df[low],
                                                              df[close],
                                                              df[volume],
                                                              n=14,
                                                              fillna=fillna)
    df['{}momentum_tsi'.format(colprefix)] = tsi(df[close], r=25, s=13,
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
    df['{}momentum_kama'.format(colprefix)] = kama(df[close], fillna=fillna)
    return df


def add_others_ta(df, close, fillna=False, colprefix=""):
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['{}others_dr'.format(colprefix)] = daily_return(df[close],
                                                       fillna=fillna)
    df['{}others_dlr'.format(colprefix)] = daily_log_return(df[close],
                                                            fillna=fillna)
    df['{}others_cr'.format(colprefix)] = cumulative_return(df[close],
                                                            fillna=fillna)
    return df


def add_all_ta(df, open, high, low, close, volume, fillna=False,
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
    return df


def add_my_ta(df, high, low, close, volume, fillna=False, colprefix=""):
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

    df['{}_sma'.format(colprefix)] = sma(df[close], n=14,
                                         fillna=fillna)
    df['{}_nvi'.format(colprefix)] = negative_volume_index(
        df[close],
        df[volume],
        fillna=fillna)
    df['{}_moving_average_convergence_divergence'.format(colprefix)] = moving_average_convergence_divergence(df[close],
                                                                                                             n1_shoter=12,
                                                                                                             n2_longer=26,
                                                                                                             fillna=fillna)
    df['{}_signal_line'.format(colprefix)] = signal_line(df[close], n=9, n1=12, n2=26, fillna=fillna)
    '''
    . . .
    您只需要在这里将新添加的指标依葫芦画瓢的放进来就OK了
    . . .
    '''
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
            run=f'{df_args[index]}=df[df_args[index]]'
            exec(run)

    run_ = f'{function_name}({indicator_fn_args.split(".")[1][:-1].split("(")[1]})'
    print(run_)
    exec(f'from .{package_name} import {function_name}')
    # exec(f'df[indicator_name]={run_}')
    df[indicator_name]=eval(run_)
    return df
