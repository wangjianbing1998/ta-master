# coding=gbk
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import argparse
import os

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)

from tqdm import tqdm

from ta import add_all_ta, plt_show
from ta import utils, add_trend_ta, add_momentum_ta, add_volatility_ta, add_volume_ta, add_my_ta, add_custom_ta


def combinate_data(dir_path='Task1/BigData2/'):
    res = None
    for file in os.listdir(dir_path):
        df = pd.read_csv(os.path.join(dir_path, file))
        df['code'] = file.split('.')[0]
        df = df[['code', *(df.columns.to_list()[:-1])]]

        if res is None:
            res = df
        else:
            res = pd.concat([res, df], axis=0)

    res.to_csv('data.csv', index=False)


# combinate_data()

def solve_data(file_path):
    extend = file_path.split('.')[-1]
    df = None
    if extend == 'xlsx':
        df = pd.read_excel(file_path)
    elif extend == 'csv':
        df = pd.read_csv(file_path)

    if args.limit_code:
        df = df[df['code'] == args.limit_code]

    df = process_one(df)
    file = os.path.basename(file_path).split('.')[0]

    plt_finplot_show(df, args.limit_code if args.limit_code else file)
    os.makedirs(args.save_dir, exist_ok=True)

    if not (args.save_dir == 'None'):
        save_path = os.path.join(args.save_dir, file + '.xlsx')
        df.to_excel(save_path, index=False)

        print(f'saving minute_data into {save_path} successfully')


def process_one(df):
    df = utils.dropna(df)
    if args.custom_indicator_fn is not None:
        if args.custom_indicator is not None:
            raise ValueError('custom indicator fn and custom indicator only can use one')

        # print(args.custom_indicator_fn)
        df = add_custom_ta(df, args.custom_indicator_fn + '(' + args.custom_indicator_args + ')')

    elif args.custom_indicator is not None:
        df = add_custom_ta(df, args.custom_indicator)

    elif args.indicator == 'all':
        df = add_all_ta(df, "open", "high", "low", "close", 'volume', fillna=True, colprefix=args.column_prefix)

    elif args.indicator == 'trend':
        df = add_trend_ta(df, "high", "low", "close", fillna=True, colprefix=args.column_prefix)

    elif args.indicator == 'momentum':
        df = add_momentum_ta(df, "high", "low", "close", 'volume', fillna=True, colprefix=args.column_prefix)

    elif args.indicator == 'my':
        df = add_my_ta(df, "high", "low", "close", 'volume', fillna=True, colprefix=args.column_prefix)

    elif args.indicator == 'volatility':
        df = add_volatility_ta(df, "high", "low", "close", fillna=True, colprefix=args.column_prefix)

    elif args.indicator == 'volume':
        df = add_volume_ta(df, "high", "low", "close", 'volume', fillna=True, colprefix=args.column_prefix)

    else:
        raise ValueError('Please use a calculation')

    return df


def main(_):
    print(f'making dir {os.path.dirname(args.save_dir)} !')

    find_csv = False
    find_xlsx = False
    for file in tqdm(os.listdir(args.data_dir)):
        if file.endswith('.csv'):
            find_csv = True
            print(f'solving csv file {file} . . .')
            solve_data(os.sep.join([args.data_dir, file]))
        elif file.endswith('.xlsx'):
            find_xlsx = True
            print(f'solving xlsx file {file} . . .')
            solve_data(os.sep.join([args.data_dir, file]))
    if not find_csv and not find_xlsx:
        print(f'*.csv or *.xlsx must be contained in the {args.data_dir}')


def check_indicator(check_indicators):
    checked_indicators_dir = 'checked_indicators/'
    # if os.path.exists(checked_indicators_dir):
    #     shutil.rmtree(checked_indicators_dir)
    os.makedirs(checked_indicators_dir, exist_ok=True)
    for c in check_indicators:
        c = 'my_' + c

        path = os.path.join(checked_indicators_dir, c)
        series = df.loc['2017':'2018', c]

        plt_show(series, title=path)


def check_indicators_():
    df = get_CSCO_PX_REL_VWAP_data()
    check_indicator(['rmi',
                     'rmi_expansion',
                     'spearman',
                     'fisher_transform',
                     'td_count',
                     'trend_stall',
                     'atm_stochastic_oscillator_signal',
                     'atm_stochastic_oscillator',
                     'atm_ultimate_oscillator',
                     'rate_of_change',
                     'atr',
                     'sharp_ratio_volatility',
                     'sharp_ratio_atr',
                     'divergence',
                     'moving_count'
                     ])
    exclude_indicators = [
        'my_alma',
        'my_chaikin_volatility_indicator',
        'my_chande_momentum_oscillator',
        'my_directional_movement',
        'my_intreday_intensity_index',
        'my_eom',
        'my_intreday_intensity_percent',
        'my_market_facilitation_index',
        'my_volatility_ratio',
    ]
    # years = ['2015', '2016', '2017', '2018', '2019', '2020']
    # for year_index, year in enumerate(years):
    #     if year_index > 0:
    #         for col in df.columns.to_list():
    #             if col not in exclude_indicators:
    #                 print(col)
    #                 path = f'indicators_result_jpgs/{years[year_index - 1]}-{year}/{col}'
    #                 os.makedirs(os.path.dirname(path), exist_ok=True)
    #                 plt_show(df.loc[years[year_index - 1]:year, col],
    #                          title=path)


def get_CSCO_PX_REL_VWAP_data():
    df = pd.read_csv('data/input CSCO PX_REL_VWAP.txt', delimiter='\t', index_col=0)
    df.rename(columns={'Cisco Systems Inc / SPDR S&P 500 ETF Trust': 'close'}, inplace=True)
    df = utils.dropna(df)
    df = df.reindex(index=df.index[::-1])
    df['open'] = df['close']
    df['high'] = df['close']
    df['low'] = df['close']
    df['volume'] = df['close']
    df = add_all_ta(df, open='open', high='high', low='low', close='close', volume='volume', fillna=True,
                    colprefix=args.column_prefix)
    # df=add_custom_ta(df, 'all.linear_regression(close,n=14)')
    df.fillna(0, inplace=True)
    return df


def plt_finplot_show(df, symbol):
    import finplot as fplt
    df.reset_index(drop=True, inplace=True)

    df = df.astype({'date': 'datetime64[ns]'})
    ax, ax2 = fplt.create_plot(symbol, rows=2)
    # plot candle sticks
    candle_src = fplt.PandasDataSource(df[['date', 'open', 'close', 'high', 'low']])
    fplt.candlestick_ochl(candle_src, ax=ax)

    # put an MA in there
    fplt.plot(df['date'], df['close'].rolling(25).mean(), ax=ax, color='#0000ff', legend='ma-25')

    # place some dumb markers
    hi_wicks = df['high'] - df[['open', 'close']].T.max().T
    df.loc[(hi_wicks > hi_wicks.quantile(0.99)), 'marker'] = df['close']
    fplt.plot(df['date'], df['marker'], ax=ax, color='#000000', style='^', legend='dumb mark')

    # draw some random crap on our second plot
    df['rnd'] = np.random.normal(size=len(df))
    fplt.plot(df['date'], df['rnd'], ax=ax2, color='#992277', legend='stuff')
    fplt.set_y_range(ax2, -1.4, +1.7)  # fix y-axis range

    fplt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trade Indicators")

    parser.add_argument('--data_dir', type=str, default='Task1/BigData2/', required=False,
                        help='the direction for reading processed minute_data.csv')
    parser.add_argument('-d', '--res_dir', type=str, default='test_data/', required=False,
                        help='the direction for reading processed minute_data.csv')
    parser.add_argument('-s', '--save_dir', type=str, default='result/', help='The result file path for saving')
    parser.add_argument('-i', '--indicator', type=str, default='all',
                        choices=['all', 'trend', 'momentum', 'my', 'volatility', 'volume', 'other'],
                        help='Which indicator(s) would you want to add? Please select in '
                             '[all,trend,momentum,my,volatility,volume,other].\n Note that the columns [Open,High,Low,Close] must be contained in minute_data')

    parser.add_argument('-lc', '--limit_code', type=str, default='0820143D',
                        help='limited code to solve, once specified, it will be calculated only in the result')
    parser.add_argument('-cp', '--column_prefix', type=str, default='', help='the prefix column names inserted')

    parser.add_argument('-cif', '--custom_indicator_fn', type=str, default=None,
                        help='the name of custom indicator inserted')  # my.sma
    parser.add_argument('-cia', '--custom_indicator_args', type=str, default='close,n=14',
                        help='the name of custom indicator inserted')
    # parser.add_argument('-ci', '--custom_indicator', type=str, default='my.sma(close,n=14)',
    parser.add_argument('-ci', '--custom_indicator', type=str, default=None,
                        help='the name and args of custom indicator inserted, equals to use the custom_indicator_fn and custom_indicator_args')
    args = parser.parse_args()

    # main('start')

    # check_indicators_()

    # df = pd.read_csv('data.csv')
    # code_ = 'AAPL_2020_2020'
    # df = df[df['code'] == code_]
    # df.set_index('date', inplace=True)
    # df = process_one(df)
    # df.reset_index(inplace=True)
    # df.to_csv('processed_data.csv', index=False)
    #
    # plt_finplot_show(df, code_)
