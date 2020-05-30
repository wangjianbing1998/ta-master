# coding=gbk

import argparse
import os

import pandas as pd
from tqdm import tqdm

from ta import add_all_ta
from ta import utils, add_trend_ta, add_momentum_ta, add_volatility_ta, add_volume_ta, add_my_ta, add_custom_ta


def solve_data(file_path):
    global df
    extend = file_path.split('.')[-1]
    if extend == 'xlsx':
        df = pd.read_excel(file_path)
    elif extend == 'csv':
        df = pd.read_csv(file_path)

    if args.limit_code:
        df = df[df['code'] == args.limit_code]
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
    os.makedirs(args.save_dir, exist_ok=True)

    if not (args.save_dir == 'None'):
        file = os.path.basename(file_path).split('.')[0]
        save_path = os.path.join(args.save_dir, file + '.xlsx')
        df.to_excel(save_path, index=False)

        print(f'saving minute_data into {save_path} successfully')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trade Indicators")
    parser.add_argument('-d', '--res_dir', type=str, default='test_data/', required=True,
                        help='the direction for reading processed minute_data.csv')
    parser.add_argument('-s', '--save_dir', type=str, default='result/', help='The result file path for saving')
    parser.add_argument('-i', '--indicator', type=str, default='all',
                        choices=['all', 'trend', 'momentum', 'my', 'volatility', 'volume', 'other'],
                        help='Which indicator(s) would you want to add? Please select in '
                             '[all,trend,momentum,my,volatility,volume,other].\n Note that the columns [Open,High,Low,Close] must be contained in minute_data')

    parser.add_argument('-lc', '--limit_code', type=str, default=None,
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

    main('start')
    # res = exec('my.sma(close,n=14)')
    # print(res)
