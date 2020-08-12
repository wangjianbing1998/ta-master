# coding=gbk
import os

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)


def get_one_column_data(data_path, column, start_date='1995/1/1', unit='D'):
    data = pd.read_excel(data_path, header=None, skiprows=1)
    columns_to_list = data.columns.to_list()
    columns = [columns_to_list[0]] + columns_to_list[2:]
    data = data[columns]
    data.rename(columns={columns[0]: 'code'}, inplace=True)
    data.set_index('code', inplace=True)
    date_columns = [pd.to_datetime(start_date) + pd.to_timedelta(index_date - 2, unit=unit) for index_date in
                    data.columns.to_list()]
    res_data = None
    for code, row in data.iterrows():
        d = pd.DataFrame({column: [row[index_date] for index_date in data.columns.to_list()],
                          'date': [date for date in date_columns]})
        d['code'] = code

        if res_data is not None:
            res_data = pd.concat([res_data, d])
        else:
            res_data = d
    print(f'finished read {os.path.basename(data_path)}')
    return res_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Trade Indicators")
    parser.add_argument('-d', '--res_dir', type=str, default='raw_data_set/weekly/', required=True,
                        help='the direction for reading trade minute_data.xlsx')
    parser.add_argument('-p', '--prefix', type=str, default='weekly',
                        help='the predix for reading trade {prefix}_open.xlsx')
    parser.add_argument('-s', '--start_date', type=str, default='2006/7/7',
                        help='the start date of minute_data column input')
    parser.add_argument('-u', '--unit', type=str, default='W',
                        help='the unit of minute_data column input')
    parser.add_argument('-r', '--result_dir', type=str, default='processed_data/data_week.csv',
                        help='the output direction for writing minute_data.xlsx')

    args = parser.parse_args()

    open_data = get_one_column_data(f'{args.data_dir}/{args.prefix}_open.xlsx', 'open', start_date=args.start_date,
                                    unit=args.unit)
    low_data = get_one_column_data(f'{args.data_dir}/{args.prefix}_low.xlsx', 'low', start_date=args.start_date,
                                   unit=args.unit)
    high_data = get_one_column_data(f'{args.data_dir}/{args.prefix}_high.xlsx', 'high', start_date=args.start_date,
                                    unit=args.unit)
    close_data = get_one_column_data(f'{args.data_dir}/{args.prefix}_close.xlsx', 'close', start_date=args.start_date,
                                     unit=args.unit)
    print(f'open={len(open_data)},  low={len(low_data)},  high={len(high_data)},  close={len(close_data)}')

    data = pd.merge(open_data, high_data, on=['code', 'date'])
    data = pd.merge(data, low_data, on=['code', 'date'])
    data = pd.merge(data, close_data, on=['code', 'date'])

    data = data[['code', 'date', 'open', 'high', 'low', 'close']]
    os.makedirs(os.path.dirname(args.result_dir), exist_ok=True)
    data.to_csv(f'{args.result_dir}', index=False)
