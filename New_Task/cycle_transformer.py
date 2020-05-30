# coding=gbk
import datetime
import os

import better_exceptions

better_exceptions.hook()

import pandas as pd
import numpy as np

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)


# https://blog.csdn.net/ramoslin304/article/details/94025179

def transform_cycle_rule(data, rule_cycle='5T'):
    cycle_df = pd.DataFrame()
    cycle_df['close'] = data['close'].resample(rule=rule_cycle).last()
    cycle_df['open'] = data['open'].resample(rule=rule_cycle).first()
    cycle_df['high'] = data['high'].resample(rule=rule_cycle).max()
    cycle_df['low'] = data['low'].resample(rule=rule_cycle).min()
    cycle_df['volume'] = data['volume'].resample(rule=rule_cycle).sum()

    return cycle_df


def read_txt(txt_path=r'F:\firstRate_data_upload\sp500_2020_all_tickers_d4hr4vi\A_2020_2020.txt'):
    data = np.loadtxt(txt_path, dtype=str, delimiter=',')
    data = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    data['date'] = pd.to_datetime(data['date'])
    data[['open', 'high', 'low', 'close', 'volume']] = data[
        ['open', 'high', 'low', 'close', 'volume']].astype('float')

    return data


def main(data_dir='BigData', result_dir='processed_transformer_dir', num_per_xlsx=2):
    os.makedirs(result_dir, exist_ok=True)

    res = pd.DataFrame()
    nums = 0
    counts = 0
    for file in os.listdir(data_dir):
        code = os.path.basename(file).split('.')[0]
        data = process_data(os.path.join(data_dir, file))
        data['code'] = code
        data.reset_index(drop=False, inplace=True)
        data = data[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
        nums += 1
        if len(res) == 0:
            res = data
        else:
            res = pd.concat([res, data])

        if nums == num_per_xlsx:
            res_code = '-'.join(res['code'].value_counts().keys().to_list())
            res.to_excel(os.path.join(result_dir, f'{res_code}.xlsx'), index=False)
            counts += 1
            nums = 0
            res = pd.DataFrame()

    res_code = '-'.join(res['code'].value_counts().keys().to_list())
    res.to_excel(os.path.join(result_dir, f'{res_code}.xlsx'), index=False)


def T30_1H(data):
    data.drop(data[(data.index.hour < 9) | (data.index.hour >= 16)].index, inplace=True)

    T30 = transform_cycle_rule(data, rule_cycle='30T')
    T30_1 = T30.shift(-1)

    d = transform_cycle_rule(T30_1, '1H')
    d['date'] = d.index
    d['date'] = d['date'] + datetime.timedelta(minutes=30)
    d.reset_index(drop=True, inplace=True)
    d.set_index('date', inplace=True)
    d.drop(d[(d.index.hour < 9) | (d.index.hour >= 16)].index, inplace=True)

    return d


def process_data(data_path='A_2020_2020.txt'):
    data = read_txt(data_path)
    data.set_index('date', inplace=True)

    data = T30_1H(data)
    return data


if __name__ == '__main__':
    main()
