# coding=gbk
import os
import warnings

import matplotlib.pyplot as plt

from cycle_transformer import read_txt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
warnings.simplefilter('ignore')
import better_exceptions
import pandas as pd
from pandas import Series

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)
better_exceptions.hook()


def remove_max_min_by_IRQ(data: Series, threshold_ratio=1.):
    down_4 = data.quantile(0.25)
    up_4 = data.quantile(0.75)
    IRQ = up_4 - down_4
    down_value = down_4 - threshold_ratio * IRQ
    up_value = up_4 + threshold_ratio * IRQ

    data[down_value > data] = down_value
    data[up_value < data] = up_value
    return data


def remove_max_min_by_n(data: Series, n=5, threshold_ratio=1.5):
    data_average = data.rolling(n).mean()
    data_medium = data.rolling(n).median()
    data_max = data.rolling(n).quantile(0.75)
    data_min = data.rolling(n).quantile(0.25)

    max_min = data_max - data_min
    down_value = data_medium - max_min * threshold_ratio
    up_value = data_medium + max_min * threshold_ratio

    down_value.fillna(method='backfill', inplace=True)
    up_value.fillna(method='backfill', inplace=True)

    data[down_value > data] = down_value
    data[up_value < data] = up_value
    return data


def remove_max_min_by_pct_change(data: pd.DataFrame, n=10, pct_n=200, threshold_ratio=1.2, inplace=False):
    if not inplace:
        import copy
        data = copy.deepcopy(data)

    def process_one_column(data_series: Series):
        data_medium = data_series.rolling(n).median()
        data_max = data_series.rolling(n).quantile(0.75)
        data_min = data_series.rolling(n).quantile(0.25)

        max_min = data_max - data_min
        up_value = data_medium + max_min * threshold_ratio

        data_series, threshold_pct = pct_change_smoothing(data_series, up_value)

        data_series, threshold_pct = pct_change_smoothing(data_series, up_value)

        return data_series

    def pct_change_smoothing(data_series, up_value):
        threshold_pct = 4 * data_series.pct_change().rolling(pct_n).std()
        data_series[threshold_pct < data_series.pct_change()] = up_value
        return data_series, threshold_pct

    new_data = pd.DataFrame()
    for c in data.columns:
        new_data[c] = process_one_column(data[c])
    return new_data


def plt_data(data, title='title', fig_sort=221):
    plt.subplot(fig_sort)
    plt.plot(data)
    # plt.axis('off')
    plt.title(title)
    plt.xticks(rotation=45)


def do_outliner_remove(source_dir, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    for file in os.listdir(source_dir):
        df = read_txt(os.path.join(source_dir, file))
        df.set_index('date', inplace=True)
        df.dropna(inplace=True)
        df.drop(df[(df.index.hour < 9) | (df.index.hour >= 16)].index, inplace=True)
        change_df = remove_max_min_by_pct_change(df, 20, 200, 1.2)

        file_replace = file.replace('.txt', '.csv')
        change_df.to_csv(os.path.join(res_dir, file_replace))

        print(f'saving {file_replace} successfully')


def get_df(code_path=r'BigData/AAPL_2020_2020.txt'):
    df = read_txt(code_path)
    df.set_index('date', inplace=True)
    df.dropna(inplace=True)
    df.drop(df[(df.index.hour < 9) | (df.index.hour >= 16)].index, inplace=True)

    return df


def test_outliner_remover(date, code_path=r'BigData/AAPL_2020_2020.txt'):
    df = read_txt(code_path)
    df.set_index('date', inplace=True)
    df.dropna(inplace=True)
    df.drop(df[(df.index.hour < 9) | (df.index.hour >= 16)].index, inplace=True)
    df = df.loc[(df.index.month >= date['month']) & (df.index.month <= date['month'])
                & (df.index.day >= date['day']) & (df.index.day <= date['day'])]
    fig = plt.figure(figsize=(24, 8))
    plt_data(df['open'], 'original_open', 241)
    plt_data(df['high'], 'original_high', 242)
    plt_data(df['low'], 'original_low', 243)
    plt_data(df['close'], 'original_close', 244)
    change_df = remove_max_min_by_pct_change(df, 20, 200, 1.2)
    plt_data(change_df['open'], 'open', 245)
    plt_data(change_df['high'], 'high', 246)
    plt_data(change_df['low'], 'low', 247)
    plt_data(change_df['close'], 'close', 248)
    plt.savefig(f'{date["year"]}-{date["month"]}-{date["day"]}.png')
    plt.title(f'{date["year"]}-{date["month"]}-{date["day"]}')
    plt.show()


if __name__ == '__main__':
    # test_outliner_remover({'year': 2020,
    #                        'month': 3,
    #                        'day': 24})
    do_outliner_remove('BigData/', 'BigData2/')
