# coding=gbk
from collections import defaultdict

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)


def get_n_data(n):
    return mask_df.rolling(n).mean().iloc[n - 1, :]


def get_N_data(ns):
    res = defaultdict(float)
    df_sum = sum([get_n_data(n) for n in ns])
    for index, row in df_sum.iteritems():
        left, right = index.split('/')
        left = left.strip()
        right = right.strip()

        res[left] += row
        res[right] += len(ns) - row

    return pd.DataFrame(res, index=['res']).T


if __name__ == '__main__':
    df = pd.read_excel('df.xlsx', index_col=[0])
    data = df.T
    data.index.name = 'date'

    mask_df = data.copy()
    mask_df[data > data.shift(-4)] = 1
    mask_df[data <= data.shift(-4)] = 0

    df_ns = get_N_data([13, 22, 33])
    df_ns.to_excel('res.xlsx')
