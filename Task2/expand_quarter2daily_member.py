# coding=gbk
from datetime import datetime

import pandas as pd
from tqdm import tqdm

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)


def expand_q2d_one(data):
    data.rename(columns=dict([(date, pd.to_datetime(date)) for date in data.columns.to_list()]), inplace=True)
    dates = data.columns.to_list()
    for index, quarter in enumerate(dates):

        for date in pd.date_range(quarter, dates[index + 1] if index + 1 < len(dates) else datetime.today(), freq='D'):
            if date.weekday() not in [0, 6]:
                data[date] = data[quarter]
    data = data.T
    data.sort_index(inplace=True)
    data = data.T

    return data


data_path = 'ALL Member Histories.xlsx'
datas = pd.read_excel(data_path, None)
res_ = 'ALL Member Daily Histories.xlsx'
write = pd.ExcelWriter(res_)

for sheet in tqdm(datas):
    data = expand_q2d_one(datas[sheet])
    data.to_excel(write, sheet_name=sheet, index=False)

write.save()
