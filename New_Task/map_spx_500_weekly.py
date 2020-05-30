# coding=gbk
import os
from collections import defaultdict

import better_exceptions

better_exceptions.hook()

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)

members = pd.read_excel('raw_data_set/weekly/members.xlsx', skiprows=1)
weekly_data = pd.read_csv('processed_data/weekly/data_weekly.csv')
weekly_code = weekly_data['code'].unique()

members = members[members.columns.to_list()[1:]]
members = members.apply(lambda x: x.apply(lambda y: str(y).split()[0]))


def find_difference(date):
    print(date)
    if type(date) is str:
        date = pd.to_datetime(date)
    mem = members[date].unique()
    mem_not_in_weekly = [code for code in mem if not code in weekly_code]
    weekly_not_in_mem = [code for code in weekly_code if not code in mem]
    print(len(mem), len(mem_not_in_weekly), len(mem) - len(mem_not_in_weekly))
    return mem_not_in_weekly, weekly_not_in_mem


def get_quarter_data(quarter, next_querter) -> pd.DataFrame:
    series = members[quarter]
    res = None
    datetime = pd.to_datetime(weekly_data['date'])
    weekly_close = weekly_data[(quarter <= datetime) & (datetime < next_querter)]
    for code in series.values:
        if code in weekly_code:
            d = weekly_close[(weekly_close['code'] == code)][['code', 'date', 'open', 'high', 'low', 'close']]
            d['C1/C0-1'] = d['close'].shift(-1) / d['close'] - 1
            d['L1/C0-1'] = d['low'].shift(-1) / d['close'] - 1
            d['Average'] = (d['C1/C0-1'] + d['L1/C0-1']) / 2

            if res is not None:
                res = pd.concat([res, d])
            else:
                res = d

    return res


def get_unique_code(data: pd.DataFrame) -> list:
    code = list()

    for index, row in data.iterrows():
        for c in data.columns:
            if row[c] != 'nan':
                code.append(row[c])

    return list(set(code))


def members_SPX(res_dir):
    os.makedirs(res_dir, exist_ok=True)
    quarter = members.columns

    res = {}
    for index, q in enumerate(quarter):

        def get_quarter_code(quarter, next_querter):
            res = []
            series = members[quarter]
            # datetime = pd.to_datetime(weekly_data['date'])
            # weekly_close = weekly_data[(quarter <= datetime) & (datetime < next_querter)]
            for code in series.values:
                # if code in weekly_close['code'].values:
                res.append(code)

            return res

        data = get_quarter_code(q, quarter[index + 1] if index + 1 < len(quarter) else pd.datetime.today())
        res[q] = data

    max_length = max(len(res[q]) for q in res)
    for q in res.keys():
        res[q] += [None] * (max_length - len(res[q]))

    df = pd.DataFrame(res)

    columns = defaultdict(list)
    for col_index, q in enumerate(df.columns):
        if q < pd.to_datetime('2000-1-1'):
            columns['1995-1999'].append(q)
        elif q < pd.to_datetime('2005-1-1'):
            columns['2000-2004'].append(q)
        elif q < pd.to_datetime('2010-1-1'):
            columns['2005-2009'].append(q)
        elif q < pd.to_datetime('2015-1-1'):
            columns['2010-2014'].append(q)
        else:
            columns['2015-2020'].append(q)

    # from openpyxl import load_workbook
    # excel_path = os.path.join(res_dir, 'members_600+_SPX_result.xlsx')
    #
    # excelWriter = pd.ExcelWriter(excel_path,
    #                              engine='openpyxl')
    # pd.DataFrame().to_excel(excel_path)
    #
    #
    # book = load_workbook(excelWriter.path)
    # excelWriter.book = book
    # for col in columns:
    #     data = df[columns[col]]
    #     data.to_excel(excel_writer=excelWriter, sheet_name=col, index=False)
    # excelWriter.close()

    new_data = {}
    for col in columns:
        data = df[columns[col]]
        unique_code = get_unique_code(data)
        unique_code.sort()
        new_data[col] = unique_code

    max_length = max(len(new_data[q]) for q in new_data)
    for q in new_data.keys():
        new_data[q] += [None] * (max_length - len(new_data[q]))

    pd.DataFrame(new_data).to_excel(os.path.join(res_dir, 'members_all_SPX_result.xlsx'), index=False)


def get_weekly_data_by_members(res_dir):
    os.makedirs(res_dir, exist_ok=True)
    quarter = members.columns
    for index, q in enumerate(quarter):
        if q < pd.to_datetime('2006-06-30'):
            continue
        q_name = q.strftime('%Y-%m-%d')

        # if q_name == '2009-06-30':
        data = get_quarter_data(q, quarter[index + 1] if index + 1 < len(quarter) else pd.datetime.today())

        save_path = os.path.join(res_dir, q_name + '.csv')
        data.to_csv(save_path, index=False)
        print(f'saveing {save_path} successfully')


def generate_members_daily(res_path='raw_data_set/members_daily.csv'):
    quarters=members.columns.tolist()
    for i in range(len(quarters)):
        pass





if __name__ == '__main__':
    # find_difference('2006-06-30')
    # mem_not_in_weekly, weekly_not_in_mem=find_difference('2018-03-30')
    # mem_not_in_weekly, weekly_not_in_mem=find_difference('2020-03-31')
    # mem_not_in_weekly, weekly_not_in_mem=find_difference('2019/12/31')
    # get_weekly_data_by_members('weekly_member_result/')
    members_SPX('members_SPX_result/')
