# coding=gbk
import better_exceptions
from sys import exit
better_exceptions.hook()

import pandas as pd

pd.set_option('expand_frame_repr', False)  # ���õ���̫�಻�����ʡ�Ժ�
pd.set_option('display.max_rows', 10)  # ������ʾ���������
pd.set_option('precision', 2)  # ������ʾ�����־���

# df = pd.read_csv('minute_data/yahoo.csv', sep=',',
#                  encoding='utf-8',
#                  parse_dates=['Date'],
#                  index_col=['Date'],
#                  usecols=['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'],
#                  error_bad_lines=False,  # ��ĳһ�в��Ǻܸɾ�����ô���Թ�����
#                  na_values='NULL',  # ����ʾΪNULL����ֵ��������
#                  )

df = pd.read_csv('data/��֤800_000906.SH.csv',
                 parse_dates=['date'],
                 # index_col='date',
                 usecols=['code', 'date', 'open', 'high', 'low', 'close', 'volume'],
                 error_bad_lines=False,
                 na_values='NULL')

df.rename(columns={'code': 'Code',
                   'date': 'Date',
                   'open': 'Open',
                   'high': 'High',
                   'low': 'Low',
                   'close': 'Close',
                   'volume': 'Volume'}, inplace=True)

df.set_index('Date', inplace=True)
# print(df['High'])  # Series
# print(df[['High']])  # DataFrame
# print(df.loc['2018-10-08'])  # Series
# print(df.iloc[-1])  # Series
# print(df.loc['2018-10-08':'2019-09-30'])  # DataFrame
# print(df.iloc[2:5, 1:3])  # DataFrame
# print(df.loc[:, 'High':'Open'])  # DataFrame
# print(df.at['2018-10-08', 'High'])  # Scale
# print(df.iat[1, 3])  # Scale
#
# df['�ɽ���'] = df['Close'] * df['Volume']
# print(df['�ɽ���'].quantile(0.25))  # 25%��λ��
# print(df['�ɽ���'].std())  # 25%��λ��
# print(df['�ɽ���'].median())  # 25%��λ��
#
# df['�����ڳɽ���'] = df['�ɽ���'].shift(-1)  # �������ϣ�-1��λ����һ��NaN
#
# df['�ǵ�'] = df['Close'].diff(1)  # �������£���һ��λ������һ��NaN
#
# print(df)
#
# del df['�ǵ�']
# df.drop(['�����ڳɽ���'], axis=1, inplace=True)
# print(df)
#
# df['�ǵ���'] = df['Close'].pct_change(1)  # ��һ��ΪNaN  (Close-Close_1)/Close_1
# df['�ۼƳɽ���'] = df['Volume'].cumsum()
# df['��Ǯ'] = (df['�ǵ���'] + 1.).cumprod()  # �۳�
#
# df['Close_rank'] = df['Close'].rank(ascending=True, pct=False)
# print(df['Close'].value_counts())  # Series
#
# print(df[(df['Close'].between(10, 11)) & (df['Volume'] >= 178159800)])
# print(df[(df['Close'].between(10, 11)) | (df['Volume'] >= 178159800)])
#
# # index = df[df.index.isin(['2006-10-01', '2006-11-01', '2006-12-01'])].index
#
# df['��ͷ'] = 'R'
# # print(df[df['��ͷ']=='R'])
#
#
# print(df.dropna(how='all', subset=['��ͷ', 'Close']))
# print(df.dropna(how='any', subset=['��ͷ', 'Close']))
#
# # print(df.fillna(value=df['Close']))
# print(df.fillna(method='bfill'))  # method='ffill' �ֱ����Ӻ���ߴ�ǰ�ҵ���һ���ǿյ�ֵȥ���ÿ�ֵ
#
# print(df.notnull())
# print(df[df['��ͷ'].isnull()])
#
# df.sort_index(ascending=0)  # ����
# df.sort_values(by=['High', 'Low'], ascending=[0, 1])  # ����\����
#
# # ����df���ºϲ�
# df1 = df.iloc[0:5]
# df2 = df.iloc[3:8]
# df12 = df1.append(df2, ignore_index=True)
#
# # ȥ��
# df12.drop_duplicates(subset=df12.columns,
#                      keep='first',
#                      inplace=True)
#
# df12.reset_index(inplace=True, drop=True)  # drop=True����ԭ����indexȥ����������
#
# df12.rename(columns={'High': 'H', 'Close': 'C'}, inplace=True)
#
# print(pd.DataFrame().empty)
#
# # �ַ�������
# df['��ͷ'] = df['��ͷ'].str.lower()
# print(df['��ͷ'].str.contains('r'))
# print(df['��ͷ'].str.replace('r', 'R'))
# # lower()/upper()
# # len()
# # strip()
#
#
# # ʱ�䴦��
# # df = pd.read_csv('minute_data/yahoo.csv', sep=',',
# #                  encoding='utf-8',
# #                  # parse_dates=['Date'],
# #                  # index_col=['Date'],
# #                  usecols=['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'],
# #                  error_bad_lines=False,  # ��ĳһ�в��Ǻܸɾ�����ô���Թ�����
# #                  na_values='NULL',  # ����ʾΪNULL����ֵ��������
# #                  )
# # print(type(df.at[0, 'Date']))
# # df['Date'] = pd.to_datetime(df['Date'])
# # print(type(df.at[0, 'Date']))
# #
# # print(df['Date'].dt.year)
# # print(df['Date'].dt.week)
# # print(df['Date'].dt.dayofyear)
# # print(df['Date'].dt.dayofweek)  # 0��������һ
# #
# # df['weekday'] = df['Date'].dt.weekday
# # df['����'] = df['Date'].dt.weekday_name
# # print(df[['weekday', '����']])
# #
# # print(df['Date'].dt.days_in_month)  # ����һ���ж�����
# # print(df['Date'].dt.is_month_start)  # ����һ���ж�����  is_month_end
# # print(df['Date']+pd.Timedelta(hours=1))
#
#
# print(df['High'].mean())
# print(df['High'].rolling(3).mean())
# print(df['High'].rolling(3).max())
# print(df['High'].rolling(3).std())
#
# # expanding �ӵ�һ�е���ǰ��
# print(df['High'].expanding().max())
#
# # ������������,
# # ʹ��HDFStore���͵��ļ� ���൱��һ�����ݿ��ļ�����Ϊ�ܶ�����ݱ�
#
# df.to_hdf('minute_data.h5',
#           key='table_name',
#           mode='w')
#
# df: pd.DataFrame = pd.read_hdf('minute_data.h5',
#                                key='table_name',
#                                mode='r')
# print(df)
#
# df: pd.DataFrame = pd.read_hdf('minute_data.h5', mode='r')
# print(df)
#
# h5_store = pd.HDFStore('minute_data.h5', mode='w')
# h5_store['table_name'] = df
# h5_store.close()
#
# with pd.HDFStore('minute_data.h5', mode='w') as h5_store:
#     h5_store['table_name'] = df
#
# with pd.HDFStore('minute_data.h5', mode='r') as h5_store:
#     print(h5_store.get('table_name'))
#     print(h5_store['table_name'])
#
# # ����Ʊ����������ת��Ϊ��������
#
# # df.set_index('Date', inplace=True)
# rule_type = '1W'
# period_df = pd.DataFrame()
# period_df['Close'] = df['Close'].resample(rule=rule_type).last()  # ȡһ����Close�����һ��
# period_df['Open'] = df['Open'].resample(rule=rule_type).first()
# period_df['High'] = df['High'].resample(rule=rule_type).max()
# period_df['Low'] = df['Low'].resample(rule=rule_type).max()
# period_df['Volume'] = df['Volume'].resample(rule=rule_type).sum()
# print(period_df)
#
# # method 2
# rule_type = '1W'
# period_df = df.resample(rule=rule_type,
#                         # on='Date', #������ȡ������ȷ������һ����ʹ��resample����Ȼ���Date�Ѿ������index���Ͳ������on������
#                         base=0,  # ��Ҫ���ڴ���������ݻ���Сʱ����
#                         label='left',  # left����ʹ��ǰ��һ�ܵ����գ������right���ǿ���ʹ�ú�һ�����ڵ�����
#                         closed='left',
#                         ).agg({
#     'Open': 'first',
#     'Close': 'last',
#     'High': 'max',
#     'Low': 'min',
#     'Volume': 'sum',
# })
#
# print(period_df)
# period_df = period_df[period_df['Volume'] > 0]

# ����
print(df.groupby('Date').size())  # ÿ�콻�׵Ĺ�Ʊ��Ŀ
print(df.groupby('Code').size())  # ÿһ�����뽻�׵Ĺ�Ʊ��Ŀ
print(df.groupby('Code').get_group('603993.SH'))  # ÿһ�����뽻�׵Ĺ�Ʊ��Ŀ
# print(df.groupby('Code').describe())  #
print(df.groupby('Code').head(2))  # first(),last() �����index��ԭ�ȵ�Date
print(df.groupby('Code').nth(0))  # ȡÿһ��ĵ�1�У���ֵ�Ͻ�����head(1)�����������index�����Code�����Բ�һ���ĵط����⣬ԭ�ȵ�index=Date���Զ�druo��
print(df.groupby('Code', as_index=False).nth(0))  # ��ȫ�ȼ���head(1)
print(df.groupby('Code', as_index=False)['Close', 'Volume'].mean())  # sum(),max(),min()
print(df.groupby('Code', as_index=False)[['Close', 'Volume']].rank())

df.reset_index(inplace=True)
df.loc[pd.to_datetime(df['Date']).dt.day < 15, '�·�'] = '��Ѯ'
df['�·�'].fillna(value='��Ѯ', inplace=True)
print(df)
# df['Date']=df['Date'].map(lambda x:x.strftime('%Y-%m-%d')) #��datetime����ת��Ϊstr����
# res['daynum']=res['daynum'].map(lambda x:str(x).split()[0])  ��timedelta����ת��Ϊstr���ͣ�����ȡ������

print(df.groupby(['Code','�·�']).size())


groups=pd.DataFrame(columns=df.columns)
for code,group in df.groupby('Code'):
    # print(code)
    # print(group)
    # result_group=group.mean()
    groups.append(group,ignore_index=True)

# groups.set_index('Date')


