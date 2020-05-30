# coding=gbk
import better_exceptions
from sys import exit
better_exceptions.hook()

import pandas as pd

pd.set_option('expand_frame_repr', False)  # 设置当列太多不会出现省略号
pd.set_option('display.max_rows', 10)  # 设置显示的最多行数
pd.set_option('precision', 2)  # 设置显示的数字精度

# df = pd.read_csv('minute_data/yahoo.csv', sep=',',
#                  encoding='utf-8',
#                  parse_dates=['Date'],
#                  index_col=['Date'],
#                  usecols=['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'],
#                  error_bad_lines=False,  # 当某一行不是很干净，那么久略过该行
#                  na_values='NULL',  # 将表示为NULL的数值记做控制
#                  )

df = pd.read_csv('data/中证800_000906.SH.csv',
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
# df['成交额'] = df['Close'] * df['Volume']
# print(df['成交额'].quantile(0.25))  # 25%分位数
# print(df['成交额'].std())  # 25%分位数
# print(df['成交额'].median())  # 25%分位数
#
# df['下周期成交额'] = df['成交额'].shift(-1)  # 从下往上，-1的位置留一个NaN
#
# df['涨跌'] = df['Close'].diff(1)  # 从上往下，第一个位置留下一个NaN
#
# print(df)
#
# del df['涨跌']
# df.drop(['下周期成交额'], axis=1, inplace=True)
# print(df)
#
# df['涨跌幅'] = df['Close'].pct_change(1)  # 第一个为NaN  (Close-Close_1)/Close_1
# df['累计成交量'] = df['Volume'].cumsum()
# df['金钱'] = (df['涨跌幅'] + 1.).cumprod()  # 累乘
#
# df['Close_rank'] = df['Close'].rank(ascending=True, pct=False)
# print(df['Close'].value_counts())  # Series
#
# print(df[(df['Close'].between(10, 11)) & (df['Volume'] >= 178159800)])
# print(df[(df['Close'].between(10, 11)) | (df['Volume'] >= 178159800)])
#
# # index = df[df.index.isin(['2006-10-01', '2006-11-01', '2006-12-01'])].index
#
# df['月头'] = 'R'
# # print(df[df['月头']=='R'])
#
#
# print(df.dropna(how='all', subset=['月头', 'Close']))
# print(df.dropna(how='any', subset=['月头', 'Close']))
#
# # print(df.fillna(value=df['Close']))
# print(df.fillna(method='bfill'))  # method='ffill' 分别代表从后或者从前找到第一个非空的值去填充该空值
#
# print(df.notnull())
# print(df[df['月头'].isnull()])
#
# df.sort_index(ascending=0)  # 降序
# df.sort_values(by=['High', 'Low'], ascending=[0, 1])  # 降序\升序
#
# # 两个df上下合并
# df1 = df.iloc[0:5]
# df2 = df.iloc[3:8]
# df12 = df1.append(df2, ignore_index=True)
#
# # 去重
# df12.drop_duplicates(subset=df12.columns,
#                      keep='first',
#                      inplace=True)
#
# df12.reset_index(inplace=True, drop=True)  # drop=True代表将原来的index去掉，不保留
#
# df12.rename(columns={'High': 'H', 'Close': 'C'}, inplace=True)
#
# print(pd.DataFrame().empty)
#
# # 字符串操作
# df['月头'] = df['月头'].str.lower()
# print(df['月头'].str.contains('r'))
# print(df['月头'].str.replace('r', 'R'))
# # lower()/upper()
# # len()
# # strip()
#
#
# # 时间处理
# # df = pd.read_csv('minute_data/yahoo.csv', sep=',',
# #                  encoding='utf-8',
# #                  # parse_dates=['Date'],
# #                  # index_col=['Date'],
# #                  usecols=['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'],
# #                  error_bad_lines=False,  # 当某一行不是很干净，那么久略过该行
# #                  na_values='NULL',  # 将表示为NULL的数值记做控制
# #                  )
# # print(type(df.at[0, 'Date']))
# # df['Date'] = pd.to_datetime(df['Date'])
# # print(type(df.at[0, 'Date']))
# #
# # print(df['Date'].dt.year)
# # print(df['Date'].dt.week)
# # print(df['Date'].dt.dayofyear)
# # print(df['Date'].dt.dayofweek)  # 0代表星期一
# #
# # df['weekday'] = df['Date'].dt.weekday
# # df['星期'] = df['Date'].dt.weekday_name
# # print(df[['weekday', '星期']])
# #
# # print(df['Date'].dt.days_in_month)  # 该月一共有多少天
# # print(df['Date'].dt.is_month_start)  # 该月一共有多少天  is_month_end
# # print(df['Date']+pd.Timedelta(hours=1))
#
#
# print(df['High'].mean())
# print(df['High'].rolling(3).mean())
# print(df['High'].rolling(3).max())
# print(df['High'].rolling(3).std())
#
# # expanding 从第一行到当前行
# print(df['High'].expanding().max())
#
# # 批量导入数据,
# # 使用HDFStore类型的文件 ，相当于一个数据库文件，分为很多的数据表
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
# # 将股票的日线数据转换为周线数据
#
# # df.set_index('Date', inplace=True)
# rule_type = '1W'
# period_df = pd.DataFrame()
# period_df['Close'] = df['Close'].resample(rule=rule_type).last()  # 取一周中Close的最后一天
# period_df['Open'] = df['Open'].resample(rule=rule_type).first()
# period_df['High'] = df['High'].resample(rule=rule_type).max()
# period_df['Low'] = df['Low'].resample(rule=rule_type).max()
# period_df['Volume'] = df['Volume'].resample(rule=rule_type).sum()
# print(period_df)
#
# # method 2
# rule_type = '1W'
# period_df = df.resample(rule=rule_type,
#                         # on='Date', #从列中取，用于确定在哪一列中使用resample，当然如果Date已经变成了index，就不用这个on参数了
#                         base=0,  # 主要用于处理分钟数据或者小时数据
#                         label='left',  # left控制使用前面一周的周日，如果是right则是控制使用后一个周期的周日
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

# 分组
print(df.groupby('Date').size())  # 每天交易的股票数目
print(df.groupby('Code').size())  # 每一个代码交易的股票数目
print(df.groupby('Code').get_group('603993.SH'))  # 每一个代码交易的股票数目
# print(df.groupby('Code').describe())  #
print(df.groupby('Code').head(2))  # first(),last() 这里的index是原先的Date
print(df.groupby('Code').nth(0))  # 取每一组的第1行，数值上近似于head(1)，但是这里的index变成了Code，所以不一样的地方在这，原先的index=Date被自动druo了
print(df.groupby('Code', as_index=False).nth(0))  # 完全等价于head(1)
print(df.groupby('Code', as_index=False)['Close', 'Volume'].mean())  # sum(),max(),min()
print(df.groupby('Code', as_index=False)[['Close', 'Volume']].rank())

df.reset_index(inplace=True)
df.loc[pd.to_datetime(df['Date']).dt.day < 15, '月份'] = '上旬'
df['月份'].fillna(value='下旬', inplace=True)
print(df)
# df['Date']=df['Date'].map(lambda x:x.strftime('%Y-%m-%d')) #将datetime类型转换为str类型
# res['daynum']=res['daynum'].map(lambda x:str(x).split()[0])  将timedelta类型转换为str类型，并且取日期数

print(df.groupby(['Code','月份']).size())


groups=pd.DataFrame(columns=df.columns)
for code,group in df.groupby('Code'):
    # print(code)
    # print(group)
    # result_group=group.mean()
    groups.append(group,ignore_index=True)

# groups.set_index('Date')


