import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

symbol = '000060'
my_df = pd.read_csv('res/t0625-get_CSI800_unique_daily_history/' + symbol + '.csv', index_col='trade_date',
                    parse_dates=True)

lq_df = pd.read_excel('res/csi800.daily.close.xlsx', index_col=0)

lq_df = lq_df[lq_df.index == f'{symbol} CH'].T
lq_df.drop(index=['PX_LAST'], inplace=True)
lq_df.index = lq_df.index.map(lambda x: pd.to_datetime(x))

start_date = '2020-03-01'
end_date = '2020-04-01'

show_lq_df = lq_df.loc[start_date:end_date, :]
show_my_df = my_df.loc[start_date:end_date, :]
show_my_df['close'].plot(label='my_df')
show_lq_df['000060 CH'].plot(label='target_df')

plt.legend()
plt.show()
