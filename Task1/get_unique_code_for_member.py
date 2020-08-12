# coding=gbk
import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)

members = pd.read_excel('members.xlsx', skiprows=1)
members = members[members.columns.to_list()[1:]]
members = members.apply(lambda x: x.apply(lambda y: str(y).split()[0]))

quarter = members.columns
c1 = []
c2 = []
for index, q in enumerate(quarter):
    if q < pd.to_datetime('2005-01-01'):
        c1 += [c for c in members[q].values.tolist() if c != 'nan']
    else:
        c2 += [c for c in members[q].values.tolist() if c != 'nan']

c1 = list(set(c1))
c2 = list(set(c2))

c1.sort()
c2.sort()
if len(c1) < len(c2):
    c1 += [None] * (len(c2) - len(c1))
else:
    c2 += [None] * (len(c1) - len(c2))

pd.DataFrame({'<2005-01-01': c1, '>=2005-01-01': c2}).to_excel('split_by_20050101_on_members.xlsx')
