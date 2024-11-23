import pandas as pd
from pandas import DatetimeIndex

from tsgraph.nodes.maths import cumsum
from tsgraph.nodes.pack import pack_ffill
from tsgraph.node import df_node


def default_index(n):
    t = pd.Timestamp('2000-01-01')
    return DatetimeIndex([t + pd.offsets.Day(i) for i in range(n)])


def timeseries(values):
    return pd.Series(values, default_index(len(values))).dropna()


a = df_node(timeseries([1, 2, 3, 4]))
b = df_node(timeseries([None, None, 6, 7]))

c = pack_ffill(a, b, state=0, columns=['a', 'b'])
d = cumsum(c)

print(d.advance(pd.Timestamp('2020-01-01')))

d.reset_all()
print(d.advance(pd.Timestamp('2000-01-02')))
print(d.advance(pd.Timestamp('2000-01-04')))

c.calc()