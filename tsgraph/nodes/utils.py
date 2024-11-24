from typing import List

import pandas as pd
from pandas import DatetimeIndex

from tsgraph.nodes.core import node, Node
from tsgraph.nodes.maths import add, mul


@node
def lag(df: pd.DataFrame, n: int, state=None):
    """Lag function, lags by a number of data points. State holds any data not yet emitted as a dataframe."""
    if df.empty:
        return df, state
    if state is None:
        state = df.iloc[:0]
    full_vals = pd.concat([state, df], ignore_index=True)
    to_output = full_vals.iloc[:-n]
    state = full_vals.iloc[-n:]
    index = df.index[-len(to_output):] if len(to_output) else DatetimeIndex([])
    return to_output.set_index(index), state


def diff(df: Node, n: int = 1, state=None):
    return add(df, mul(lag(df, n, state=state), -1))


@node
def join(n1, n2, join_time: pd.Timestamp):
    """Outputs n1 until the join time, then switches to n2. On join time, result will be n2."""
    if n1.empty and n2.empty:
        return n1

    if not n1.empty:
        if n1.index[-1] < join_time:
            pass
        elif n1.index[0] >= join_time:
            n1 = n1.iloc[:0]
        else:
            n1 = n1.loc[n1.index < join_time]

    if not n2.empty:
        if n2.index[0] >= join_time:
            pass
        elif n2.index[-1] < join_time:
            n2 = n2.iloc[:0]
        else:
            n2 = n2.loc[n2.index >= join_time]

    return pd.concat([n1, n2])


def stitch(data_nodes: List[Node], dates: List[pd.Timestamp]):
    result = data_nodes[0]
    for node, start_dt in zip(data_nodes[1:], dates[:-1]):
        result = join(result, node, join_time=start_dt)
    return result
