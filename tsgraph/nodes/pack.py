import pandas as pd

from tsgraph.node import FuncNode, Node


def pack_ffill(*args, state=None, columns=None):
    cols = sum((df.columns for df in args), []) if columns is None else columns

    def pack_ffill(*dfs, state=state):
        # State is the initial values as we will be forward filling.
        # Note it might be worth implementing this without state by recording the latest value on each node.
        df = pd.concat(dfs, axis=1, ignore_index=True).ffill()
        df.columns = cols
        if state is None:
            df = df.dropna()
        else:
            df = df.fillna(state)
        if not df.empty:
            state = df.iloc[-1]
        return df, state

    return FuncNode(pack_ffill, *args, columns=cols)
