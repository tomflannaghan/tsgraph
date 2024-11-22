import pandas as pd

from tsgraph.node import node, FuncNode


def pack_ffill(*args, state=None, columns=None):
    cols = sum((df.columns for df in args), []) if columns is None else columns

    def _pack_impl(*dfs, state=state):
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

    return FuncNode(_pack_impl, *args, columns=cols)


@node
def df_add(df: pd.DataFrame):
    return df.dropna().sum(axis=1)


def add(*args):
    return df_add(pack_ffill(*args))


@node
def cumsum(df, state=None):
    result = df.cumsum()
    if not result.empty:
        if state is not None:
            result += state
        state = result.iloc[-1]
    return result, state
