import pandas as pd

from tsgraph.node import node
from tsgraph.nodes.pack import pack_ffill


@node
def df_add(df: pd.DataFrame) -> pd.DataFrame:
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
