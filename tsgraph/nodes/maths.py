import pandas as pd
from more_itertools.recipes import partition

from tsgraph.node import node, Node
from tsgraph.nodes.pack import pack_ffill


@node
def df_add(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().sum(axis=1)


@node
def scalar_add(df: pd.DataFrame, *scalars) -> pd.DataFrame:
    return df + sum(scalars)


def add(*args):
    nodes, scalars = partition(lambda v: isinstance(v, Node), args)
    nodes = list(nodes)
    scalars = list(scalars)
    if len(nodes) > 1:
        result = df_add(pack_ffill(*nodes))
    else:
        result = nodes[0]
    if scalars:
        result = scalar_add(result, *scalars)
    return result


@node
def cumsum(df, state=None):
    result = df.cumsum()
    if not result.empty:
        if state is not None:
            result += state
        state = result.iloc[-1]
    return result, state
