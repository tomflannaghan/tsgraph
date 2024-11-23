import numpy as np
import pandas as pd
from more_itertools.recipes import partition
from numba import jit

from tsgraph.node import node, Node, scalar_node
from tsgraph.nodes.pack import pack_ffill


@scalar_node
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


@jit(nopython=True)
def _ewma_impl(data: np.array, state, alpha):
    # Calculates the fast ewma, so when the number of observations is low, we keep track of the denominator
    # in the average too. State is of the form (n_obs, num, denom). Requires 1d data.
    n_obs, num, denom = state
    result = np.zeros(data.shape)
    final_denom = 1 / alpha
    for i, obs in enumerate(data):
        if n_obs < 10 / alpha:
            denom += (1 - alpha) ** n_obs
        else:
            denom = final_denom
        num = obs + (1 - alpha) * num
        result[i] = num / denom
        n_obs += 1
    return result, (n_obs, num, denom)


@node
def ewma(df: pd.DataFrame, span: float, state=None):
    if state is None:
        state = [(0, 0, 0)] * df.shape[-1]
    result = {}
    for i, col in enumerate(df.columns):
        result[col], state[i] = _ewma_impl(df.values[:,0], state[i], 2 / (1 + span))

    return pd.DataFrame(result, index=df.index), state
