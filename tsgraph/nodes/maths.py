import numpy as np
import pandas as pd
from numba import jit

from tsgraph.nodes.align import aligned_node
from tsgraph.nodes.core import node, scalar_node


@scalar_node
def df_sum(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().sum(axis=1)


@aligned_node
def add(*args) -> pd.DataFrame:
    result = args[0].copy()
    for v in args[1:]:
        result += v
    return result.dropna()


@scalar_node
def df_prod(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().prod(axis=1)


@aligned_node
def mul(*args) -> pd.DataFrame:
    result = args[0].copy()
    for v in args[1:]:
        result *= v
    return result.dropna()


@aligned_node
def div(num, denom) -> pd.DataFrame:
    return (num / denom).dropna()


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
    # in the average too. State is of the form (n_obs, num, denom, denom_term). Requires 1d data.
    n_obs, num, denom, denom_term = state
    result = np.zeros(data.shape)
    for i, obs in enumerate(data):
        denom += denom_term
        denom_term *= (1 - alpha)
        num = obs + (1 - alpha) * num
        result[i] = num / denom
        n_obs += 1
    return result, (n_obs, num, denom, denom_term)


@node
def ewma(df: pd.DataFrame, span: float, state=None):
    if state is None:
        state = [(0, 0, 0, 1)] * df.shape[-1]
    result = []
    for i, col in enumerate(df.columns):
        this_result, state[i] = _ewma_impl(df.values[:, i], state[i], 2 / (1 + span))
        result.append(this_result)
    return pd.DataFrame(np.vstack(result).T, index=df.index, columns=df.columns), state
