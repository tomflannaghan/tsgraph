import pandas as pd

from tests.utils import timeseries
from tsgraph.node import node, df_node


@node
def stateful(df, state=0):
    # This is an invalid node as state depends on batching, but we'll use it to track how many times it's called.
    state += 1
    return df * 2, state


def test_state():
    node = stateful(df_node(timeseries([1,2,3,4,5,6])))
    result = node.advance(pd.Timestamp('1999-01-01'))
    assert node._state == 1
    assert result.empty
    result = node.advance(pd.Timestamp('2000-01-01'))
    assert node._state == 2
    assert len(result) == 1
    result = node.advance(pd.Timestamp('2000-01-01'))
    assert node._state == 2
    assert result.empty
    node.reset()
    assert node._state == 0
    node.advance(pd.Timestamp('1999-01-01'))
    assert node._state == 1
    node.advance(pd.Timestamp('2001-01-01'))
    assert node._state == 2
