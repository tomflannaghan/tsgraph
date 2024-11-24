import pickle

import pandas as pd

from tests.utils import timeseries
from tsgraph.nodes.core import node, df_node
from tsgraph.nodes.utils import diff


@node
def stateful(df, state=0):
    # This is an invalid node as state depends on batching, but we'll use it to track how many times it's called.
    state += 1
    return df * 2, state


def test_state():
    n = stateful(df_node(timeseries([1, 2, 3, 4, 5, 6])))
    result = n.advance(pd.Timestamp('1999-01-01'))
    assert n._state == 1
    assert result.empty
    result = n.advance(pd.Timestamp('2000-01-01'))
    assert n._state == 2
    assert len(result) == 1
    result = n.advance(pd.Timestamp('2000-01-01'))
    assert n._state == 2
    assert result.empty
    n.reset()
    assert n._state == 0
    n.advance(pd.Timestamp('1999-01-01'))
    assert n._state == 1
    n.advance(pd.Timestamp('2001-01-01'))
    assert n._state == 2


def test_pickle():
    # Diff is a fairly complex set of nodes so a good test case.
    n = diff(df_node(timeseries([1.0, 2, 4, 7, 11, 16, 32, 55])), 2)
    expected = n.calc()
    n.reset_all()
    # Partial advance, so we can test pickling of state
    pd.testing.assert_frame_equal(expected.iloc[:4], n.advance(expected.index[3]))
    # Get a version by pickling and loading
    n2 = pickle.loads(pickle.dumps(n))
    # Check that both advance from the point they were originally advanced to correctly.
    result_orig = n.advance(expected.index[-1])
    pd.testing.assert_frame_equal(expected.iloc[4:], result_orig)
    result_new = n2.advance(expected.index[-1])
    pd.testing.assert_frame_equal(expected.iloc[4:], result_new)
