import pickle

import pandas as pd

from tests.utils import timeseries
from tsgraph.nodes.core import node, df_node
from tsgraph.nodes.utils import diff


def test_pickle():
    # Diff is a fairly complex set of nodes so a good test case.
    n = diff(df_node(timeseries([1.0, 2, 4, 7, 11, 16, 32, 55])), 2)
    expected = n.calc()
    # Partial advance, so we can test pickling of state
    pd.testing.assert_frame_equal(expected.iloc[:4], n[:expected.index[3]])
    # Get a version by pickling and loading
    n2 = pickle.loads(pickle.dumps(n))
    # Check that both advance from the point they were originally advanced to correctly.
    result_orig = n[expected.index[3]:expected.index[-1]]
    pd.testing.assert_frame_equal(expected.iloc[4:], result_orig)
    result_new = n2[expected.index[3]:expected.index[-1]]
    pd.testing.assert_frame_equal(expected.iloc[4:], result_new)
