import pandas as pd

from tests.utils import timeseries
from tsgraph.nodes.align import pack, get_col, get_col_index
from tsgraph.nodes.core import df_node, as_valid_result


def test_pack():
    a = df_node(timeseries([1,2]))
    b = df_node(timeseries([3,4]))
    c = pack({'a': a, 'b': b, 'foo': 'bar'})
    result = c.calc()
    assert list(result.columns) == ['a', 'b', 'foo']
    assert c.columns == ['a', 'b', 'foo']
    assert list(result['a']) == [1, 2]
    assert list(result['b']) == [3, 4]
    assert list(result['foo']) == ['bar', 'bar']

    pd.testing.assert_frame_equal(get_col(c, 'a').calc(), as_valid_result(result['a']))
    pd.testing.assert_frame_equal(get_col(c, 'b').calc(), as_valid_result(result['b']))
    pd.testing.assert_frame_equal(get_col(c, 'foo').calc(), as_valid_result(result['foo']))

    pd.testing.assert_frame_equal(get_col_index(c, 0).calc(), as_valid_result(result['a']))
    pd.testing.assert_frame_equal(get_col_index(c, 1).calc(), as_valid_result(result['b']))
    pd.testing.assert_frame_equal(get_col_index(c, 2).calc(), as_valid_result(result['foo']))
