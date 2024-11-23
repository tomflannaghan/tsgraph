import pandas as pd

from tests.utils import timeseries, calc_check_consistency
from tsgraph.node import df_node, as_valid_result
from tsgraph.nodes.maths import ewma, cumsum, add


def test_ewma():
    sample_data = as_valid_result(timeseries([1, 2, 3, 4, 5] * 20))
    node = ewma(df_node(sample_data), span=5)
    result = calc_check_consistency(node)
    expected = sample_data.ewm(span=5).mean()
    pd.testing.assert_frame_equal(result.round(5), expected.round(5))


def test_ewma_multi_col():
    sample_data = pd.DataFrame({'a': timeseries([1, 2, 3, 4, 5] * 20), 'b': timeseries([3, 3, 3, 3, 3] * 20),
                                'c': timeseries([0, 1, 0, 1, 0] * 20)})
    node = ewma(df_node(sample_data), span=5)
    result = calc_check_consistency(node)
    expected = sample_data.ewm(span=5).mean()
    pd.testing.assert_frame_equal(result.round(5), expected.round(5))


def test_cumsum():
    sample_data = as_valid_result(timeseries([1, 2, 3, 4, 5]))
    node = cumsum(df_node(sample_data))
    result = calc_check_consistency(node)
    expected = sample_data.cumsum()
    pd.testing.assert_frame_equal(result, expected)


def test_add():
    sample_data = as_valid_result(timeseries([1, 2, 3]))
    data_node = df_node(sample_data)
    node = add(data_node, 3)
    node.calc().equals(sample_data + 3)
    node = add(data_node, 4, 2)
    node.calc().equals(sample_data + 6)
    node = add(data_node, 4, 2, data_node)
    node.calc().equals(2 * sample_data + 6)
    node = add(data_node, 4, 2, df_node(sample_data))
    node.calc().equals(2 * sample_data + 6)
