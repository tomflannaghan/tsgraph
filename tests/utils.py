import pandas as pd
from pandas import DatetimeIndex


def default_index(n, t = pd.Timestamp('2000-01-01')):
    return DatetimeIndex([t + pd.offsets.Day(i) for i in range(n)])


def timeseries(values, t = pd.Timestamp('2000-01-01')):
    return pd.Series(values, default_index(len(values), t)).dropna()


def calc_check_consistency(node, max_days=200):
    start = node.graph_start_dt()
    # batched eval
    segments = []
    for date in pd.date_range(start, start + pd.offsets.Day(max_days), freq='3D'):
        segments.append(node.advance(date))
    # Check it was a sufficiently long evaluation, otherwise could be meaningless.
    assert len(segments) > 1
    result_chunks = pd.concat(segments)
    # straight through eval
    result = node.calc()
    pd.testing.assert_frame_equal(result, result_chunks, check_freq=False)
    return result
