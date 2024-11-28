from tests.utils import timeseries
from tsgraph.nodes.core import df_node
from tsgraph.nodes.maths import add, mul
from tsgraph.nodes.utils import lag
from tsgraph.render import tree_str


def test_tree_str():
    df = df_node(timeseries([1,2,3,4]))
    n1 = add(df, mul(lag(df, 1), -1))
    n2 = lag(df, 1)
    result = tree_str([n1, n2, n1], str_func=lambda n: n.name)
    expected = '''@2 add
├─@1 df_node
└─align_ffill
  ├─mul
  │ └─lag
  │   └─@1
  └─@1
lag
└─@1
@2'''
    assert result == expected
