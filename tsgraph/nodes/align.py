from typing import Dict

import pandas as pd

from tsgraph.nodes.core import FuncNode, Node, scalar_node, to_series, HomogeneousNodeDecorator


@scalar_node
def make_index_union(*nodes):
    index = nodes[0].index
    for n in nodes[1:]:
        index = index.union(n.index)
    index = index.sort()
    return index


def align_ffill(node: Node, index_node: Node, state=None):
    """
    Returns the dataframe node aligned to the index of index_node, forward filling. State can be used to give
    initial values to be used when index_node ticks before node.
    """

    def align_ffill(df: pd.DataFrame, df_index, state=state):
        df = df.reindex(df_index.index, method='ffill')
        if state is None:
            df = df.dropna()
        else:
            df = df.fillna(state)
        if not df.empty:
            state = df.iloc[-1]
        return df, state

    return FuncNode(align_ffill, node, index_node, columns=node.columns)


def align_sum(node: Node, index_node: Node, state=None):
    """
    Returns the dataframe node aligned to the index of index_node, preserving the sum of the data.
    State can be used to give initial values to be used when index_node ticks before node.
    At some point might be nice to implement this directly to avoid any loss of precision.
    """
    from tsgraph.nodes.maths import cumsum
    from tsgraph.nodes.utils import diff
    return diff(align_ffill(cumsum(node), index_node, state=state), 1, state=0)


def align(node: Node, index_node: Node, how='ffill', state=None):
    if node == index_node:
        return node
    if how == 'ffill':
        return align_ffill(node, index_node, state=state)
    elif how == 'sum':
        return align_sum(node, index_node, state=state)
    else:
        raise ValueError(f"Invalid how parameter {how}")


class AlignedNodeDecorator(HomogeneousNodeDecorator):

    def __call__(self, *args, columns=None, aligner='left', how='ffill', state=None,  **kwargs):
        nodes = [n for n in args if isinstance(n, Node)]
        hows = how if isinstance(how, list) else [how] * len(nodes)
        states = state if isinstance(state, list) else [state] * len(nodes)
        if len(hows) != len(nodes):
            raise ValueError("If multiple hows given, must be one per node")
        if len(states) != len(nodes):
            raise ValueError("If multiple states given, must be one per node")
        if aligner == 'left':
            index_node = nodes[0]
        elif aligner == 'union':
            index_node = make_index_union(nodes)
        else:
            raise ValueError(f"Invalid aligner {aligner}")

        aligned_nodes = [align(n, index_node, how=h, state=s) for n, h, s in zip(nodes, hows, states)]
        aligned_args = []
        node_index = 0
        for v in args:
            if isinstance(v, Node):
                aligned_args.append(aligned_nodes[node_index])
                node_index += 1
            else:
                aligned_args.append(v)

        return super().__call__(*aligned_args, columns=columns, **kwargs)


aligned_node = AlignedNodeDecorator


@aligned_node
def _pack(values, columns):
    return pd.DataFrame({col: to_series(v) for col, v in zip(columns, values)}).dropna()


def pack(col_to_value: Dict, aligner='left', how='ffill', state=None):
    """
    Forms a multicolumn node with the given columns. Can be nodes or values. All inputs must be 1d.
    """
    return _pack(values=col_to_value.values(), aligner=aligner, how=how, state=state, columns=list(col_to_value))


@scalar_node
def get_col(df, key):
    return df.loc[:, key]


@scalar_node
def get_col_index(df, index):
    return df.iloc[:, index]
