from typing import Iterable, List

import graphviz

from tsgraph.nodes.core import Node


def node_graph(node: Node) -> graphviz.Digraph:
    node_to_graphviz = {}
    index = [0]

    def get_gnode(dot, node):
        if node not in node_to_graphviz:
            dot.node(str(index[0]), str(node))
            node_to_graphviz[node] = str(index[0])
            index[0] += 1
        return node_to_graphviz[node]

    def connect_edges(dot, node):
        gnode = get_gnode(dot, node)
        for p in node.parents:
            connect_edges(dot, p)
            dot.edge(get_gnode(dot, p), gnode)

    dot = graphviz.Digraph('node-graph')
    connect_edges(dot, node)
    return dot


def tree_str(nodes: Node | Iterable[Node], str_func=str) -> str:
    if isinstance(nodes, Node):
        nodes = [nodes]

    nodes_seen = set()
    refs = {}
    refs_seen = set()

    def _find_required_refs(node):
        if node in nodes_seen:
            if node not in refs:
                refs[node] = len(refs) + 1
        else:
            nodes_seen.add(node)
            for n in node.parents:
                _find_required_refs(n)

    for node in nodes:
        _find_required_refs(node)

    def _stringify(node):
        prefix = f'@{refs[node]} ' if node in refs else ''
        return prefix + str_func(node)

    def _prefix(i_inner, i_outer, n_outer):
        end_of_list = i_outer == n_outer - 1
        if i_inner == 0:
            return '└─' if end_of_list else '├─'
        else:
            return '  ' if end_of_list else '│ '

    def _tree_str_impl(node) -> List[str]:
        if node in refs_seen:
            return [f'@{refs[node]}']
        if node in refs:
            refs_seen.add(node)
        result = [_stringify(node)]
        for i, n in enumerate(node.parents):
            lines = _tree_str_impl(n)
            result.extend(_prefix(j, i, len(node.parents)) + l for j, l in enumerate(lines))
        return result

    print(refs)
    result = sum((_tree_str_impl(n) for n in nodes), [])
    return '\n'.join(result)
