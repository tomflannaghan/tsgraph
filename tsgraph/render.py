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
        for p in node._parents:
            connect_edges(dot, p)
            dot.edge(get_gnode(dot, p), gnode)

    dot = graphviz.Digraph('node-graph')
    connect_edges(dot, node)
    return dot
