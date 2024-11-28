from collections import defaultdict
from graphlib import TopologicalSorter


def expand_range(old_range, new_range):
    if old_range is None:
        return new_range, True
    old_min, old_max = old_range
    new_min, new_max = new_range
    expanded_range = (None if old_min is None or new_min is None else min(old_min, new_min), max(old_max, new_max))
    return expanded_range, expanded_range != old_range


def get_execution_plan(nodes, start_dt, end_dt):
    node_to_parents = defaultdict(set)

    # Now find the broadest range of evaluation needed for every node in the graph.
    current_ranges = {n: (start_dt, end_dt) for n in nodes}
    updated_nodes = set(nodes)
    while updated_nodes:
        node = updated_nodes.pop()
        this_start_dt, this_end_dt = current_ranges[node]
        node_evals = node.get_required_evaluations(this_start_dt, this_end_dt)
        for n, r in node_evals:
            node_to_parents[node].add(n)
            new_range, change = expand_range(current_ranges.get(n), r)
            if change:
                updated_nodes.add(n)
                current_ranges[n] = new_range

    # The nodes are a DAG so we can topo-sort them to get the execution order.
    topo_sort = list(TopologicalSorter(node_to_parents).static_order())

    # Now find the order in which we can garbage-collect intermediate data. We do that by finding the last
    # node that needs each thing.
    disposals = defaultdict(set)
    already_disposed = set(nodes)  # We don't want to dispose of the things we are trying to calculate!
    for node in topo_sort[::-1]:
        for p in node_to_parents[node]:
            if p not in already_disposed:
                already_disposed.add(p)
                disposals[node].add(p)

    return [(n, current_ranges[n], disposals[n]) for n in topo_sort]


def run_execution_plan(plan):
    working_data = {}
    for node, (start_dt, end_dt), disposals in plan:
        working_data[node] = node.evaluate(start_dt, end_dt, working_data)
        for n in disposals:
            del working_data[n]
    return working_data


def evaluate_nodes(nodes, start_dt, end_dt):
    plan = get_execution_plan(nodes, start_dt, end_dt)
    return run_execution_plan(plan)
