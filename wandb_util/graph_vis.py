from collections import defaultdict

import networkx as nx


def dag_pos(dag: nx.DiGraph):
    depths = {}
    for node in nx.topological_sort(dag):
        preds = list(dag.predecessors(node))
        if preds:
            depths[node] = 1 + max(depths[p] for p in preds)
        else:
            depths[node] = 0

    # Group nodes by depth
    layers = defaultdict(list)
    for node, depth in depths.items():
        layers[depth].append(node)

    # Assign positions: x by depth, y by order within the layer
    pos = {}
    for x, nodes in layers.items():
        for y, node in enumerate(nodes):
            pos[node] = (x, -y)  # y is negative to have top-down layering

    return pos
