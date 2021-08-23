"""
Graph algorithms and utility functions.
"""


# ------------------------------------------------------------------------------------------------

def connected_component(nodes, edges, reference_node):
    """
    :param nodes: the list of graph nodes.
    :param edges: a list of pairs (i, j), indicating an edge between nodes[i] and nodes[j].
    :param reference_node: the reference node.
    :return: the set of nodes in the same connected component as the reference node.
    :remark the implementation has O(E * N) complexity, not recommended for large and dense graphs.
    """

    try:
        ref_idx = nodes.index(reference_node)
    except Exception as e:
        print(e)
        return None

    visited = [False] * len(nodes)
    visited[ref_idx] = True

    while True:
        changed = False

        for edge in edges:
            if (visited[edge[0]] and not visited[edge[1]]) or \
                    (visited[edge[1]] and not visited[edge[0]]):
                visited[edge[0]] = True
                visited[edge[1]] = True
                changed = True

        if not changed:
            break

    cc = []
    for i in range(len(nodes)):
        if visited[i]:
            cc.append(nodes[i])

    return cc


# ------------------------------------------------------------------------------------------------
