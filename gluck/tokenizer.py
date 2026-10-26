from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import torch
from torch_geometric.data import Data

from .constants import CONTEXT, CT, OO, ORIGINAL, OT, TC, TO, TOKEN, TT


@dataclass(frozen=True)
class Token:
    kind: str
    members: frozenset
    edges: frozenset


def canonical_edge(u, v):
    return min(int(u), int(v)), max(int(u), int(v))


def make_token(kind, nodes, edges):
    return Token(kind, frozenset(map(int, nodes)), frozenset(canonical_edge(*e) for e in edges))


def unique_tokens(tokens):
    seen, result = set(), []
    for token in tokens:
        key = token.kind, token.edges
        if key not in seen:
            seen.add(key)
            result.append(token)
    return result


def residual_graph(graph, covered):
    residual = nx.Graph()
    residual.add_nodes_from(graph.nodes)
    residual.add_edges_from(e for e in graph.edges if canonical_edge(*e) not in covered)
    return residual


def ordered_path(component):
    endpoints = [n for n, degree in component.degree() if degree == 1]
    if not endpoints:
        return []
    current, previous = min(endpoints), None
    path = [current]
    while True:
        neighbors = [n for n in component.neighbors(current) if n != previous]
        if not neighbors:
            return path
        following = min(neighbors)
        path.append(following)
        previous, current = current, following


def is_motif_bridge(path, memberships):
    if len(path) != 2:
        return False
    left, right = memberships.get(path[0], set()), memberships.get(path[-1], set())
    return bool(left and right and (left - right or right - left))


def extract_tokens(graph, dataset):
    """Legacy clique/ring/star/chain extraction used in the paper."""
    cliques = []
    for nodes in nx.find_cliques(graph):
        nodes = sorted(nodes)
        if len(nodes) >= 4:
            cliques.append(make_token("clique", nodes, combinations(nodes, 2)))
    cliques = unique_tokens(cliques)

    rings = []
    max_ring = 15 if dataset == "NCI109" else 6
    for cycle in nx.cycle_basis(graph):
        if 3 <= len(cycle) <= max_ring:
            edges = zip(cycle, cycle[1:] + cycle[:1])
            token = make_token("ring", cycle, edges)
            if not any(token.edges <= clique.edges for clique in cliques):
                rings.append(token)
    rings = unique_tokens(rings)
    covered = {edge for token in cliques + rings for edge in token.edges}

    residual = residual_graph(graph, covered)
    candidates = []
    for center in sorted(residual.nodes):
        neighbors = sorted(residual.neighbors(center))
        if len(neighbors) >= 3:
            leaves = neighbors[:9]
            candidates.append(make_token("star", [center] + leaves, ((center, v) for v in leaves)))
    stars, star_edges = [], set()
    for token in sorted(unique_tokens(candidates), key=lambda t: (-len(t.edges), -len(t.members), t.kind, sorted(t.members))):
        if not star_edges.intersection(token.edges):
            stars.append(token)
            star_edges.update(token.edges)
    covered.update(star_edges)

    anchors = cliques + rings + stars
    memberships = {}
    for i, token in enumerate(anchors):
        for node in token.members:
            memberships.setdefault(node, set()).add(i)
    residual = residual_graph(graph, covered)
    chains = []
    for nodes in nx.connected_components(residual):
        component = residual.subgraph(nodes).copy()
        if component.number_of_edges() == 0:
            continue
        if max(dict(component.degree()).values()) > 2:
            continue
        if component.number_of_edges() >= component.number_of_nodes():
            continue
        path = ordered_path(component)
        if is_motif_bridge(path, memberships):
            continue
        for start in range(0, len(path), 10):
            block = path[start:start + 10]
            if len(block) >= 2 and not is_motif_bridge(block, memberships):
                chains.append(make_token("chain", block, zip(block, block[1:])))

    tokens = anchors + unique_tokens(chains)
    represented = {node for token in tokens for node in token.members}
    tokens.extend(make_token("singleton", [node], []) for node in sorted(graph.nodes) if node not in represented)
    return tokens


def build_multilevel_graph(data, dataset):
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from((u, v) for u, v in data.edge_index.t().tolist() if u != v)
    tokens = extract_tokens(graph, dataset)
    if not tokens:
        raise ValueError("A graph must contain at least one original node")
    n = data.num_nodes
    context = n + len(tokens)
    memberships = {node: [] for node in range(n)}
    for slot, token in enumerate(tokens):
        for node in sorted(token.members):
            memberships[node].append(n + slot)

    token_edges = set()
    for owners in memberships.values():
        token_edges.update(combinations(sorted(set(owners)), 2))
    covered = {edge for token in tokens for edge in token.edges}
    for u, v in graph.edges:
        if canonical_edge(u, v) not in covered:
            for a in memberships[u]:
                for b in memberships[v]:
                    if a != b:
                        token_edges.add(canonical_edge(a, b))

    edges = data.edge_index.t().tolist()
    edge_types = [OO] * len(edges)
    for a, b in sorted(token_edges):
        edges.extend([[a, b], [b, a]])
        edge_types.extend([TT, TT])
    for node, owners in memberships.items():
        for owner in owners:
            edges.extend([[node, owner], [owner, node]])
            edge_types.extend([OT, TO])
    owners = list(range(n, context))
    edges.extend([owner, context] for owner in owners)
    edges.extend([context, owner] for owner in owners)
    edge_types.extend([TC] * len(tokens) + [CT] * len(tokens))

    token_x = torch.cat([data.x[sorted(t.members)].mean(0, keepdim=True) for t in tokens])
    x = torch.cat([data.x, token_x, token_x.mean(0, keepdim=True)])
    node_type = torch.tensor([ORIGINAL] * n + [TOKEN] * len(tokens) + [CONTEXT])
    return Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        node_type=node_type,
        y=data.y.clone(),
        num_nodes=context + 1,
    )
