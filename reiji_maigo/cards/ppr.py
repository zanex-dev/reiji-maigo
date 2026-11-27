"""Pure standard-library Personalized PageRank (PPR) for multi-hop card diffusion."""

from __future__ import annotations

from collections.abc import Mapping


def personalized_pagerank(
    adj: dict[str, dict[str, float]],
    seeds: dict[str, float],
    *,
    damping: float = 0.85,
    iters: int = 12,
) -> dict[str, float]:
    """Compute Personalized PageRank scores over a weighted directed graph.

    Args:
        adj: Adjacency map ``{src: {dst: weight}}`` with non-negative edge weights.
        seeds: Personalization seed weights ``{node: weight}`` with non-negative weights.
        damping: Random-walk continuation factor in ``[0.0, 1.0]``.
        iters: Number of power-iteration steps (non-negative integer).

    Returns:
        Mapping of reachable nodes with positive score sorted descending by score.
    """
    if isinstance(damping, bool) or not isinstance(damping, (int, float)):
        raise TypeError("damping must be a float in [0.0, 1.0]")
    if not (0.0 <= float(damping) <= 1.0):
        raise ValueError("damping must be within [0.0, 1.0]")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 0:
        raise ValueError("iters must be a non-negative int")

    alpha = float(damping)

    # Validate and normalize personalization vector.
    clean_seeds: dict[str, float] = {}
    seed_total = 0.0
    for node, raw_weight in seeds.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"Seed weight for {node!r} must be numeric")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"Seed weight for {node!r} must be non-negative")
        if weight > 0.0:
            clean_seeds[node] = weight
            seed_total += weight

    if seed_total <= 0.0:
        return {}

    personalization: dict[str, float] = {
        node: weight / seed_total for node, weight in clean_seeds.items()
    }

    # Collect full node set and validate positive outgoing transitions.
    nodes: set[str] = set(personalization.keys())
    pos_adj: dict[str, list[tuple[str, float]]] = {}
    out_sums: dict[str, float] = {}

    for src, neighbors in adj.items():
        nodes.add(src)
        if not isinstance(neighbors, Mapping):
            raise TypeError(f"Adjacency entry for {src!r} must be a mapping")
        edges: list[tuple[str, float]] = []
        total_w = 0.0
        for dst, raw_w in neighbors.items():
            nodes.add(dst)
            if isinstance(raw_w, bool) or not isinstance(raw_w, (int, float)):
                raise TypeError(f"Edge weight ({src!r} -> {dst!r}) must be numeric")
            w = float(raw_w)
            if w < 0.0:
                raise ValueError(f"Edge weight ({src!r} -> {dst!r}) must be non-negative")
            if w > 0.0:
                edges.append((dst, w))
                total_w += w
        if total_w > 0.0:
            pos_adj[src] = edges
            out_sums[src] = total_w

    dangling_nodes: list[str] = [node for node in nodes if node not in out_sums]

    rank: dict[str, float] = {node: personalization.get(node, 0.0) for node in nodes}

    for _ in range(iters):
        next_rank: dict[str, float] = {node: 0.0 for node in nodes}

        # Teleportation + dangling-node mass redistribution back to personalization seeds.
        dangling_mass = 0.0
        for node in dangling_nodes:
            dangling_mass += rank[node]

        seed_injection = (1.0 - alpha) + alpha * dangling_mass
        for seed_node, p_val in personalization.items():
            next_rank[seed_node] += seed_injection * p_val

        # Transition step along weighted outgoing edges.
        for src, edges in pos_adj.items():
            src_rank = rank[src]
            if src_rank == 0.0:
                continue
            factor = alpha * src_rank / out_sums[src]
            for dst, weight in edges:
                next_rank[dst] += factor * weight

        rank = next_rank

    positive_items = [(node, score) for node, score in rank.items() if score > 0.0]
    positive_items.sort(key=lambda item: (-item[1], item[0]))
    return dict(positive_items)
