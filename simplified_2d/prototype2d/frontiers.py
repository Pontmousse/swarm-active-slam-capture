"""Frontier points on the dense target boundary: observed ids adjacent to unobserved."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set


def compute_map_frontier(
    known_ids: Iterable[int],
    dense_adjacency: Dict[int, List[int]],
    dense_id_set: Set[int],
) -> List[int]:
    """
    Return dense point ids that are in known_ids and have at least one neighbor
    not in known_ids (boundary between mapped and unmapped region along topology).
    """
    known = set(int(x) for x in known_ids)
    frontier: Set[int] = set()
    for pid in known:
        if pid not in dense_id_set:
            continue
        for nb in dense_adjacency.get(pid, []):
            if nb not in known:
                frontier.add(pid)
                break
    return sorted(frontier)
