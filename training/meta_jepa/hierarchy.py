"""Hierarchical clustering over rule-family embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


def _ensure_torch() -> None:
    if torch is None or F is None:  # pragma: no cover - defensive
        raise RuntimeError("PyTorch is required for Meta-JEPA hierarchy utilities")


@dataclass(frozen=True)
class ClusterLevel:
    """Cluster assignments and centroids for a specific hierarchy depth."""

    clusters: int
    assignments: Mapping[str, int]
    centroids: "torch.Tensor"


@dataclass
class RuleFamilyHierarchy:
    """Stores multi-level clustering information for rule families."""

    levels: Mapping[int, ClusterLevel]

    def available_levels(self) -> Tuple[int, ...]:
        return tuple(sorted(self.levels.keys(), reverse=True))

    def cluster_id(self, family_id: str, clusters: int) -> int:
        if clusters not in self.levels:
            raise KeyError(f"cluster level {clusters} not available")
        assignments = self.levels[clusters].assignments
        if family_id not in assignments:
            raise KeyError(f"family '{family_id}' missing from hierarchy")
        return assignments[family_id]

    def centroids(self, clusters: int) -> "torch.Tensor":
        if clusters not in self.levels:
            raise KeyError(f"cluster level {clusters} not available")
        return self.levels[clusters].centroids


def build_rule_family_hierarchy(
    family_ids: Sequence[str],
    embeddings: "torch.Tensor",
    *,
    levels: Sequence[int],
) -> RuleFamilyHierarchy:
    """Build a clustering hierarchy using cosine similarity between embeddings."""

    _ensure_torch()
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (families, dim)")
    if len(family_ids) != embeddings.shape[0]:
        raise ValueError("family_ids must align with embeddings")
    if not levels:
        raise ValueError("levels must contain at least one entry")

    num_families = embeddings.shape[0]
    device = embeddings.device
    normalized = F.normalize(embeddings.float(), dim=-1)

    requested = {max(1, min(num_families, int(level))) for level in levels}
    requested.add(1)
    requested.add(num_families)

    clusters = [set([idx]) for idx in range(num_families)]
    centroids = [normalized[idx].detach().clone() for idx in range(num_families)]
    snapshots: Dict[int, ClusterLevel] = {}

    def snapshot() -> None:
        assignments = {
            family_ids[member]: cluster_idx
            for cluster_idx, members in enumerate(clusters)
            for member in members
        }
        centroid_tensor = torch.stack(centroids, dim=0) if centroids else normalized.new_zeros((0, normalized.shape[1]))
        centroid_tensor = centroid_tensor.detach().clone()
        snapshots[len(clusters)] = ClusterLevel(
            clusters=len(clusters),
            assignments=assignments,
            centroids=centroid_tensor,
        )

    snapshot()
    while len(clusters) > 1:
        best_score = float("-inf")
        best_pair: Tuple[int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = float(torch.dot(centroids[i], centroids[j]))
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        members = clusters[i] | clusters[j]
        member_indices = sorted(members)
        new_centroid = F.normalize(normalized[member_indices].mean(dim=0), dim=0)
        clusters[i] = members
        centroids[i] = new_centroid.detach().clone()
        del clusters[j]
        del centroids[j]
        snapshot()

    available_levels = {}
    for level in requested:
        if level not in snapshots:
            raise ValueError(f"unable to build hierarchy level {level}")
        available_levels[level] = snapshots[level]

    return RuleFamilyHierarchy(levels=available_levels)


__all__ = [
    "ClusterLevel",
    "RuleFamilyHierarchy",
    "build_rule_family_hierarchy",
]
