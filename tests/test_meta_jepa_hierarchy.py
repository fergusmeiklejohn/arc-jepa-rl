import pytest

torch = pytest.importorskip("torch")

from training.meta_jepa.hierarchy import build_rule_family_hierarchy


def test_hierarchy_clusters_merge_similar_embeddings():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.8, -0.2],
        ]
    )
    family_ids = ["a", "b", "c", "d"]

    hierarchy = build_rule_family_hierarchy(family_ids, embeddings, levels=(4, 2, 1))

    level_two = hierarchy.levels[2]
    cluster_a = level_two.assignments["a"]
    cluster_b = level_two.assignments["b"]
    cluster_c = level_two.assignments["c"]

    assert cluster_a == cluster_b
    assert cluster_a != cluster_c
    assert level_two.centroids.shape == (2, 2)
    assert torch.allclose(level_two.centroids.norm(dim=-1), torch.ones(2), atol=1e-5)
