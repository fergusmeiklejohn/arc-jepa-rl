from training.jepa.ablation import AblationVariant, apply_variant, build_ablation_variants, load_base_config


def test_ablation_variants_cover_matrix():
    variants = build_ablation_variants()
    names = {variant.name for variant in variants}
    expected = {
        "baseline_infonce",
        "vq_only",
        "vq_relational",
        "vq_relational_invariance",
        "vq_relational_invariance_sigreg",
    }
    assert expected.issubset(names)


def test_variant_overrides_are_applied(tmp_path):
    base = {
        "object_encoder": {"num_embeddings": 128, "relational": True},
        "sigreg": {"weight": 0.1},
    }
    variant = AblationVariant(
        name="baseline_infonce",
        description="",
        overrides={"object_encoder": {"num_embeddings": None, "relational": False}, "sigreg": {"weight": 0.0}},
    )
    merged = apply_variant(base, variant)
    assert merged["object_encoder"]["num_embeddings"] is None
    assert merged["object_encoder"]["relational"] is False
    assert merged["sigreg"]["weight"] == 0.0


def test_load_base_config_reads_yaml(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text("object_encoder:\n  hidden_dim: 16\n", encoding="utf-8")
    cfg = load_base_config(yaml_path)
    assert cfg["object_encoder"]["hidden_dim"] == 16
