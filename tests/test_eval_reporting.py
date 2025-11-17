from pathlib import Path

from training.eval import (
    EvaluationVariant,
    VariantMetrics,
    build_summary,
    load_synthetic_tasks_jsonl,
)


def make_metrics(name: str) -> list[VariantMetrics]:
    variant = EvaluationVariant(name=name, description=f"{name} desc", max_nodes=1)
    metrics = VariantMetrics(
        variant=variant,
        total_tasks=1,
        successes=1,
        success_rate=1.0,
        avg_programs_tested=1.0,
        details=(),
        novel_discoveries=0,
        novelty_candidates=0,
    )
    return [metrics]


def test_build_summary_without_surprise_section():
    metrics = make_metrics("base")
    summary = build_summary(
        "arc_dev",
        metrics=metrics,
        task_count=5,
        task_source=Path("arc_dev"),
    )
    assert summary["dataset"] == "arc_dev"
    assert summary["task_count"] == 5
    assert summary["task_source"].endswith("arc_dev")
    assert summary["results"][0]["variant"] == "base"
    assert summary["results"][0]["novelty_rate"] is None
    assert "surprise_results" not in summary


def test_build_summary_with_surprise_results():
    metrics = make_metrics("base")
    surprise_metrics = make_metrics("surprise")
    summary = build_summary(
        "arc_dev",
        metrics=metrics,
        task_count=5,
        surprise={
            "label": "surprise_tasks",
            "task_count": 2,
            "metrics": surprise_metrics,
            "source": Path("data/ood_surprise_tasks.jsonl"),
        },
    )
    assert "surprise_results" in summary
    surprise_entry = summary["surprise_results"]
    assert surprise_entry["dataset"] == "surprise_tasks"
    assert surprise_entry["task_count"] == 2
    assert surprise_entry["results"][0]["variant"] == "surprise"
    assert surprise_entry["task_source"].endswith("ood_surprise_tasks.jsonl")
    assert surprise_entry["results"][0]["novel_rule_discoveries"] == 0


def test_surprise_manifest_loads():
    manifest = Path("data/ood_surprise_tasks.jsonl")
    tasks = load_synthetic_tasks_jsonl(manifest)
    assert len(tasks) >= 1
    first = tasks[0]
    assert first.metadata.get("phase") == "SURPRISE"
    assert first.rule_trace
