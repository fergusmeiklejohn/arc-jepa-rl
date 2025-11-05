from arcgen import GeneratorConfig, SyntheticARCGenerator, SyntheticTask


def build_generator(seed: int = 7) -> SyntheticARCGenerator:
    config = GeneratorConfig(
        min_grid_size=5,
        max_grid_size=6,
        min_colors=3,
        max_colors=4,
        fill_probability=0.8,
        max_parameter_retries=10,
        max_task_retries=20,
    )
    return SyntheticARCGenerator(config, seed=seed)


def test_atomic_generation_produces_single_rule():
    generator = build_generator()
    task = generator.sample_task("atomic")

    assert isinstance(task, SyntheticTask)
    assert task.phase == "I"
    assert len(task.rule_trace) == 1
    assert task.input_grid.cells != task.output_grid.cells
    assert task.metadata["program_length"] == 1


def test_sequential_generation_has_multiple_steps():
    generator = build_generator(seed=11)
    task = generator.sample_task("sequential")

    assert task.phase == "II"
    assert len(task.rule_trace) >= 2
    assert task.metadata["program_length"] == len(task.rule_trace)
    assert all(hasattr(step, "primitive") for step in task.rule_trace)


def test_jsonl_export_round_trip(tmp_path):
    generator = build_generator(seed=13)
    tasks = [generator.sample_task("atomic") for _ in range(2)]

    destination = tmp_path / "synthetic.jsonl"
    SyntheticARCGenerator.export_jsonl(destination, tasks)

    assert destination.exists()
    contents = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 2
    assert all('"rule_trace"' in line for line in contents)
