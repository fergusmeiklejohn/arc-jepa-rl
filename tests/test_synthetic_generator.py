from arcgen import GeneratorConfig, SyntheticARCGenerator, SyntheticTask
from scripts.generate_dataset import summarise


def build_generator(seed: int = 7, **kwargs) -> SyntheticARCGenerator:
    config = GeneratorConfig(
        min_grid_size=5,
        max_grid_size=6,
        min_colors=3,
        max_colors=4,
        fill_probability=0.8,
        max_parameter_retries=10,
        max_task_retries=20,
    )
    return SyntheticARCGenerator(config, seed=seed, **kwargs)


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


def test_program_length_schedule_respected():
    generator = build_generator(
        seed=17,
        program_length_schedule={"sequential": {4: 1.0}},
    )

    task = generator.sample_task("sequential")
    assert len(task.rule_trace) == 4
    assert task.metadata["program_length"] == 4


def test_program_max_depth_limits_lengths():
    generator = build_generator(seed=19, max_program_length=2)

    tasks = [generator.sample_task("sequential") for _ in range(3)]
    assert all(len(task.rule_trace) <= 2 for task in tasks)


def test_summary_includes_program_histogram():
    generator = build_generator(
        seed=23,
        program_length_schedule={"sequential": {3: 1.0}},
    )
    tasks = [generator.sample_task("sequential") for _ in range(3)]

    summary = summarise(tasks)
    assert summary["program_length_histogram"] == {3: 3}
