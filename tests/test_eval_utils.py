import json

from arcgen import Grid
from training.eval import ArcExample, ArcTask, load_arc_dev_tasks


def test_load_arc_dev_tasks_parses_directory(tmp_path):
    root = tmp_path / "arc_dev"
    root.mkdir()

    task_data = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
        ],
        "test": [
            {"input": [[1]], "output": [[2]]},
            {"input": [[2]]},
        ],
    }
    (root / "abc123.json").write_text(json.dumps(task_data), encoding="utf-8")

    tasks = load_arc_dev_tasks(root)
    assert len(tasks) == 1

    task = tasks[0]
    assert isinstance(task, ArcTask)
    assert task.task_id == "abc123"
    assert len(task.train_examples) == 1
    assert isinstance(task.train_examples[0], ArcExample)
    assert task.train_examples[0].output_grid.cells == ((3, 2), (1, 0))
    assert len(task.test_examples) == 2
    assert task.test_examples[0].output_grid == Grid([[2]])
    assert task.test_examples[1].output_grid is None
