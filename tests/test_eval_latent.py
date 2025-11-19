from arcgen import Grid

from training.eval import LatentDistanceTracker


def _grid_sum(grid: Grid) -> float:
    return float(sum(sum(row) for row in grid.cells))


def _abs_distance(a, b) -> float:
    return abs(float(a) - float(b))


def test_latent_distance_tracker_records_distances():
    tracker = LatentDistanceTracker(_grid_sum, _abs_distance)
    start = Grid([[0, 1], [0, 0]])
    target = Grid([[1, 1], [0, 0]])
    prediction = Grid([[1, 1], [0, 0]])

    record = tracker.record("train", 0, start=start, target=target, prediction=prediction)
    assert record is not None
    assert record.example_kind == "train"
    assert record.index == 0
    assert record.start_distance == 1.0
    assert record.final_distance == 0.0


def test_latent_distance_tracker_skips_missing_targets():
    tracker = LatentDistanceTracker(_grid_sum, _abs_distance)
    start = Grid([[0]])
    assert tracker.record("test", 0, start=start, target=None, prediction=None) is None
