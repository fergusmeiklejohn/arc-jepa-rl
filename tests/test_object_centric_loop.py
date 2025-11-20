import json

import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid
from training.jepa import (
    ObjectCentricJEPAExperiment,
    InMemoryGridPairDataset,
    ManifestGridPairDataset,
    build_dummy_dataset,
)


def config_with_optimizer():
    return {
        "tokenizer": {
            "max_objects": 2,
            "max_color_features": 2,
        },
        "object_encoder": {
            "hidden_dim": 8,
            "num_embeddings": 16,
        },
        "optimizer": {
            "lr": 1e-3,
        },
        "data": {
            "context_window": 3,
        },
    }


def build_sample_batch(context_length: int = 3):
    def make_grid() -> Grid:
        return Grid(
            [
                [0, 1, 1],
                [0, 0, 2],
                [0, 0, 2],
            ]
        )

    context = [tuple(make_grid() for _ in range(context_length)) for _ in range(2)]
    target = [make_grid(), make_grid()]
    return context, target


def test_experiment_train_step_runs_and_returns_loss():
    config = config_with_optimizer()
    experiment = ObjectCentricJEPAExperiment(config)

    context, target = build_sample_batch(experiment.context_length)
    result = experiment.train_step(context, target)

    assert isinstance(result.loss, float)
    assert result.encoded_context.shape == (len(target), config["object_encoder"]["hidden_dim"])
    assert result.encoded_target.shape == (len(target), config["object_encoder"]["hidden_dim"])


def test_train_epoch_returns_average_loss():
    config = config_with_optimizer()
    experiment = ObjectCentricJEPAExperiment(config)
    dataset = build_dummy_dataset(num_batches=3, context_length=experiment.context_length)

    loss = experiment.train_epoch(dataset)
    assert isinstance(loss, float)


def test_train_over_epochs_accumulates_losses():
    config = config_with_optimizer()
    experiment = ObjectCentricJEPAExperiment(config)

    dataset = InMemoryGridPairDataset([build_sample_batch(experiment.context_length)])
    losses = experiment.train(dataset, epochs=2)

    assert len(losses) == 2
    assert experiment.queue.get_negatives().shape[0] > 0


def test_experiment_train_epoch_with_manifest(tmp_path):
    config = config_with_optimizer()
    experiment = ObjectCentricJEPAExperiment(config)

    frames = [
        [
            [0, 0],
            [0, 1],
        ],
        [
            [0, 0],
            [0, 2],
        ],
        [
            [0, 0],
            [0, 3],
        ],
        [
            [0, 0],
            [0, 4],
        ],
    ]
    manifest_path = tmp_path / "tiny_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"id": "seq-1", "frames": frames}, handle)
        handle.write("\n")

    dataset = ManifestGridPairDataset(
        manifest_path,
        batch_size=1,
        shuffle=False,
        context_window=experiment.context_length,
        target_offset=1,
    )

    loss = experiment.train_epoch(dataset)
    assert isinstance(loss, float)


def test_target_encoder_ema_and_stop_gradient():
    config = config_with_optimizer()
    config["loss"] = {"use_target_encoder": True, "target_ema_decay": 0.5}
    experiment = ObjectCentricJEPAExperiment(config)

    assert experiment._use_target_encoder
    assert experiment._target_encoder is not None

    context, target = build_sample_batch(experiment.context_length)
    before = [param.detach().clone() for param in experiment._target_encoder.parameters()]

    result = experiment.train_step(context, target)
    assert isinstance(result.loss, float)

    after = [param.detach().clone() for param in experiment._target_encoder.parameters()]
    assert any(not torch.allclose(prev, curr) for prev, curr in zip(before, after))
    assert all(not param.requires_grad for param in experiment._target_encoder.parameters())


def test_amp_flag_falls_back_on_cpu():
    config = config_with_optimizer()
    config["training"] = {"amp": True}

    with pytest.warns(RuntimeWarning):
        experiment = ObjectCentricJEPAExperiment(config, device="cpu")

    assert not experiment._amp_enabled

    context, target = build_sample_batch(experiment.context_length)
    result = experiment.train_step(context, target)
    assert isinstance(result.loss, float)


def test_embedding_metrics_emitted_when_interval_met():
    config = config_with_optimizer()
    config["diagnostics"] = {
        "embedding_metrics": {
            "interval": 1,
            "max_samples": 32,
        }
    }
    experiment = ObjectCentricJEPAExperiment(config)

    context, target = build_sample_batch(experiment.context_length)
    experiment.train_step(context, target)
    events = experiment.consume_embedding_metrics()
    assert events, "expected embedding diagnostics events to be emitted"
    event = events[0]
    assert "context" in event and "target" in event
    assert event["context"]["variance"] >= 0.0


def test_sigreg_loss_metrics_emitted_when_enabled():
    config = config_with_optimizer()
    config["sigreg"] = {"weight": 0.1, "num_slices": 3, "num_points": 5}
    experiment = ObjectCentricJEPAExperiment(config)

    context, target = build_sample_batch(experiment.context_length)
    experiment.train_step(context, target)

    events = experiment.consume_loss_metrics()
    assert events, "expected per-step loss metrics"
    assert events[-1]["sigreg"] >= 0.0
    assert events[-1]["info_nce"] > 0.0


def test_queue_updates_each_microbatch_with_grad_accum():
    config = config_with_optimizer()
    config["training"] = {"grad_accum_steps": 2}
    config["loss"] = {"queue_size": 8}
    experiment = ObjectCentricJEPAExperiment(config)

    # Three batches of size 2 with grad accumulation = 2 should enqueue 6 projections.
    dataset = build_dummy_dataset(num_batches=3, context_length=experiment.context_length, batch_size=2)
    experiment.train_epoch(dataset)

    negatives = experiment.queue.get_negatives()
    assert negatives.shape[0] == 6


def test_queue_enqueue_happens_after_optimizer_step(monkeypatch):
    config = config_with_optimizer()
    experiment = ObjectCentricJEPAExperiment(config)

    calls = []
    original_step = experiment._step_optimizer
    original_enqueue = experiment.queue.enqueue

    def step_wrapper(loss):
        calls.append("step")
        return original_step(loss)

    def enqueue_wrapper(tensor):
        calls.append("enqueue")
        return original_enqueue(tensor)

    monkeypatch.setattr(experiment, "_step_optimizer", step_wrapper)
    monkeypatch.setattr(experiment.queue, "enqueue", enqueue_wrapper)

    context, target = build_sample_batch(experiment.context_length)
    experiment.train_step(context, target)

    assert calls == ["step", "enqueue"]


def test_queue_commit_deferred_until_accumulated_step(monkeypatch):
    config = config_with_optimizer()
    config["training"] = {"grad_accum_steps": 2}
    config["loss"] = {"queue_size": 16}
    experiment = ObjectCentricJEPAExperiment(config)

    events = []
    original_step = experiment._step_optimizer
    original_enqueue = experiment.queue.enqueue

    def step_wrapper(loss):
        events.append("step")
        return original_step(loss)

    def enqueue_wrapper(tensor):
        events.append(f"enqueue:{tensor.shape[0]}")
        return original_enqueue(tensor)

    monkeypatch.setattr(experiment, "_step_optimizer", step_wrapper)
    monkeypatch.setattr(experiment.queue, "enqueue", enqueue_wrapper)

    dataset = build_dummy_dataset(num_batches=3, context_length=experiment.context_length, batch_size=2)
    experiment.train_epoch(dataset)

    assert events == ["step", "enqueue:4", "step", "enqueue:2"]
    assert experiment.queue.get_negatives().shape[0] == 6
