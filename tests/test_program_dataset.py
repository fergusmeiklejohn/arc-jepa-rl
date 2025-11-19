import pytest

torch = pytest.importorskip("torch")

from training.jepa import ObjectTokenizerConfig, ProgramTripleDataset, collate_program_triples


def _dataset(tmp_max_length: int = 4) -> ProgramTripleDataset:
    config = ObjectTokenizerConfig(max_objects=3, max_color_features=2)
    return ProgramTripleDataset(
        "tests/data/program_triples_tiny.jsonl",
        tokenizer_config=config,
        max_program_length=tmp_max_length,
    )


def test_program_triple_dataset_loads_records():
    dataset = _dataset()
    assert len(dataset) == 2

    sample = dataset[0]
    assert sample.input_features.shape[0] == 3
    assert sample.program_ids.shape[0] == dataset.max_program_length
    assert sample.program_mask.sum() == 1
    assert dataset.vocab_size >= 2


def test_collate_program_triples_stacks_batches():
    dataset = _dataset()
    batch = collate_program_triples([dataset[0], dataset[1]])
    assert batch.input_features.shape[0] == 2
    assert batch.program_ids.shape[1] == dataset.max_program_length
    assert batch.program_mask.dtype == torch.float32
