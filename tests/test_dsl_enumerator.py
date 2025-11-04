from training.dsl import (
    build_default_primitive_registry,
    ProgramEnumerator,
    NeuralGuidedScorer,
)


def test_program_enumerator_generates_programs():
    registry = build_default_primitive_registry()
    enumerator = ProgramEnumerator(registry, max_depth=2, max_nodes=3)
    programs = list(enumerator.enumerate(["mirror_x"]))
    assert programs
    assert programs[0].nodes


def test_neural_guided_scorer_wrapper():
    registry = build_default_primitive_registry()
    enumerator = ProgramEnumerator(registry, max_depth=1)
    program = next(iter(enumerator.enumerate(["mirror_x"])))
    scorer = NeuralGuidedScorer(lambda prog: len(prog.nodes))
    result = scorer.score(program)
    assert result.score == len(program.nodes)
