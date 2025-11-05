from arcgen import Grid as ArcGrid

from training.dsl import (
    build_default_primitive_registry,
    ProgramEnumerator,
    ProgramInterpreter,
    NeuralGuidedScorer,
    InputVar,
    Grid,
)


def test_typed_enumerator_produces_grid_programs():
    registry = build_default_primitive_registry(color_constants=(0, 1))
    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", Grid)],
        target_type=Grid,
        max_nodes=3,
    )
    programs = list(enumerator.enumerate())

    assert programs
    mirror_programs = [prog for prog in programs if prog.root.primitive and prog.root.primitive.name == "mirror_x"]
    assert mirror_programs

    interpreter = ProgramInterpreter()
    input_grid = ArcGrid([[0, 1], [0, 0]])
    result_grid = interpreter.evaluate(mirror_programs[0], {"grid": input_grid})
    assert isinstance(result_grid, ArcGrid)
    assert result_grid.shape == input_grid.shape


def test_enumerator_handles_recolor_with_color_constants():
    registry = build_default_primitive_registry(color_constants=(0, 1))
    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", Grid)],
        target_type=Grid,
        max_nodes=3,
    )
    recolor_programs = [
        prog
        for prog in enumerator.enumerate()
        if prog.root.primitive and prog.root.primitive.name == "recolor"
    ]

    assert recolor_programs  # program uses grid + const colors

    interpreter = ProgramInterpreter()
    input_grid = ArcGrid([[0, 1], [1, 1]])
    changed_output = None
    for program in recolor_programs:
        output = interpreter.evaluate(program, {"grid": input_grid})
        assert isinstance(output, ArcGrid)
        if output.cells != input_grid.cells:
            changed_output = output
            break

    assert changed_output is not None


def test_neural_guided_scorer_wrapper():
    registry = build_default_primitive_registry()
    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", Grid)],
        target_type=Grid,
        max_nodes=2,
    )
    program = next(iter(enumerator.enumerate()))
    scorer = NeuralGuidedScorer(lambda prog: len(prog))
    result = scorer.score(program)
    assert result.score == len(program)
