from arcgen import Grid

from training.dsl.enumerator import InputVar, ProgramEnumerator
from training.dsl.metrics import description_length
from training.dsl.primitives import build_default_primitive_registry
from training.dsl.types import Grid as GridType
from training.solver.constraints import ConstraintChecker


def _program_by_name(name: str, *, max_nodes: int = 3):
    registry = build_default_primitive_registry()
    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=max_nodes,
    )
    for program in enumerator.enumerate():
        if program.root.primitive is not None and program.root.primitive.name == name:
            return program
    raise AssertionError(f"primitive {name} not found in enumerator output")


def test_description_length_penalises_rotations():
    identity = _program_by_name("identity")
    rotate = _program_by_name("rotate_180")

    assert description_length(identity) < description_length(rotate)


def test_constraint_checker_flags_palette_mismatch():
    checker = ConstraintChecker(allow_palette_mismatch=False)
    identity = _program_by_name("identity")
    context = Grid([[0, 1], [0, 1]])
    target = Grid([[0, 2], [0, 2]])  # introduces new color

    assert checker.pre_check(identity, context, target)


def test_constraint_checker_allows_color_change_program():
    checker = ConstraintChecker(allow_palette_mismatch=False)
    recolor = _program_by_name("recolor", max_nodes=4)
    context = Grid([[0, 1], [0, 1]])
    target = Grid([[0, 2], [0, 2]])

    # Program uses recolor, so palette mismatch should not prune.
    assert not checker.pre_check(recolor, context, target)

    candidate = Grid([[0, 2], [0, 2]])
    assert not checker.post_check(recolor, context, candidate, target)

    # Candidate introducing unseen colors should be rejected for color-neutral programs.
    identity = _program_by_name("identity")
    noisy = Grid([[0, 5], [0, 5]])
    assert checker.post_check(identity, context, noisy, target)
