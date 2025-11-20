import pytest
from arcgen import Grid

from training.dsl.enumerator import InputVar, ProgramEnumerator, ProgramInterpreter
from training.dsl.guide import GuidedBeamSearch, ProgramEncoder
from training.dsl.metrics import description_length
from training.dsl.primitives import build_default_primitive_registry
from training.dsl.types import Grid as GridType
from training.solver.constraints import ConstraintChecker
from training.solver import FewShotSolver
from training.solver.cache import EvaluationCache


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


def test_evaluation_cache_memoises_program_execution(monkeypatch):
    registry = build_default_primitive_registry()
    interpreter = ProgramInterpreter()
    program = _program_by_name("identity")
    grid = Grid([[1, 0], [0, 1]])

    calls = {"count": 0}
    original = interpreter.evaluate

    def wrapped(program_arg, inputs):
        calls["count"] += 1
        return original(program_arg, inputs)

    monkeypatch.setattr(interpreter, "evaluate", wrapped)
    solver = FewShotSolver(registry, interpreter=interpreter)

    first = solver._evaluate_with_cache(program, grid)
    second = solver._evaluate_with_cache(program, grid)

    assert calls["count"] == 1
    assert isinstance(first, Grid) and isinstance(second, Grid)
    assert first.cells == second.cells


def test_guided_beam_search_respects_constraints_and_mdl(monkeypatch):
    torch = pytest.importorskip("torch")
    registry = build_default_primitive_registry(color_constants=(0, 1, 2))
    program_encoder = ProgramEncoder(registry, embedding_dim=8)

    class ZeroLatent:
        def __call__(self, grid):
            return torch.zeros(8)

    class ZeroScorer(torch.nn.Module):
        def forward(self, features):
            return torch.tensor(0.0, device=features.device)

    beam = GuidedBeamSearch(
        registry,
        ZeroScorer(),
        program_encoder,
        ProgramInterpreter(),
        ZeroLatent(),
        beam_width=20,
        length_penalty=0.0,
        constraint_checker=ConstraintChecker(),
    )

    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=3,
    )

    context = Grid([[0, 1]])
    target = Grid([[0, 2]])
    cache = EvaluationCache()
    results = beam.search(
        torch.zeros(8),
        torch.zeros(8),
        enumerator,
        context,
        target,
        cache=cache,
        mdl_weight=0.5,
    )

    names = [prog.root.primitive.name for prog, _ in results if prog.root.primitive]
    assert "identity" not in names  # pruned by constraint checker
    assert any(name == "recolor" for name in names)

    symmetric = Grid([[1, 0], [0, 1]])
    simple_enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=1,
    )
    results_sym = beam.search(
        torch.zeros(8),
        torch.zeros(8),
        simple_enumerator,
        symmetric,
        symmetric,
        cache=cache,
        mdl_weight=0.5,
    )
    ordered_names = [prog.root.primitive.name for prog, _ in results_sym if prog.root.primitive]
    assert ordered_names[0] == "identity"
    assert "rotate_180" in ordered_names
    assert ordered_names.index("identity") < ordered_names.index("rotate_180")
