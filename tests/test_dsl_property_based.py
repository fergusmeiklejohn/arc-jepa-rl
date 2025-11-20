from collections import Counter

from hypothesis import assume, given, settings, strategies as st

from arcgen import Grid
from training.dsl.enumerator import InputVar, ProgramEnumerator, ProgramInterpreter
from training.dsl.primitives import build_default_primitive_registry
from training.dsl.types import Bool, Color, Grid as GridType, GridValue, is_value_of_type


@st.composite
def grids_with_objects(draw):
    height = draw(st.integers(min_value=1, max_value=4))
    width = draw(st.integers(min_value=1, max_value=4))
    palette = st.integers(min_value=0, max_value=5)
    rows = draw(
        st.lists(
            st.lists(palette, min_size=width, max_size=width),
            min_size=height,
            max_size=height,
        )
    )
    if all(cell == 0 for row in rows for cell in row):
        y = draw(st.integers(min_value=0, max_value=height - 1))
        x = draw(st.integers(min_value=0, max_value=width - 1))
        rows[y][x] = draw(st.integers(min_value=1, max_value=5))
    return Grid(rows)


def _area(grid: Grid) -> int:
    return len(grid.cells) * (len(grid.cells[0]) if grid.cells else 0)


def _palette_counts(grid: Grid) -> Counter[int]:
    return Counter(value for row in grid.cells for value in row)


def _uses_primitives(program, names: set[str]) -> bool:
    return any(expr.primitive and expr.primitive.name in names for expr in program.traverse())


def _safe_programs() -> list:
    registry = build_default_primitive_registry()
    input_var = InputVar("grid", GridType)
    target_types = [GridType, GridValue, Color, Bool]
    programs = []
    for target in target_types:
        enumerator = ProgramEnumerator(registry, inputs=[input_var], target_type=target, max_nodes=3)
        programs.extend(
            program
            for program in enumerator.enumerate()
            if not _uses_primitives(program, {"flood_fill"})
        )
    return programs


PROGRAM_SAMPLES = _safe_programs()
PROGRAM_STRATEGY = st.sampled_from(PROGRAM_SAMPLES)


@settings(max_examples=30, deadline=500)
@given(grid=grids_with_objects())
def test_geometric_primitives_preserve_area_and_palette(grid: Grid):
    registry = build_default_primitive_registry()
    transform_names = ["identity", "mirror_x", "mirror_y", "rotate_cw", "rotate_ccw", "rotate_180"]
    baseline_area = _area(grid)
    baseline_counts = _palette_counts(grid)

    for name in transform_names:
        transformed = registry.get(name)(grid)
        assert _area(transformed) == baseline_area
        assert _palette_counts(transformed) == baseline_counts


@settings(max_examples=30, deadline=500)
@given(
    grid=grids_with_objects(),
    source=st.integers(min_value=0, max_value=5),
    target=st.integers(min_value=0, max_value=5),
)
def test_recolor_changes_only_requested_color(grid: Grid, source: int, target: int):
    registry = build_default_primitive_registry()
    recolor = registry.get("recolor")

    before = _palette_counts(grid)
    result = recolor(grid, source, target)
    after = _palette_counts(result)

    if source == target or before.get(source, 0) == 0:
        assert after == before
        return

    for color, count in before.items():
        if color not in (source, target):
            assert after[color] == count
    moved = before.get(source, 0)
    assert after.get(source, 0) == 0
    assert after.get(target, 0) == before.get(target, 0) + moved


@settings(max_examples=25, deadline=500)
@given(grid=grids_with_objects(), program=PROGRAM_STRATEGY)
def test_enumerated_programs_return_declared_type(grid: Grid, program):
    assume(PROGRAM_SAMPLES)
    interpreter = ProgramInterpreter()
    result = interpreter.evaluate(program, {"grid": grid})
    assert is_value_of_type(result, program.root.type)
