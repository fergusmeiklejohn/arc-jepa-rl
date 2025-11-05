from arcgen import Grid
from envs import make_primitive_option

from training.dsl.primitives import build_default_primitive_registry
from training.options import OptionApplication, OptionEpisode
from training.solver import FewShotSolver, augment_registry_with_options


def build_episode(option_a, option_b, grid):
    mid = option_a.apply(grid)
    final = option_b.apply(mid)
    return OptionEpisode(
        steps=(
            OptionApplication(option=option_a, before=grid, after=mid, reward=0.3, success=False),
            OptionApplication(option=option_b, before=mid, after=final, reward=1.0, success=True),
        )
    )


def test_few_shot_solver_uses_discovered_primitives():
    mirror_x = make_primitive_option("mirror_x")
    shift_right = make_primitive_option("translate", dx=1, dy=0, fill=0)

    grid1 = Grid([[0, 1, 2], [3, 4, 5]])
    grid2 = Grid([[5, 4, 0], [1, 2, 3]])

    final1 = shift_right.apply(mirror_x.apply(grid1))
    final2 = shift_right.apply(mirror_x.apply(grid2))

    episodes = [
        build_episode(mirror_x, shift_right, grid1),
        build_episode(mirror_x, shift_right, grid2),
    ]

    registry = build_default_primitive_registry()
    solver = FewShotSolver(registry)

    examples = [(grid1, final1), (grid2, final2)]
    result = solver.solve(examples, max_nodes=1)
    assert not result.solved(), "base registry should not solve with single primitive"

    augment_registry_with_options(registry, episodes, min_support=2, max_sequence_length=2)

    result = solver.solve(examples, max_nodes=1)
    assert result.solved(), "solver should find program using promoted primitive"
    assert result.program is not None
    assert result.program.root.primitive is not None
    assert result.program.root.primitive.name.startswith("auto_")
    output = solver.interpreter.evaluate(result.program, {"grid": grid1})
    assert output.cells == final1.cells

