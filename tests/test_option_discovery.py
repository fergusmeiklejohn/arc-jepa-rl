import pytest

from arcgen import Grid
from envs import Option, make_primitive_option
from training.dsl.primitives import build_default_primitive_registry
from training.options import (
    OptionApplication,
    OptionEpisode,
    discover_option_sequences,
    promote_discovered_option,
)


def build_composite_episode(option_a: Option, option_b: Option, grid: Grid, *, success: bool = True) -> OptionEpisode:
    mid = option_a.apply(grid)
    final = option_b.apply(mid)
    return OptionEpisode(
        steps=(
            OptionApplication(option=option_a, before=grid, after=mid, reward=0.2, success=False),
            OptionApplication(option=option_b, before=mid, after=final, reward=1.0, success=success),
        )
    )


def test_discover_composite_option_and_promote():
    mirror_x = make_primitive_option("mirror_x")
    mirror_y = make_primitive_option("mirror_y")

    grid1 = Grid([[0, 1], [2, 3]])
    grid2 = Grid([[1, 0], [3, 2]])

    episode1 = build_composite_episode(mirror_x, mirror_y, grid1)
    episode2 = build_composite_episode(mirror_x, mirror_y, grid2)

    discovered = discover_option_sequences(
        [episode1, episode2],
        max_sequence_length=2,
        min_support=2,
    )
    assert discovered, "expected at least one discovered option"
    option = discovered[0]

    assert option.sequence_names == ("mirror_x", "mirror_y")
    assert option.support == 2
    assert option.success_rate == pytest.approx(1.0)

    registry = build_default_primitive_registry()
    primitive_name = promote_discovered_option(option, registry)

    promoted = registry.get(primitive_name)
    result = promoted.implementation(grid1)
    target = mirror_y.apply(mirror_x.apply(grid1))
    assert result.cells == target.cells


def test_discovery_respects_shape_invariant():
    def shrink(grid: Grid) -> Grid:
        return Grid([[grid.cells[0][0]]])

    shrink_option = Option(name="shrink", apply=shrink, description="reduce to 1x1")

    grid = Grid([[0, 1], [2, 3]])
    episode = OptionEpisode(
        steps=(OptionApplication(option=shrink_option, before=grid, after=shrink_option.apply(grid), reward=0.5, success=True),)
    )

    discovered = discover_option_sequences(
        [episode],
        min_support=1,
        allow_singleton=True,
    )
    assert not discovered, "shape-changing option should be filtered out"


def test_discovery_requires_success_rate_threshold():
    mirror_x = make_primitive_option("mirror_x")
    mirror_y = make_primitive_option("mirror_y")

    grid1 = Grid([[0, 1], [2, 3]])
    grid2 = Grid([[1, 0], [3, 2]])

    episode1 = build_composite_episode(mirror_x, mirror_y, grid1, success=False)
    episode2 = build_composite_episode(mirror_x, mirror_y, grid2, success=False)

    discovered = discover_option_sequences(
        [episode1, episode2],
        max_sequence_length=2,
        min_support=2,
        min_success_rate=0.5,
    )
    assert not discovered, "low success rate sequences should be ignored"

