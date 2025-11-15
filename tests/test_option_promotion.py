from arcgen import Grid
from arcgen.primitives import PrimitiveRegistry as ArcPrimitiveRegistry
from envs import Option

from training.dsl.primitives import PrimitiveRegistry as DSLPrimitiveRegistry
from training.options import DiscoveredOption, promote_discovered_option


def make_option(name: str, delta: int) -> Option:
    def apply(grid: Grid) -> Grid:
        return Grid([[value + delta for value in row] for row in grid.cells])

    return Option(name=name, apply=apply, description=f"add {delta}")


def build_discovered() -> DiscoveredOption:
    add_one = make_option("add_one", 1)
    add_two = make_option("add_two", 2)

    base = Grid([[0, 1], [2, 3]])
    after = add_two.apply(add_one.apply(base))

    return DiscoveredOption(
        name="auto_add",
        sequence=(add_one, add_two),
        sequence_names=(add_one.name, add_two.name),
        support=3,
        success_rate=0.8,
        avg_reward=0.5,
        occurrences=((base, after),),
    )


def test_promote_discovered_option_registers_new_primitives():
    arc_registry = ArcPrimitiveRegistry()
    dsl_registry = DSLPrimitiveRegistry()
    discovered = build_discovered()

    name = promote_discovered_option(discovered, dsl_registry, arc_registry=arc_registry)

    arc_spec = arc_registry.get(name)
    dsl_primitive = dsl_registry.get(name)

    grid = Grid([[3, 4], [5, 6]])
    expected = discovered.apply(grid)

    assert arc_spec.apply(grid).cells == expected.cells
    assert dsl_primitive.implementation(grid).cells == expected.cells


def test_promote_discovered_option_is_idempotent():
    arc_registry = ArcPrimitiveRegistry()
    dsl_registry = DSLPrimitiveRegistry()
    discovered = build_discovered()

    promote_discovered_option(discovered, dsl_registry, arc_registry=arc_registry)
    # Should not raise when called again even though primitives already exist.
    promote_discovered_option(discovered, dsl_registry, arc_registry=arc_registry)

    assert len(arc_registry.list()) == 1
    assert len(dsl_registry.list()) == 1
