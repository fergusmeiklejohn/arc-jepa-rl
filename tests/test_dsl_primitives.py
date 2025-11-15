import pytest

from arcgen import Grid

from training.dsl.primitives import build_default_primitive_registry


@pytest.fixture()
def registry():
    return build_default_primitive_registry()


def test_flood_fill_updates_region(registry):
    flood_fill = registry.get("flood_fill")
    grid = Grid([[1, 1, 2], [1, 0, 2]])
    filled = flood_fill.implementation(grid, 0, 0, 9)
    assert filled.cells[0][0] == 9
    assert filled.cells[1][1] == 0  # untouched cell


def test_connected_components_and_area(registry):
    components = registry.get("connected_components")
    shapes = components.implementation(Grid([[1, 1], [0, 2]]))

    length = registry.get("shape_list_len").implementation(shapes)
    assert length == 2

    first_shape = registry.get("shape_list_get").implementation(shapes, 0)
    area = registry.get("shape_area").implementation(first_shape)
    assert area == 2


def test_shape_bbox_and_centroid(registry):
    grid = Grid([[0, 3, 0], [0, 3, 3], [0, 0, 0]])
    shapes = registry.get("connected_components").implementation(grid)
    shape = registry.get("shape_list_get").implementation(shapes, 0)

    bbox = registry.get("shape_bbox").implementation(shape, 0)
    assert bbox.cells == ((3, 0), (3, 3))

    centroid = registry.get("shape_centroid").implementation(shape)
    y = registry.get("position_y").implementation(centroid)
    x = registry.get("position_x").implementation(centroid)
    assert y >= 0 and x >= 0


def test_components_filters(registry):
    grid = Grid([[1, 1, 1], [2, 0, 0]])
    shapes = registry.get("connected_components").implementation(grid)

    filter_color = registry.get("components_filter_by_color")
    filtered = filter_color.implementation(shapes, 1)
    assert registry.get("shape_list_len").implementation(filtered) == 1

    filter_area = registry.get("components_filter_by_area")
    large = filter_area.implementation(shapes, 3)
    assert registry.get("shape_list_len").implementation(large) == 1


def test_map_to_subgrids_and_select(registry):
    grid = Grid([[5, 5], [0, 4]])
    shapes = registry.get("connected_components").implementation(grid)
    grid_list = registry.get("components_map_to_subgrids").implementation(shapes, 0)
    assert registry.get("grid_list_len").implementation(grid_list) == 2

    first = registry.get("grid_list_get").implementation(grid_list, 0)
    assert first.shape == (1, 2) or first.shape == (2, 1)


def test_fold_overlay_recreates_original_colors(registry):
    grid = Grid([[1, 0], [0, 2]])
    shapes = registry.get("connected_components").implementation(grid)
    blank = Grid([[0, 0], [0, 0]])
    folded = registry.get("components_fold_overlay").implementation(shapes, blank)
    assert folded.cells == grid.cells


def test_if_then_else_uses_condition(registry):
    grid_true = Grid([[7]])
    grid_false = Grid([[8]])
    condition = registry.get("shape_list_nonempty")
    shapes = registry.get("connected_components").implementation(Grid([[1]]))
    cond_value = condition.implementation(shapes)
    chosen = registry.get("if_then_else").implementation(cond_value, grid_true, grid_false)
    assert chosen.cells == grid_true.cells


def test_value_comparators_and_booleans(registry):
    gt = registry.get("value_greater_than")
    eq = registry.get("value_equals")
    assert gt.implementation(3, 1) is True
    assert eq.implementation(2, 2) is True


def test_flood_fill_from_shape_centroid(registry):
    grid = Grid([[0, 3, 3], [0, 0, 0]])
    shapes = registry.get("connected_components").implementation(grid)
    shape = registry.get("shape_list_get").implementation(shapes, 0)
    centroid = registry.get("shape_centroid").implementation(shape)
    y = registry.get("position_y").implementation(centroid)
    x = registry.get("position_x").implementation(centroid)
    flood_fill = registry.get("flood_fill")
    filled = flood_fill.implementation(grid, y, x, 9)
    assert all(value in {0, 9} for row in filled.cells for value in row)


def test_grid_list_get_errors_on_empty(registry):
    with pytest.raises(ValueError):
        registry.get("grid_list_get").implementation((), 0)
