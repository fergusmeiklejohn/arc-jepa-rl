import pytest

from training.dsl.types import (
    Bool,
    Color,
    Grid,
    GridList,
    GridValue,
    Position,
    ProductType,
    SumType,
    infer_type,
    is_value_of_type,
)


def test_product_type_matches_tuple():
    pos = Position
    assert isinstance(pos, ProductType)
    assert is_value_of_type((1, 2), pos)
    assert not is_value_of_type((1, True), pos)
    assert not is_value_of_type((1,), pos)


def test_sum_type_matches_any_option():
    opt = SumType("ColorOrBool", options=(Color, Bool))
    assert is_value_of_type(3, opt)
    assert is_value_of_type(True, opt)
    assert not is_value_of_type("foo", opt)


def test_infer_type_best_effort():
    assert infer_type(True) == Bool
    assert infer_type(5) == GridValue
    assert infer_type((1, 2)) == Position
    assert infer_type([]) is None


def test_interpreter_checks_types():
    from training.dsl.enumerator import Expression, InputVar, Program, ProgramInterpreter
    from training.dsl.primitives import Primitive

    # Primitive expecting Position returns first coordinate
    def first_coord(pos):
        return pos[0]

    prim = Primitive("first_y", input_types=(Position,), output_type=GridValue, implementation=first_coord)
    expr = Expression(
        type=GridValue,
        primitive=prim,
        args=(Expression(type=Position, var=InputVar("p", Position)),),
    )
    program = Program(expr)
    interpreter = ProgramInterpreter()

    with pytest.raises(KeyError):
        interpreter.evaluate(program, inputs={})
    assert interpreter.evaluate(program, inputs={"p": (3, 4)}) == 3
    with pytest.raises(TypeError):
        interpreter.evaluate(program, inputs={"p": "wrong"})
