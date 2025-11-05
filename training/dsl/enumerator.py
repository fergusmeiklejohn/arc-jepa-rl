"""Program enumeration scaffolding for the ARC DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from .primitives import Primitive, PrimitiveRegistry
from .types import DSLType


@dataclass(frozen=True)
class InputVar:
    """Represents an input placeholder available during enumeration."""

    name: str
    type: DSLType


@dataclass(frozen=True)
class Expression:
    """Single typed expression node."""

    type: DSLType
    primitive: Primitive | None = None
    args: Tuple["Expression", ...] = ()
    var: InputVar | None = None
    _size: int = 0
    _signature: Tuple | None = None

    def __post_init__(self) -> None:
        if (self.primitive is None) == (self.var is None):
            raise ValueError("expression must be either a primitive call or an input reference")

        if self.var is not None:
            size = 0
            signature: Tuple = ("var", self.var.name)
        else:
            child_sizes = tuple(arg.size for arg in self.args)
            size = 1 + sum(child_sizes)
            signature = ("call", self.primitive.name, tuple(arg.signature for arg in self.args))

        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "_signature", signature)

    @property
    def size(self) -> int:
        return self._size

    @property
    def signature(self) -> Tuple:
        sig = self._signature
        assert sig is not None  # pragma: no cover - guaranteed post-init
        return sig

    def is_variable(self) -> bool:
        return self.var is not None

    def __hash__(self) -> int:  # pragma: no cover - trivial wrapper
        return hash(self.signature)


@dataclass(frozen=True)
class Program:
    """A typed DSL program rooted at an expression."""

    root: Expression

    def __len__(self) -> int:  # pragma: no cover - convenience
        return self.root.size

    def traverse(self) -> Iterator[Expression]:
        stack = [self.root]
        while stack:
            expr = stack.pop()
            yield expr
            stack.extend(reversed(expr.args))


class ProgramEnumerator:
    """Bottom-up typed enumerator producing programs within a node budget."""

    def __init__(
        self,
        registry: PrimitiveRegistry,
        inputs: Sequence[InputVar],
        target_type: DSLType,
        *,
        max_nodes: int = 6,
    ) -> None:
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

        self.registry = registry
        self.inputs = tuple(inputs)
        self.target_type = target_type
        self.max_nodes = max_nodes

    def enumerate(self) -> Iterator[Program]:
        expressions_by_type: Dict[DSLType, Dict[int, List[Expression]]] = {}
        signatures: Dict[DSLType, set] = {}

        def register(expr: Expression) -> None:
            type_buckets = expressions_by_type.setdefault(expr.type, {})
            seen = signatures.setdefault(expr.type, set())
            if expr.signature in seen:
                return
            seen.add(expr.signature)
            type_buckets.setdefault(expr.size, []).append(expr)

        for var in self.inputs:
            register(Expression(type=var.type, var=var))

        primitives = self.registry.list()

        for size in range(1, self.max_nodes + 1):
            for primitive in primitives:
                arity = len(primitive.input_types)

                if arity == 0:
                    if size != 1:
                        continue
                    expr = Expression(type=primitive.output_type, primitive=primitive, args=())
                    register(expr)
                    if expr.type == self.target_type:
                        yield Program(expr)
                    continue

                for combo in self._iter_argument_combinations(
                    primitive.input_types,
                    expressions_by_type,
                    size_limit=size - 1,
                ):
                    expr = Expression(type=primitive.output_type, primitive=primitive, args=combo)
                    register(expr)
                    if expr.type == self.target_type:
                        yield Program(expr)

    def _iter_argument_combinations(
        self,
        input_types: Sequence[DSLType],
        pool: Mapping[DSLType, Mapping[int, List[Expression]]],
        *,
        size_limit: int,
    ) -> Iterator[Tuple[Expression, ...]]:
        if not input_types:
            return

        size_buckets: List[List[Tuple[int, List[Expression]]]] = []
        for dtype in input_types:
            buckets = pool.get(dtype, {})
            filtered = [(expr_size, exprs) for expr_size, exprs in buckets.items() if expr_size <= size_limit]
            size_buckets.append(filtered)

        def backtrack(index: int, remaining: int, current: List[Expression]) -> Iterator[Tuple[Expression, ...]]:
            if index == len(input_types):
                if remaining >= 0:
                    yield tuple(current)
                return

            for expr_size, exprs in size_buckets[index]:
                if expr_size > remaining:
                    continue
                for expr in exprs:
                    current.append(expr)
                    yield from backtrack(index + 1, remaining - expr_size, current)
                    current.pop()

        yield from backtrack(0, size_limit, [])


class ProgramInterpreter:
    """Evaluates programs by executing primitive implementations."""

    def evaluate(self, program: Program, inputs: Mapping[str, object]) -> object:
        return self._evaluate_expr(program.root, inputs)

    def _evaluate_expr(self, expr: Expression, inputs: Mapping[str, object]) -> object:
        if expr.var is not None:
            if expr.var.name not in inputs:
                raise KeyError(f"missing input value for '{expr.var.name}'")
            return inputs[expr.var.name]

        args = [self._evaluate_expr(child, inputs) for child in expr.args]
        return expr.primitive.implementation(*args)
