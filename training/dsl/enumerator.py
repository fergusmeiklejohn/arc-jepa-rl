"""Program enumeration scaffolding for the ARC DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from .primitives import Primitive, PrimitiveRegistry


@dataclass(frozen=True)
class ProgramNode:
    primitive: Primitive
    args: Sequence[int]  # references to child indices in program list


@dataclass(frozen=True)
class Program:
    nodes: Sequence[ProgramNode]

    def __len__(self):  # pragma: no cover - convenience
        return len(self.nodes)


class ProgramEnumerator:
    def __init__(self, registry: PrimitiveRegistry, max_depth: int = 4, max_nodes: int = 8) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

        self.registry = registry
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def enumerate(self, root_primitives: Iterable[str]) -> Iterable[Program]:
        primitives = [self.registry.get(name) for name in root_primitives]

        for primitive in primitives:
            yield from self._enumerate_from_primitive(primitive, depth=1, nodes=[])

    def _enumerate_from_primitive(
        self,
        primitive: Primitive,
        *,
        depth: int,
        nodes: List[ProgramNode],
    ) -> Iterable[Program]:
        if depth > self.max_depth or len(nodes) >= self.max_nodes:
            return

        node_index = len(nodes)
        arg_indices = tuple(range(node_index - len(primitive.input_types), node_index))
        node = ProgramNode(primitive=primitive, args=arg_indices)
        nodes.append(node)

        if depth == self.max_depth or not primitive.input_types:
            yield Program(tuple(nodes))
        else:
            for child_primitive in self.registry.list():
                yield from self._enumerate_from_primitive(
                    child_primitive,
                    depth=depth + 1,
                    nodes=nodes,
                )

        nodes.pop()
