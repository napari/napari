import ast
import inspect
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionDependencies:
    attributes: frozenset[str] = frozenset()
    globals: frozenset[str] = frozenset()
    nonlocals: frozenset[str] = frozenset()
    builtins: frozenset[str] = frozenset()

    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, Mapping):
            value = FunctionDependencies(
                **{k: frozenset(v) for k, v in value.items()}
            )
        if not isinstance(value, FunctionDependencies):
            return False  # pragma: no cover
        return (
            self.attributes == value.attributes
            and self.globals == value.globals
            and self.nonlocals == value.nonlocals
            and self.builtins == value.builtins
        )


def attribute_path(node: ast.expr) -> str | None:
    parts = []
    current: ast.expr = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)
        return '.'.join(reversed(parts))

    return None


class DependencyVisitor(ast.NodeVisitor):
    def __init__(
        self,
        params: tuple[str, ...],
        globals: set[str],  # noqa: A002
        nonlocals: set[str],
        builtins: set[str],
    ):
        self.parameters = params
        self.globals = globals
        self.nonlocals = nonlocals
        self.builtins = builtins

        self.attributes: set[str] = set()
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            # For `self.value.copy()`, visit `self.value` but not `.copy`.
            self.visit(node.func.value)
        else:
            # Preserve bare calls such as `tuple(...)` and `round(...)`.
            self.visit(node.func)

        for argument in node.args:
            self.visit(argument)

        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = attribute_path(node)
        if not isinstance(node.ctx, ast.Load):
            return
        if path:
            name, tail = path.split('.', 1)
            if name in self.parameters:
                self.attributes.add(tail)
                return
            if (
                name in self.globals
                or name in self.nonlocals
                or name in self.builtins
            ):
                self.names.add(name)
                return

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for el in node.body:
            self.generic_visit(el)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def property_dependencies(prop: Callable | property) -> FunctionDependencies:
    func: Callable
    if isinstance(prop, property):
        if prop.fget is None:
            raise ValueError(
                'Property has no getter function'
            )  # pragma: no cover
        func = prop.fget
    else:
        func = prop
    if not inspect.isfunction(func):
        raise TypeError(f'Expected a callable or property, got {type(func)}')

    parameters = tuple(inspect.signature(func).parameters)
    if not len(parameters) == 1:
        raise ValueError(
            f'Expected a property getter with a single parameter, got {len(parameters)} parameters: {parameters}'
        )

    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)

    closure = inspect.getclosurevars(func)
    visitor = DependencyVisitor(
        parameters,
        set(closure.globals.keys()),
        set(closure.nonlocals.keys()),
        set(closure.builtins.keys()),
    )
    visitor.visit(tree)

    globals_used = visitor.names & closure.globals.keys()
    nonlocals_used = visitor.names & closure.nonlocals.keys()
    builtins_used = visitor.names & closure.builtins.keys()

    return FunctionDependencies(
        attributes=frozenset(visitor.attributes),
        globals=frozenset(globals_used),
        nonlocals=frozenset(nonlocals_used),
        builtins=frozenset(builtins_used),
    )
