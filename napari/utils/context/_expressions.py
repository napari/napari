"""This module provides the  :class:`Expr` and its subclasses.

`Expr` is a subclass of `ast.AST` that provides rich dunder methods that
facilitate joining and comparing typed expressions.  It only implements a
subset of ast Expressions (for safety of evaluation), but provides more
than ast.literal_eval.

Expressions that are supported:
- Names: 'myvar' (these must be evaluated along with some context)
- Constants: '1'
- Comparisons: 'myvar > 1'
- Boolean Operators: 'myvar and yourvar'
- Binary Operators: 'myvar + 42'
- Unary Operators: 'not myvar'

Things that are *NOT* supported:
- attribute access: 'my.attr'
- calls: 'f(x)'
- containers (lists, tuples, sets, dicts)
- indexing or slicing
- joined strings (f-strings)
- named expressions (walrus operator)
- comprehensions (list, set, dict, generator)
- statements & assignments (e.g. 'a = b')
"""
from __future__ import annotations

import ast
import sys
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    SupportsIndex,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from ..translations import trans

ConstType = Union[None, str, bytes, bool, int, float]
PassedType = TypeVar(
    "PassedType",
    bound=Union[
        ast.cmpop, ast.operator, ast.boolop, ast.unaryop, ast.expr_context
    ],
)
T = TypeVar('T')
T2 = TypeVar('T2', bound=Union[ConstType, 'Expr'])
V = TypeVar('V', bound=ConstType)


def parse_expression(expr: str) -> Expr:
    """Parse string expression into an :class:`Expr` instance.

    Parameters
    ----------
    expr : str
        Expression to parse.

    Returns
    -------
    Expr
        Instance of `Expr`.

    Raises
    ------
    SyntaxError
        If the provided string is not an expression (e.g. it's a statement), or
        if it uses any forbidden syntax components (e.g. Call, Attribute,
        Containers, Indexing, Slicing, f-strings, named expression,
        comprehensions.)
    """
    try:
        # mode='eval' means the expr must consist of a single expression
        tree = ast.parse(expr, mode='eval')
        if not isinstance(tree, ast.Expression):
            raise SyntaxError  # pragma: no cover
        return ExprTranformer().visit(tree.body)
    except SyntaxError as e:
        raise SyntaxError(
            trans._(
                "{expr} is not a valid expression: ({error}).",
                deferred=True,
                expr=f"{expr!r}",
                error=e,
            )
        ) from None


def safe_eval(expr: str, context: Mapping = {}) -> Any:
    """Safely evaluate `expr` string given `context` dict.

    This lets you evaluate a string expression with broader expression
    support than `ast.literal_eval`, but much less support than `eval()`.
    """
    return parse_expression(expr).eval(context)


class Expr(ast.AST, Generic[T]):
    """Base Expression class providing dunder and convenience methods.

    This class is not meant to be instantiated.
    Instead, use :func:`parse_expression`, or the `Expr.parse` classmethod
    to create an expression instance.

    Once created, an expression can be joined with other expressions, or
    constants.

    Methods
    -------
    Outside of the dunder methods providing operator support, key methods are:

    eval(context: dict) -> Any
        Evaluate this expression in the `context` namespace.
    parse(expr: str) -> Expr
        (classmethod) - Parse a string to create a new expression
    _serialize() -> str
        serialize the expression to a string

    Examples
    --------
    >>> expr = parse_expression('myvar > 5')

    # combine expressions with operators
    >>> new_expr = expr & parse_expression('v2')

    # nice repr
    >>> new_expr
    BoolOp(
        op=And(),
        values=[
            Compare(
            left=Name(id='myvar', ctx=Load()),
            ops=[
                Gt()],
            comparators=[
                Constant(value=5)]),
            Name(id='v2', ctx=Load())])

    # evaluate in some context
    >>> new_expr.eval(dict(v2='hello!', myvar=8))
    'hello!'

    # serialize
    >>> print(new_expr)
    myvar > 5 and v2

    One reason you might want to use this is to capture named expressions
    that can be evaluated repeatedly as some underlying context changes.

    .. code-block:: python

        light_is_green = Name[bool]('light_is_green')
        count = Name[int]('count')
        is_ready = light_is_green & count > 5

        assert is_ready.eval({'count': 4, 'light_is_green': True}) == False
        assert is_ready.eval({'count': 7, 'light_is_green': False}) == False
        assert is_ready.eval({'count': 7, 'light_is_green': True}) == True

    this will also preserve type information:
    `reveal_type(is_ready())`  # revealed type is `bool`
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self).__name__ == 'Expr':
            raise RuntimeError(
                trans._(
                    "Don't instantiate Expr. Use `Expr.parse`", deferred=True
                )
            )
        super().__init__(*args, **kwargs)
        ast.fix_missing_locations(self)

    def eval(self, context: Mapping = {}) -> T:
        """Evaluate this expression with names in `context`"""
        code = compile(ast.Expression(body=self), '<Expr>', 'eval')
        try:
            return eval(code, {}, context)
        except NameError:
            miss = {k for k in _iter_names(self) if k not in context}
            raise NameError(
                trans._(
                    'Names required to eval this expression are missing: {miss}',
                    deferred=True,
                    miss=miss,
                )
            )

    @classmethod
    def parse(cls, expr: str) -> Expr:
        """Parse string into Expr (classmethod).

        see docstring of :func:`parse_expression` for details.
        """
        return parse_expression(str(expr))

    def __str__(self) -> str:
        """Serialize this expression to string form."""
        return self._serialize()

    def _serialize(self) -> str:
        """Serialize this expression to string form."""
        return str(ExprSerializer(self))

    def __repr__(self) -> str:
        if sys.version_info >= (3, 9):
            return ast.dump(self, indent=2)
        return ast.dump(self)

    @staticmethod
    def _cast(obj: Any) -> Expr:
        """Cast object into an Expression."""
        return obj if isinstance(obj, Expr) else Constant(obj)

    # boolean operators
    # '&' and '|' are normaly binary operators... but we use them here to
    # combine expression objects meaning "and" and "or".
    # if you want the binary operators, use Expr.bitand, and Expr.bitor

    def __and__(
        self, other: Union[Expr[T2], Expr[T], ConstType, Compare]
    ) -> BoolOp[Union[T, T2]]:
        return BoolOp(ast.And(), [self, other])

    def __or__(
        self, other: Union[Expr[T2], Expr[T], ConstType, Compare]
    ) -> BoolOp[Union[T, T2]]:
        return BoolOp(ast.Or(), [self, other])

    # comparisons

    def __lt__(self, other: Any) -> Compare:
        return Compare(self, [ast.Lt()], [other])

    def __le__(self, other: Any) -> Compare:
        return Compare(self, [ast.LtE()], [other])

    def __eq__(self, other: Any) -> Compare:  # type: ignore
        return Compare(self, [ast.Eq()], [other])

    def __ne__(self, other: Any) -> Compare:  # type: ignore
        return Compare(self, [ast.NotEq()], [other])

    def __gt__(self, other: Any) -> Compare:
        return Compare(self, [ast.Gt()], [other])

    def __ge__(self, other: Any) -> Compare:
        return Compare(self, [ast.GtE()], [other])

    # using __contains__ always returns a bool... so we provide our own
    # Expr.in_ and Expr.not_in methods

    def in_(self, other: Any) -> Compare:
        # not a dunder, use with Expr.in_(a, other)
        return Compare(self, [ast.In()], [other])

    def not_in(self, other: Any) -> Compare:
        # not a dunder, use with Expr.not_in(a, other)
        return Compare(self, [ast.NotIn()], [other])

    # binary operators
    # (note that __and__ and __or__ are reserved for boolean operators.)

    def __add__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.Add(), other)

    def __sub__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.Sub(), other)

    def __mul__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.Mult(), other)

    def __truediv__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.Div(), other)

    def __floordiv__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.FloorDiv(), other)

    def __mod__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.Mod(), other)

    def __matmul__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.MatMult(), other)

    def __pow__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.Pow(), other)

    def __xor__(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.BitXor(), other)

    def bitand(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.BitAnd(), other)

    def bitor(self, other: Union[T, Expr[T]]) -> BinOp[T]:
        return BinOp(self, ast.BitOr(), other)

    # unary operators

    def __neg__(self) -> UnaryOp[T]:
        return UnaryOp(ast.USub(), self)

    def __pos__(self) -> UnaryOp[T]:
        # usually a no-op
        return UnaryOp(ast.UAdd(), self)

    def __invert__(self) -> UnaryOp[T]:
        # note: we're using the invert operator `~` to mean "not ___"
        return UnaryOp(ast.Not(), self)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Tuple[Any, ...]:
        rv = list(super().__reduce_ex__(protocol))
        rv[1] = tuple(getattr(self, f) for f in self._fields)
        return tuple(rv)


class Name(Expr[T], ast.Name):
    """A variable name.

    `id` holds the name as a string.
    """

    def __init__(
        self, id: str, ctx: ast.expr_context = ast.Load(), **kwargs: Any
    ) -> None:
        kwargs['ctx'] = ast.Load()
        super().__init__(id, **kwargs)

    def eval(self, context: Mapping = {}) -> T:
        return super().eval(context=context)


class Constant(Expr[V], ast.Constant):
    """A constant value.

    The `value` attribute contains the Python object it represents.
    types supported: NoneType, str, bytes, bool, int, float
    """

    value: V

    def __init__(
        self, value: V, kind: Optional[str] = None, **kwargs: Any
    ) -> None:
        _valid_type = (type(None), str, bytes, bool, int, float)
        if not isinstance(value, _valid_type):
            raise TypeError(
                trans._(
                    "Constants must be type: {_valid_type!r}",
                    deferred=True,
                    _valid_type=_valid_type,
                )
            )
        super().__init__(value, kind, **kwargs)


class Compare(Expr[bool], ast.Compare):
    """A comparison of two or more values.

    `left` is the first value in the comparison, `ops` the list of operators,
    and `comparators` the list of values after the first element in the
    comparison.
    """

    def __init__(
        self,
        left: Expr,
        ops: Sequence[ast.cmpop],
        comparators: Sequence[Expr],
        **kwargs,
    ) -> None:
        super().__init__(
            Expr._cast(left),
            ops,
            [Expr._cast(c) for c in comparators],
            **kwargs,
        )


class BinOp(Expr[T], ast.BinOp):
    """A binary operation (like addition or division).

    `op` is the operator, and `left` and `right` are any expression nodes.
    """

    def __init__(
        self,
        left: Union[T, Expr[T]],
        op: ast.operator,
        right: Union[T, Expr[T]],
        **k,
    ) -> None:
        super().__init__(Expr._cast(left), op, Expr._cast(right), **k)


class BoolOp(Expr[T], ast.BoolOp):
    """A boolean operation, ‘or’ or ‘and’.

    `op` is Or or And. `values` are the values involved. Consecutive operations
    with the same operator, such as a or b or c, are collapsed into one node
    with several values.

    This doesn’t include `not`, which is a :class:`UnaryOp`.
    """

    def __init__(
        self,
        op: ast.boolop,
        values: Sequence[Union[ConstType, Expr]],
        **kwargs,
    ):
        super().__init__(op, [Expr._cast(v) for v in values], **kwargs)


class UnaryOp(Expr[T], ast.UnaryOp):
    """A unary operation.

    `op` is the operator, and `operand` any expression node.
    """

    def __init__(self, op: ast.unaryop, operand: Expr, **kwargs) -> None:
        super().__init__(op, Expr._cast(operand), **kwargs)


class IfExp(Expr, ast.IfExp):
    """An expression such as `'a if b else c'`.

    `body` if `test` else `orelse`
    """

    def __init__(self, test: Expr, body: Expr, orelse: Expr, **kwargs) -> None:
        super().__init__(
            Expr._cast(test), Expr._cast(body), Expr._cast(orelse), **kwargs
        )


class ExprTranformer(ast.NodeTransformer):
    """This transformer converts an ast.expr into an :class:`Expr`.

    Examples
    --------
    >>> tree = ast.parse('my_var > 11', mode='eval')
    >>> tree = ExprTranformer().visit(tree)  # transformed
    """

    _SUPPORTED_NODES = frozenset(
        k
        for k, v in globals().items()
        if isinstance(v, type) and issubclass(v, Expr)
    )

    # fmt: off
    @overload
    def visit(self, node: ast.expr) -> Expr: ...  # noqa
    @overload
    def visit(self, node: PassedType) -> PassedType: ...  # noqa
    # fmt: on

    def visit(self, node):
        """Visit a node in the tree, transforming into Expr."""

        if isinstance(
            node,
            (
                ast.cmpop,
                ast.operator,
                ast.boolop,
                ast.unaryop,
                ast.expr_context,
            ),
        ):
            # all operation types just get passed through
            return node

        # filter here for supported expression node types
        type_ = type(node).__name__

        if type_ not in ExprTranformer._SUPPORTED_NODES:
            raise SyntaxError(
                trans._(
                    "Type {type_!r} not supported", deferred=True, type_=type_
                )
            )

        # providing fake lineno and col_offset here rather than using
        # ast.fill_missing_locations for typing purposes
        kwargs: Dict[str, Any] = dict(lineno=1, col_offset=0)

        for name, field in ast.iter_fields(node):
            if isinstance(field, ast.expr):
                kwargs[name] = self.visit(field)
            elif isinstance(field, list):
                kwargs[name] = [self.visit(item) for item in field]
            else:
                kwargs[name] = field
        return globals()[type_](**kwargs)


class ExprSerializer(ast.NodeVisitor):
    """Serializes an :class:`Expr` into a string.

    Examples
    --------
    >>> expr = Expr.parse('a + b == c')
    >>> print(expr)
    'a + b == c'

    or ... using this visitor directly:

    >>> ser = ExprSerializer()
    >>> ser.visit(expr)
    >>> out = "".join(ser.result)
    """

    def __init__(self, node: Optional[Expr] = None) -> None:
        self._result: List[str] = []

        def write(*params):
            for item in params:
                if isinstance(item, ast.AST):
                    self.visit(item)
                elif item:
                    self._result.append(item)

        self.write = write

        if node is not None:
            self.visit(node)

    def __str__(self):
        return "".join(self._result)

    def visit_Name(self, node: ast.Name) -> None:
        self.write(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        self.write(repr(node.value))

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        op = f' {_OPS[type(node.op)]} '
        for idx, value in enumerate(node.values):
            self.write(idx and op or '', value)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        for op, right in zip(node.ops, node.comparators):
            self.write(f' {_OPS[type(op)]} ', right)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.write(node.left, f' {_OPS[type(node.op)]} ', node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        sym = _OPS[type(node.op)]
        self.write(sym, ' ' if sym.isalpha() else '', node.operand)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.write(node.body, ' if ', node.test, ' else ', node.orelse)


OpType = Union[
    Type[ast.operator], Type[ast.cmpop], Type[ast.boolop], Type[ast.unaryop]
]
_OPS: Dict[OpType, str] = {
    # ast.boolop
    ast.Or: 'or',
    ast.And: 'and',
    # ast.cmpop
    ast.Eq: '==',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.In: 'in',
    ast.Is: 'is',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.NotIn: 'not in',
    ast.IsNot: 'is not',
    # ast.operator
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.BitAnd: '&',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Mod: '%',
    ast.FloorDiv: '//',
    ast.MatMult: '@',
    ast.Pow: '**',
    # ast.unaryop
    ast.Not: 'not',
    ast.Invert: '~',
    ast.UAdd: '+',
    ast.USub: '-',
}


def _iter_names(expr: Expr) -> Iterator[str]:
    """Iterate all (nested) names used in the expression.

    Could be used to provide nicer error messages when eval() fails.
    """
    if isinstance(expr, Name):
        yield expr.id
    elif isinstance(expr, Expr):
        for field, val in ast.iter_fields(expr):
            val = val if isinstance(val, list) else [val]
            for v in val:
                yield from _iter_names(v)
