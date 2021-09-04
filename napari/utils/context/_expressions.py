from __future__ import annotations

import ast
import itertools
import operator
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterable, Sequence, Tuple, TypeVar, Union

__all__ = [
    'parse',
    'Expr',
    'Name',
    'Constant',
    'BoolOp',
    'Compare',
    'UnaryOp',
    'BinOp',
]

T = TypeVar('T')
ConstType = Union[None, str, bytes, bool, int, float]


def _iter_pairs(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _expr_cast(obj: Any) -> Expr:
    return obj if isinstance(obj, Expr) else Constant(obj)


class Expr(ABC):
    @abstractmethod
    def eval(self, context: dict = {}):
        ...

    @abstractmethod
    def _sig_repr(self) -> str:
        ...

    @abstractmethod
    def serialize(self) -> str:
        ...

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._sig_repr()!r})'

    # boolean operators (normaly binary operators)

    def __and__(self, other: Any) -> BoolOp:
        other = other if isinstance(other, Expr) else Constant(other)
        return BoolOp(BoolOp.BoolOpType.And, [self, other])

    def __or__(self, other: Any) -> BoolOp:
        other = other if isinstance(other, Expr) else Constant(other)
        return BoolOp(BoolOp.BoolOpType.Or, [self, other])

    # comparisons

    def __lt__(self, other: Any) -> Compare:
        return Compare(self, [Compare.CmpType.Lt], [other])

    def __le__(self, other: Any) -> Compare:
        return Compare(self, [Compare.CmpType.LtE], [other])

    def __eq__(self, other: Any) -> Compare:  # type: ignore
        return Compare(self, [Compare.CmpType.Eq], [other])

    def __ne__(self, other: Any) -> Compare:  # type: ignore
        return Compare(self, [Compare.CmpType.NotEq], [other])

    def __gt__(self, other: Any) -> Compare:
        return Compare(self, [Compare.CmpType.Gt], [other])

    def __ge__(self, other: Any) -> Compare:
        return Compare(self, [Compare.CmpType.GtE], [other])

    # binary operators

    def __add__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.Add, other)

    def __sub__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.Sub, other)

    def __mul__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.Mult, other)

    def __truediv__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.Div, other)

    def __floordiv__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.FloorDiv, other)

    def __mod__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.Mod, other)

    def __pow__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.Pow, other)

    def __xor__(self, other: Any) -> BinOp:
        return BinOp(self, BinOp.BinOpType.BitXor, other)

    def __str__(self) -> str:
        return self.serialize()

    @classmethod
    def parse(cls, string: str) -> Expr:
        try:
            exp_node = ast.parse(string, mode='eval')
            if not isinstance(exp_node, ast.Expression):
                raise SyntaxError
        except SyntaxError:
            raise ValueError(f"string is an invalid expression: {string!r}")
        return cls._from_ast_node(exp_node.body)

    @classmethod
    def _from_ast_node(cls, node: ast.expr) -> Expr:
        if isinstance(node, ast.Expression):
            node = node.body
        elif isinstance(node, ast.Name):
            return Name(node.id)
        elif isinstance(node, ast.Constant):
            return Constant(node.value)
        elif isinstance(node, ast.BoolOp):
            bop = getattr(BoolOp.BoolOpType, type(node.op).__name__)
            return BoolOp(bop, [Expr._from_ast_node(v) for v in node.values])
        elif isinstance(node, ast.UnaryOp):
            uop = getattr(UnaryOp.UnaryOpType, type(node.op).__name__)
            return UnaryOp(uop, Expr._from_ast_node(node.operand))
        elif isinstance(node, ast.Compare):
            left = Expr._from_ast_node(node.left)
            op = [getattr(Compare.CmpType, type(o).__name__) for o in node.ops]
            comparators = [Expr._from_ast_node(n) for n in node.comparators]
            return Compare(left, op, comparators)
        elif isinstance(node, ast.BinOp):
            left = Expr._from_ast_node(node.left)
            right = Expr._from_ast_node(node.right)
            binop = getattr(BinOp.BinOpType, type(node.op).__name__)
            return BinOp(left, binop, right)
        raise NotImplementedError(
            f'Cannot convert ast node of type: {type(node)}'
        )


parse = Expr.parse


class Name(Expr):
    def __init__(self, id: str):
        self.id = id

    def eval(self, context: dict = {}) -> Any:
        return context[self.id]

    def _sig_repr(self):
        return self.id

    def serialize(self):
        return self.id


class Constant(Expr):
    def __init__(self, value: ConstType):
        if not isinstance(value, (str, bytes, bool, int, float, type(None))):
            raise TypeError(
                "Constant value must be of type: "
                f"{{None, str, bytes, bool, int, float}}. Got: {type(value)}"
            )
        self.value = value

    def eval(self, context: dict = {}) -> Any:
        return self.value

    def _sig_repr(self):
        return self.value

    def serialize(self):
        return repr(self.value)


class BoolOp(Expr):
    class BoolOpType(Enum):
        And = all  # 'and'
        Or = any  # 'or'

        def serialize(self) -> str:
            return type2char[self]

    def __init__(self, op: BoolOpType, values: Sequence[Expr]):
        self.op = op
        self.values = values

    def eval(self, context: dict = {}) -> bool:
        return self.op.value(v.eval(context) for v in self.values)

    def _sig_repr(self):
        return self.op, self.values

    def serialize(self):
        j = f' {self.op.serialize()} '
        return j.join(v.serialize() for v in self.values)


def _in(a: Any, b: Any) -> bool:
    # swapping order of a, b
    return operator.contains(b, a)


def _not_in(a: Any, b: Any) -> bool:
    # swapping order of a, b
    return not operator.contains(b, a)


class Compare(Expr):
    class CmpType(Enum):
        Eq = operator.eq  # '=='
        NotEq = operator.ne  # '!='
        Lt = operator.lt  # '<'
        LtE = operator.le  # '<='
        Gt = operator.gt  # '>'
        GtE = operator.ge  # '>='
        Is = _in  # 'is'
        IsNot = _not_in  # 'is not'
        In = operator.is_  # 'in'
        NotIn = operator.is_not  # 'not in'

        def serialize(self) -> str:
            return type2char[self]

    def __init__(
        self,
        left: Expr,
        ops: Sequence[CmpType],
        comparators: Sequence[Union[Expr, ConstType]],
    ) -> None:
        self.left = left
        self.ops = ops
        self.comparators = [_expr_cast(c) for c in comparators]

    def eval(self, context: dict = {}) -> Any:
        pairs = _iter_pairs(itertools.chain([self.left], self.comparators))
        return all(
            op.value(l.eval(context), r.eval(context))
            for (l, r), op in zip(pairs, self.ops)
        )

    def _sig_repr(self):
        return self.left, self.ops, self.comparators

    def serialize(self) -> str:
        out = self.left.serialize()
        for o, c in zip(self.ops, self.comparators):
            out += f' {o.serialize()} {c.serialize()}'
        return out


class UnaryOp(Expr):
    class UnaryOpType(Enum):
        Invert = operator.invert  # '~'
        Not = operator.not_  # 'not'
        UAdd = operator.pos  # '+'
        USub = operator.neg  # '-'

        def serialize(self):
            return type2char[self]

    def __init__(self, op: UnaryOpType, operand: Expr) -> None:
        self.op = op
        self.operand = operand

    def eval(self, context: dict = {}) -> Any:
        return self.op.value(self.operand.eval(context))

    def _sig_repr(self):
        return self.op, self.operand

    def serialize(self) -> str:
        return self.op.serialize() + self.operand.serialize()


class BinOp(Expr):
    class BinOpType(Enum):
        Add = operator.add  # '+'
        BitAnd = operator.and_  # '&'  reserved for boolean AND op
        BitOr = operator.or_  # '|'    reserved for boolean OR op
        BitXor = operator.xor  # '^'
        Div = operator.truediv  # '/'
        FloorDiv = operator.floordiv  # '//'
        Mod = operator.mod  # '%'
        Mult = operator.mul  # '*'
        Pow = operator.pow  # '**'
        Sub = operator.sub  # '-'

        def serialize(self):
            return type2char[self]

    def __init__(
        self, left: Expr, op: BinOpType, right: Union[Expr, ConstType]
    ) -> None:
        self.left = left
        self.op = op
        self.right = _expr_cast(right)

    def eval(self, context: dict = {}) -> Any:
        return self.op.value(self.left.eval(context), self.right.eval(context))

    def _sig_repr(self):
        return self.left, self.op, self.right

    def serialize(self) -> str:
        return ' '.join(
            (
                self.left.serialize(),
                self.op.serialize(),
                self.right.serialize(),
            )
        )


type2char = {
    BoolOp.BoolOpType.And: 'and',
    BoolOp.BoolOpType.And: 'or',
    Compare.CmpType.Eq: '==',
    Compare.CmpType.NotEq: '!=',
    Compare.CmpType.Lt: '<',
    Compare.CmpType.LtE: '<=',
    Compare.CmpType.Gt: '>',
    Compare.CmpType.GtE: '>=',
    Compare.CmpType.Is: 'is',
    Compare.CmpType.IsNot: 'is not',
    Compare.CmpType.In: 'in',
    Compare.CmpType.NotIn: 'not in',
    UnaryOp.UnaryOpType.Invert: '~',
    UnaryOp.UnaryOpType.Not: 'not',
    UnaryOp.UnaryOpType.UAdd: '+',
    UnaryOp.UnaryOpType.USub: '-',
    BinOp.BinOpType.Add: '+',
    BinOp.BinOpType.BitAnd: '&',
    BinOp.BinOpType.BitOr: '|',
    BinOp.BinOpType.BitXor: '^',
    BinOp.BinOpType.Div: '/',
    BinOp.BinOpType.FloorDiv: '//',
    BinOp.BinOpType.Mod: '%',
    BinOp.BinOpType.Mult: '*',
    BinOp.BinOpType.Pow: '**',
    BinOp.BinOpType.Sub: '-',
}
