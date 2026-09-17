"""Discover property dependencies without requiring Python source files.

Analysis follows attribute paths through bytecode, never evaluating the
getter or its objects.
Dynamic attribute names and attributes of call results require explicit model
dependencies.

This code is not intended to be a general-purpose bytecode analyzer;
it is created to analyze property getters for static attribute reads in the napari event system.
"""

import dis
import inspect
import sys
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import CodeType


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


@dataclass(frozen=True)
class _Value:
    # Paths still identify an object; reads have been consumed by an operation
    # and must not be extended by a subsequent attribute lookup.
    paths: frozenset[tuple[str, ...]] = frozenset()
    reads: frozenset[tuple[str, ...]] = frozenset()
    inplace: frozenset[tuple[str, ...]] = frozenset()
    unresolved: frozenset[str] = frozenset()

    def merge(self, other: '_Value') -> '_Value':
        return _Value(
            self.paths | other.paths,
            self.reads | other.reads,
            self.inplace | other.inplace,
            self.unresolved | other.unresolved,
        )

    @property
    def dependencies(self) -> frozenset[tuple[str, ...]]:
        return self.paths | self.reads


_UNKNOWN = _Value()
_SELF = _Value(paths=frozenset({()}))


class _BytecodeDependencies:
    def __init__(self, func: Callable, *, strict: bool = False):
        self.strict = strict
        self.globals = func.__globals__
        self.builtins = func.__builtins__
        self.attributes: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self.builtin_names: set[str] = set()

    def record(self, *values: _Value) -> None:
        for value in values:
            self.attributes.update(
                '.'.join(path) for path in value.dependencies if path
            )

    def scan(
        self,
        code: CodeType,
        bindings: dict[str, _Value],
        external_nonlocals: set[str],
    ) -> None:
        scanner = _CodeScanner(self, code, bindings, external_nonlocals)
        scanner.run()
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                self.scan(
                    constant,
                    {
                        name: scanner.captured.get(name, _UNKNOWN)
                        for name in constant.co_freevars
                    },
                    external_nonlocals & set(constant.co_freevars),
                )


@dataclass
class _Frame:
    """Symbolic stack and local bindings at one instruction."""

    stack: list[_Value]
    locals: dict[str, _Value]

    def copy(self) -> '_Frame':
        return _Frame(self.stack.copy(), self.locals.copy())

    def pop(self) -> _Value:
        return self.stack.pop() if self.stack else _UNKNOWN

    def take(self, count: int) -> list[_Value]:
        return [self.pop() for _ in range(count)]

    def load(self, name: str) -> None:
        self.stack.append(self.locals.get(name, _UNKNOWN))


def _stack_effect(
    instruction: dis.Instruction, *, jump: bool | None = None
) -> int:
    args = () if instruction.arg is None else (instruction.arg,)
    return dis.stack_effect(instruction.opcode, *args, jump=jump)


class _CodeScanner:
    """Analyze one code object, merging frames when control-flow paths meet."""

    def __init__(
        self,
        dependencies: _BytecodeDependencies,
        code: CodeType,
        bindings: dict[str, _Value],
        external_nonlocals: set[str],
    ) -> None:
        self.dependencies = dependencies
        self.code = code
        self.external_nonlocals = external_nonlocals
        self.instructions = list(dis.get_instructions(code))
        self.offsets = {
            instruction.offset: i
            for i, instruction in enumerate(self.instructions)
        }
        self.exception_entries = dis.Bytecode(code).exception_entries
        self.captured = dict(bindings)
        self.states: dict[int, _Frame] = {}
        self.pending: deque[int] = deque()
        # Bound growing aliases such as `node = node.child` in loops.
        captured_depth = max(
            (len(path) for value in bindings.values() for path in value.paths),
            default=0,
        )
        self.path_limit = (
            captured_depth
            + sum(i.opname == 'LOAD_ATTR' for i in self.instructions)
            + 1
        )
        self._enqueue(0, _Frame([], bindings))

    def run(self) -> None:
        while self.pending:
            index = self.pending.popleft()
            frame = self.states[index].copy()
            instruction = self.instructions[index]
            self._enqueue_exception_handlers(instruction, frame)
            for successor, next_frame in self._successors(
                index, instruction, frame
            ):
                self._enqueue(successor, next_frame)

    def _enqueue(self, index: int, frame: _Frame) -> None:
        if index >= len(self.instructions):
            return
        if previous := self.states.get(index):
            frame = self._merge(previous, frame)
            if frame == previous:
                return
        self.states[index] = frame.copy()
        self.pending.append(index)

    def _merge(self, previous: _Frame, incoming: _Frame) -> _Frame:
        if len(previous.stack) != len(incoming.stack):
            # Opaque operations can obscure provenance. Preserve known reads,
            # but never manufacture a path from mismatched stack positions.
            self.dependencies.record(*previous.stack, *incoming.stack)
            stack = [_UNKNOWN] * max(len(previous.stack), len(incoming.stack))
        else:
            stack = [
                a.merge(b)
                for a, b in zip(previous.stack, incoming.stack, strict=True)
            ]
        local = {
            name: previous.locals.get(name, _UNKNOWN).merge(
                incoming.locals.get(name, _UNKNOWN)
            )
            for name in previous.locals.keys() | incoming.locals.keys()
        }
        return _Frame(stack, local)

    def _enqueue_exception_handlers(
        self, instruction: dis.Instruction, frame: _Frame
    ) -> None:
        for entry in self.exception_entries:
            if entry.start <= instruction.offset < entry.end:
                stack = frame.stack[: entry.depth] + [_UNKNOWN] * (
                    1 + entry.lasti
                )
                self._enqueue(
                    self.offsets[entry.target], _Frame(stack, frame.locals)
                )

    def _store_local(self, frame: _Frame, name: str) -> None:
        value = frame.pop()
        frame.locals[name] = value
        self.captured[name] = self.captured.get(name, _UNKNOWN).merge(value)

    def _successors(
        self, index: int, instruction: dis.Instruction, frame: _Frame
    ) -> list[tuple[int, _Frame]]:
        """Apply one instruction and return frames for its possible successors."""
        op = instruction.opname
        successors = []
        if op == 'FOR_ITER':
            exhausted = frame.copy()
            # 3.12+ removes the exhausted iterator in END_FOR.
            exhausted.take(-_stack_effect(instruction, jump=True))
            successors.append((self.offsets[instruction.argval], exhausted))
            frame.stack.append(_UNKNOWN)
        elif op in {'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP'}:
            self.dependencies.record(
                frame.stack[-1] if frame.stack else _UNKNOWN
            )
            successors.append((self.offsets[instruction.argval], frame.copy()))
            frame.pop()
        elif op.startswith('POP_JUMP'):
            self.dependencies.record(frame.pop())
            successors.append((self.offsets[instruction.argval], frame.copy()))
        elif op.startswith('JUMP'):
            return [(self.offsets[instruction.argval], frame)]
        elif op in {
            'RETURN_VALUE',
            'RETURN_CONST',
            'RAISE_VARARGS',
            'RERAISE',
        }:
            self.dependencies.record(*frame.stack)
            return []
        elif op == 'YIELD_VALUE':
            self.dependencies.record(frame.pop())
            frame.stack.append(_UNKNOWN)
        elif op == 'RETURN_GENERATOR':
            # Resumption supplies a value consumed by POP_TOP.
            frame.stack.append(_UNKNOWN)
        elif op == 'POP_TOP':
            self.dependencies.record(frame.pop())
        elif not self._execute(instruction, frame):
            # Opaque operations preserve reads but discard path provenance.
            self.dependencies.record(*frame.stack)
            depth = len(frame.stack)
            frame.stack = [_UNKNOWN] * max(
                0, depth + _stack_effect(instruction)
            )
            if (
                instruction.opcode in dis.hasjabs
                or instruction.opcode in dis.hasjrel
            ):
                branch = _Frame(
                    [_UNKNOWN]
                    * max(0, depth + _stack_effect(instruction, jump=True)),
                    frame.locals.copy(),
                )
                successors.append((self.offsets[instruction.argval], branch))
        successors.append((index + 1, frame))
        return successors

    def _execute(self, instruction: dis.Instruction, frame: _Frame) -> bool:
        """Try instruction families in order; False requests opaque handling."""
        return (
            self._variables(instruction, frame)
            or self._attributes(instruction, frame)
            or self._operators(instruction, frame)
            or self._calls(instruction, frame)
            or self._containers(instruction, frame)
        )

    def _variables(self, instruction: dis.Instruction, frame: _Frame) -> bool:
        """Load and store bindings, tracking captured and external names."""
        op = instruction.opname
        arg = instruction.arg or 0
        name = instruction.argval
        if op in {
            'CACHE',
            'RESUME',
            'NOP',
            'EXTENDED_ARG',
            'MAKE_CELL',
            'COPY_FREE_VARS',
            'KW_NAMES',
            'NOT_TAKEN',
        }:
            pass
        elif op in {
            'LOAD_FAST_LOAD_FAST',
            'LOAD_FAST_BORROW_LOAD_FAST_BORROW',
        }:
            frame.load(self.code.co_varnames[arg >> 4])
            frame.load(self.code.co_varnames[arg & 15])
        elif op == 'STORE_FAST_LOAD_FAST':
            self._store_local(frame, self.code.co_varnames[arg >> 4])
            frame.load(self.code.co_varnames[arg & 15])
        elif op == 'STORE_FAST_STORE_FAST':
            self._store_local(frame, self.code.co_varnames[arg >> 4])
            self._store_local(frame, self.code.co_varnames[arg & 15])
        elif op in {
            'LOAD_FAST',
            'LOAD_FAST_CHECK',
            'LOAD_FAST_BORROW',
            'LOAD_DEREF',
            'LOAD_CLOSURE',
            'LOAD_CLASSDEREF',
            'LOAD_FAST_AND_CLEAR',
        }:
            if (
                op in {'LOAD_DEREF', 'LOAD_CLASSDEREF'}
                and name in self.external_nonlocals
            ):
                self.dependencies.nonlocal_names.add(name)
            frame.load(name)
            if op == 'LOAD_FAST_AND_CLEAR':
                frame.locals[name] = _UNKNOWN
        elif op in {'STORE_FAST', 'STORE_DEREF', 'STORE_NAME'}:
            self._store_local(frame, name)
        elif op in {'DELETE_FAST', 'DELETE_DEREF', 'DELETE_NAME'}:
            frame.locals.pop(name, None)
        elif op in {'LOAD_GLOBAL', 'LOAD_NAME'}:
            value = _UNKNOWN
            if name in self.dependencies.globals:
                self.dependencies.global_names.add(name)
            elif name in self.dependencies.builtins:
                self.dependencies.builtin_names.add(name)
            elif op == 'LOAD_NAME' and name in frame.locals:
                value = frame.locals[name]
            elif self.dependencies.strict:
                value = _Value(unresolved=frozenset({name}))
            values = [_UNKNOWN] * _stack_effect(instruction)
            # LOAD_GLOBAL may also push NULL for a subsequent call.
            values[0 if sys.version_info >= (3, 13) else -1] = value
            frame.stack.extend(values)
        elif op in {
            'LOAD_CONST',
            'LOAD_SMALL_INT',
            'PUSH_NULL',
            'LOAD_BUILD_CLASS',
            'LOAD_ASSERTION_ERROR',
        }:
            frame.stack.append(_UNKNOWN)
        else:
            return False
        return True

    def _attributes(self, instruction: dis.Instruction, frame: _Frame) -> bool:
        """Extend object paths and distinguish reads from writes."""
        op = instruction.opname
        arg = instruction.arg or 0
        name = instruction.argval
        if op in {'LOAD_ATTR', 'LOAD_METHOD'} and frame.stack:
            for root in sorted(frame.stack[-1].unresolved):
                raise ValueError(f'Unexpected attribute access: {root}.{name}')
        if op == 'LOAD_ATTR':
            receiver = frame.pop()
            if sys.version_info >= (3, 12) and arg & 1:
                self.dependencies.record(receiver)
                frame.stack.extend([_UNKNOWN, _UNKNOWN])
            else:
                paths = set()
                for path in receiver.paths:
                    if len(path) < self.path_limit:
                        paths.add((*path, name))
                    else:
                        self.dependencies.record(
                            _Value(paths=frozenset({path}))
                        )
                frame.stack.append(_Value(frozenset(paths), receiver.reads))
        elif op == 'LOAD_METHOD':
            self.dependencies.record(frame.pop())
            frame.stack.extend([_UNKNOWN, _UNKNOWN])
        elif op == 'STORE_ATTR':
            receiver, value = frame.pop(), frame.pop()
            targets = {(*path, name) for path in receiver.paths}
            # Match the existing AST behavior for bookkeeping such as
            # self._counter += 1, while retaining independent RHS reads.
            self.dependencies.record(
                _Value(reads=value.dependencies - (value.inplace & targets))
            )
        elif op == 'DELETE_ATTR':
            frame.pop()
        else:
            return False
        return True

    def _operators(self, instruction: dis.Instruction, frame: _Frame) -> bool:
        """Combine dependency values without extending computed results."""
        op = instruction.opname
        arg = instruction.arg or 0
        if op == 'COPY':
            frame.stack.append(
                frame.stack[-arg] if arg <= len(frame.stack) else _UNKNOWN
            )
        elif op == 'SWAP':
            if arg <= len(frame.stack):
                frame.stack[-1], frame.stack[-arg] = (
                    frame.stack[-arg],
                    frame.stack[-1],
                )
        elif op == 'BINARY_OP':
            right, left = frame.pop(), frame.pop()
            inplace = (
                left.paths - right.dependencies
                if instruction.argrepr.endswith('=')
                else frozenset()
            )
            frame.stack.append(
                _Value(
                    reads=left.dependencies | right.dependencies,
                    inplace=inplace,
                )
            )
        elif op.startswith('UNARY_') or op in {'TO_BOOL', 'CONVERT_VALUE'}:
            frame.stack.append(_Value(reads=frame.pop().dependencies))
        elif op in {'COMPARE_OP', 'IS_OP', 'CONTAINS_OP', 'BINARY_SUBSCR'}:
            left, right = frame.pop(), frame.pop()
            frame.stack.append(
                _Value(reads=left.dependencies | right.dependencies)
            )
        else:
            return False
        return True

    def _calls(self, instruction: dis.Instruction, frame: _Frame) -> bool:
        """Consume call arguments and normalize version-specific call layouts."""
        op = instruction.opname
        arg = instruction.arg or 0
        if op == 'PRECALL':
            # Python 3.11 accounts for argument removal in PRECALL;
            # CALL removes the callable and NULL/self slots afterward.
            self.dependencies.record(*frame.take(arg))
        elif op in {'CALL', 'CALL_KW', 'CALL_FUNCTION_EX'}:
            values = frame.take(1 - _stack_effect(instruction))
            self.dependencies.record(*values[:-2])
            # The two remaining slots contain the callable and NULL/self.
            # Expanded-argument calls can use ordinary LOAD_ATTR instead
            # of method-loading bytecode, so strip the called attribute.
            # Since 3.13 the callable always precedes NULL/self. Earlier
            # method loads already recorded the receiver in LOAD_METHOD.
            callable_index = -1 if sys.version_info >= (3, 13) else -2
            receiver_index = -2 if callable_index == -1 else -1
            self.dependencies.record(values[receiver_index])
            value = values[callable_index]
            self.dependencies.record(
                _Value(
                    paths=frozenset(path[:-1] for path in value.paths if path),
                    reads=value.reads,
                )
            )
            frame.stack.append(_UNKNOWN)
        elif op in {'CALL_INTRINSIC_1', 'CALL_INTRINSIC_2'}:
            self.dependencies.record(
                *frame.take(1 - _stack_effect(instruction))
            )
            frame.stack.append(_UNKNOWN)
        elif op == 'MAKE_FUNCTION':
            # Nested code is analyzed separately with captured bindings.
            values = frame.take(1 - _stack_effect(instruction))
            if sys.version_info < (3, 13) and arg & 8:
                del values[1]  # closure tuple, immediately below the code
            self.dependencies.record(*values)
            frame.stack.append(_UNKNOWN)
        elif op == 'SET_FUNCTION_ATTRIBUTE':
            function, attribute = frame.pop(), frame.pop()
            if arg != 8:  # captured cells are not reads of their values
                self.dependencies.record(attribute)
            frame.stack.append(function)
        else:
            return False
        return True

    def _containers(self, instruction: dis.Instruction, frame: _Frame) -> bool:
        """Collect reads from container construction and iteration."""
        op = instruction.opname
        arg = instruction.arg or 0
        if op in {
            'LIST_APPEND',
            'SET_ADD',
            'LIST_EXTEND',
            'SET_UPDATE',
            'DICT_UPDATE',
            'DICT_MERGE',
            'MAP_ADD',
        }:
            values = frame.take(2 if op == 'MAP_ADD' else 1)
            reads = frozenset().union(
                *(value.dependencies for value in values)
            )
            if arg <= len(frame.stack):
                frame.stack[-arg] = frame.stack[-arg].merge(
                    _Value(reads=reads)
                )
            else:
                self.dependencies.record(*values)
        elif op.startswith('BUILD_'):
            values = frame.take(1 - _stack_effect(instruction))
            frame.stack.append(
                _Value(
                    reads=frozenset().union(*(v.dependencies for v in values))
                )
            )
        elif op in {'UNPACK_SEQUENCE', 'UNPACK_EX'}:
            value = frame.pop()
            self.dependencies.record(value)
            frame.stack.extend([_UNKNOWN] * (1 + _stack_effect(instruction)))
        elif op in {
            'GET_ITER',
            'GET_YIELD_FROM_ITER',
            'GET_AITER',
            'GET_AWAITABLE',
        }:
            self.dependencies.record(frame.pop())
            frame.stack.append(_UNKNOWN)
        else:
            return False
        return True


def property_dependencies(
    prop: Callable | property, *, strict: bool = False
) -> FunctionDependencies:
    """Inspect bytecode for static attribute reads made by a property getter.

    Supports source-less functions, decorated getters, local aliases, branches,
    and nested functions. Calls are not executed or followed into other methods;
    dynamic getattr and attributes of computed objects need explicit dependencies.

    With ``strict=True``, attribute reads on unresolved global names raise
    ValueError. This is opt-in because globals can be bound after class creation.
    """
    if isinstance(prop, property):
        if prop.fget is None:
            raise ValueError('Property has no getter function')
        func = prop.fget
    else:
        func = prop
    if not inspect.isfunction(func):
        raise TypeError(f'Expected a callable or property, got {type(func)}')
    func = inspect.unwrap(func)
    if not inspect.isfunction(func):
        raise TypeError(f'Expected a Python function, got {type(func)}')
    parameters = tuple(inspect.signature(func).parameters)
    if len(parameters) != 1:
        raise ValueError(
            f'Expected a property getter with a single parameter, got {len(parameters)} parameters: {parameters}'
        )
    visitor = _BytecodeDependencies(func, strict=strict)
    visitor.scan(
        func.__code__, {parameters[0]: _SELF}, set(func.__code__.co_freevars)
    )
    return FunctionDependencies(
        attributes=frozenset(visitor.attributes),
        globals=frozenset(visitor.global_names),
        nonlocals=frozenset(visitor.nonlocal_names),
        builtins=frozenset(visitor.builtin_names),
    )


__all__ = ['FunctionDependencies', 'property_dependencies']
