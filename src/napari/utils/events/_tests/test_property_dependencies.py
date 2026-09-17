import builtins
import math

import pytest

from napari.utils.events._property_dependencies import property_dependencies

D = 1


class B:  # pragma: no cover
    def __init__(self):
        self.a = 1


class A:  # pragma: no cover
    def __init__(self):
        self.a = 1
        self.d = 4
        self._counter = 0
        self.s = 'aaa'
        self.b = B()

    @property
    def prop(self):
        return self.a + self.b.a + D

    @property
    def simple(self) -> builtins.int:
        return self.a * 2

    @property
    def sum_of_a_and_d(self):
        return self.a + self.d

    @property
    def sum_with_non_self_name(self2):
        return self2.a + self2.d

    @property
    def prop_with_wrong_code(self):
        return self.a + self2.a  # noqa: F821

    @property
    def using_math(self):
        return self.a * math.pi

    @property
    def prop_with_counter(self):
        self._counter += 1
        return self.a

    @property
    def using_nested_function(self):
        def helper():
            return self.b.a

        return helper()

    @property
    def using_builtin_attribute(self):
        return int.bit_length(self.a)

    @property
    def normalized_string(self):
        return self.s.lower()


def two_args_function(a, b):  # pragma: no cover
    return a + b


def class_ggenerator():
    z = B()

    class C:  # pragma: no cover
        def __init__(self):
            self.a = 1

        @property
        def prop(self):
            return self.a + z.a

    return C


def test_simple_analysis():
    deps = property_dependencies(A.simple)
    assert deps == {'attributes': {'a'}}


def test_sum_of_a_and_d_analysis():
    deps = property_dependencies(A.sum_of_a_and_d)
    assert deps == {'attributes': {'a', 'd'}}


def test_sum_with_non_self_name_analysis():
    deps = property_dependencies(A.sum_with_non_self_name)
    assert deps == {'attributes': {'a', 'd'}}


def test_prop_with_wrong_code_analysis():
    with pytest.raises(
        ValueError, match=r'Unexpected attribute access: self2\.a'
    ):
        property_dependencies(A.prop_with_wrong_code, strict=True)


def test_unresolved_globals_are_allowed_by_default():
    assert property_dependencies(A.prop_with_wrong_code).attributes == {'a'}


def test_strict_analysis_with_known_global():
    assert property_dependencies(A.using_math, strict=True) == {
        'attributes': {'a'},
        'globals': {'math'},
    }


@pytest.mark.parametrize('expression', ['missing.value', 'missing.method()'])
def test_strict_analysis_with_unresolved_alias(expression):
    namespace = {}
    exec(
        f'def getter(self):\n    missing = self2\n    return {expression}',
        namespace,
    )
    with pytest.raises(
        ValueError, match=r'Unexpected attribute access: self2\.'
    ):
        property_dependencies(namespace['getter'], strict=True)


def test_complex_analysis():
    deps = property_dependencies(A.prop)
    assert deps == {'attributes': {'a', 'b.a'}, 'globals': {'D'}}


def test_using_math_analysis():
    deps = property_dependencies(A.using_math)
    assert deps == {'attributes': {'a'}, 'globals': {'math'}}


def test_prop_with_counter_analysis():
    deps = property_dependencies(A.prop_with_counter)
    assert deps == {'attributes': {'a'}}


def test_nested_function_dependencies_are_included():
    assert property_dependencies(A.using_nested_function) == {
        'attributes': {'b.a'},
    }


def test_two_args_function_dependencies():
    with pytest.raises(
        ValueError, match='Expected a property getter with a single parameter'
    ):
        property_dependencies(two_args_function)


def test_exception_for_non_callable():
    with pytest.raises(TypeError, match='Expected a callable or property'):
        property_dependencies(42)


def test_class_generator_dependencies():
    C = class_ggenerator()
    deps = property_dependencies(C.prop)
    assert deps == {'attributes': {'a'}, 'nonlocals': {'z'}}


def test_using_builtin_attribute_analysis():
    deps = property_dependencies(A.using_builtin_attribute)
    assert deps == {'attributes': {'a'}, 'builtins': {'int'}}


def test_normalized_string_analysis():
    deps = property_dependencies(A.normalized_string)
    assert deps == {'attributes': {'s'}}


@pytest.mark.parametrize(
    ('body', 'expected'),
    [
        ('return self.child.value', {'child.value'}),
        ('return self.child, self.child.value', {'child', 'child.value'}),
        (
            'return self.child.value + self.other.value',
            {'child.value', 'other.value'},
        ),
        ('child = self.child\nreturn child.value', {'child.value'}),
        (
            'return (self.left if self.flag else self.right).value',
            {'left.value', 'flag', 'right.value'},
        ),
        (
            'return self.left.value if self.flag else self.right.value',
            {'left.value', 'flag', 'right.value'},
        ),
        ('return self.left or self.right.value', {'left', 'right.value'}),
        ('return self.value.copy()', {'value'}),
        (
            'return self.child.method(*self.args, **self.kwargs)',
            {'child', 'args', 'kwargs'},
        ),
        (
            'return self.value.method(self.other.value, flag=self.flag)',
            {'value', 'other.value', 'flag'},
        ),
        (
            'self.counter += self.increment\nreturn self.value',
            {'increment', 'value'},
        ),
        (
            'self.counter = self.other.value\nreturn self.value',
            {'other.value', 'value'},
        ),
        ('del self.counter\nreturn self.value', {'value'}),
        ('self.counter += 1\nreturn self.counter', {'counter'}),
        (
            'self.counter += self.counter\nreturn self.value',
            {'counter', 'value'},
        ),
        (
            'for x in self.items:\n    self.counter += 1\nreturn self.value',
            {'items', 'value'},
        ),
        (
            'while self.flag:\n    self.counter += 1\nreturn self.value',
            {'flag', 'value'},
        ),
        (
            'try:\n    return self.child.value\nexcept AttributeError:\n    return self.fallback',
            {'child.value', 'fallback'},
        ),
        ('return [self.value for x in self.items]', {'value', 'items'}),
        ('return (self.value for x in self.items)', {'value', 'items'}),
        (
            'def helper():\n    return self.child.value\nreturn helper()',
            {'child.value'},
        ),
        (
            'child = self.child\ndef helper():\n    return child.value\nreturn helper()',
            {'child.value'},
        ),
        (
            'child = self.a.b.c\ndef helper():\n    return child.value\nreturn helper()',
            {'a.b.c.value'},
        ),
        (
            'def helper(x=self.default):\n    return self.value\nreturn helper()',
            {'default', 'value'},
        ),
        (
            'def helper(self):\n    return self.unrelated\nreturn self.value',
            {'value'},
        ),
        (
            'def helper():\n    def inner():\n        return self.child.value\n    return inner()\nreturn helper()',
            {'child.value'},
        ),
        ('return f"value: {self.child.value}"', {'child.value'}),
        ('return self.values[self.index]', {'values', 'index'}),
        ('return self.values[:self.end]', {'values', 'end'}),
    ],
)
def test_source_less_attribute_dependencies(body, expected):
    """Exercise actual source-less bytecode, including compiler optimizations."""
    namespace = {}
    exec(
        'def getter(self):\n'
        + '\n'.join('    ' + line for line in body.splitlines()),
        namespace,
    )
    assert property_dependencies(namespace['getter']).attributes == expected


def test_wrapped_source_less_getter():
    from functools import wraps

    namespace = {}
    exec('def getter(self):\n    return self.child.value', namespace)
    getter = namespace['getter']

    @wraps(getter)
    def wrapped(*args, **kwargs):
        return getter(*args, **kwargs)

    assert property_dependencies(property(wrapped)).attributes == {
        'child.value'
    }


def test_nested_external_names():
    other = B()

    def getter(self):
        def helper():
            return self.value + other.a + math.pi + len(())

        return helper()

    assert property_dependencies(getter) == {
        'attributes': {'value'},
        'nonlocals': {'other'},
        'globals': {'math'},
        'builtins': {'len'},
    }


def test_getter_is_not_executed():
    def getter(self):
        raise RuntimeError(self.value)

    assert property_dependencies(getter).attributes == {'value'}


def test_property_without_getter():
    with pytest.raises(ValueError, match='no getter'):
        property_dependencies(property())


def test_loop_with_growing_alias_terminates():
    def getter(self):
        node = self.child
        while self.flag:
            node = node.child
        return node.value

    deps = property_dependencies(getter).attributes
    assert 'flag' in deps
    assert 'child.value' in deps
