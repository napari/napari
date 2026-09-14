import builtins
import math

import pytest

from napari.utils.events._ast_visitor import property_dependencies

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


@pytest.mark.xfail(reason='Need to decide')
def test_prop_with_wrong_code_analysis():
    with pytest.raises(
        ValueError, match=r'Unexpected attribute access: self2.a'
    ):
        property_dependencies(A.prop_with_wrong_code)


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
