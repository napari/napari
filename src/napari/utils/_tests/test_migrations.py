import re

import pytest

from napari.utils.migrations import (
    RenamedProperty,
    _DeprecatingDict,
    add_deprecated_property,
    deprecated_class_name,
    rename_argument,
)


def test_simple():
    @rename_argument('a', 'b', '1', '0.5')
    def sample_fun(b):
        return b

    assert sample_fun(1) == 1
    assert sample_fun(b=1) == 1
    with pytest.deprecated_call():
        assert sample_fun(a=1) == 1
    with pytest.raises(ValueError, match='already defined'):
        sample_fun(b=1, a=1)


def test_constructor():
    class Sample:
        @rename_argument('a', 'b', '1', '0.5')
        def __init__(self, b) -> None:
            self.b = b

    assert Sample(1).b == 1
    assert Sample(b=1).b == 1
    with pytest.deprecated_call():
        assert Sample(a=1).b == 1


@pytest.mark.parametrize('due_date', [None, 'spring 2027'])
def test_deprecated_property(due_date) -> None:
    @add_deprecated_property(
        previous_name='old_property',
        new_name='new_property',
        since_version='0.0.0',
        due_date=due_date,
    )
    class Dummy:
        def __init__(self) -> None:
            self._value = 0

        @property
        def new_property(self) -> int:
            return self._value

        @new_property.setter
        def new_property(self, value: int) -> int:
            self._value = value

    instance = Dummy()

    assert instance.new_property == 0

    instance.new_property = 1

    assert isinstance(Dummy.old_property, RenamedProperty)
    msg = 'Dummy.old_property is deprecated since 0.0.0.'
    if due_date is not None:
        msg += f' Removal is scheduled for {due_date}.'
    msg += ' Please use new_property instead.'
    assert Dummy.old_property.message.endswith(msg)
    msg2 = 'Dummy.events.old_property is deprecated since 0.0.0.'
    if due_date is not None:
        msg2 += f' Removal is scheduled for {due_date}.'
    msg2 += ' Please use .*events.new_property instead.'
    assert re.search(msg2, Dummy.old_property.event_message)

    with pytest.warns(FutureWarning, match=msg):
        assert instance.old_property == 1

    with pytest.warns(FutureWarning, match=msg):
        instance.old_property = 2

    assert instance.new_property == 2


@pytest.mark.parametrize('due_date', [None, 'fall 2027'])
def test_deprecated_property_legacy_version(due_date):

    with pytest.warns(
        FutureWarning, match="'version'.*deprecated.*ignored"
    ) as recorded:
        decorator = add_deprecated_property(
            previous_name='old',
            new_name='new',
            due_date=due_date,
            version='0.1.0',
            since_version='0.0.0',
        )
    assert recorded[0].filename == __file__

    class Sample:
        new = 1

    assert decorator(Sample) is Sample
    assert isinstance(Sample.old, RenamedProperty)
    assert '0.1.0' not in Sample.old.message
    assert 'since 0.0.0' in Sample.old.message
    assert ('Removal is scheduled' in Sample.old.message) == (
        due_date is not None
    )
    if due_date is not None:
        assert due_date in Sample.old.message
    instance = Sample()
    with pytest.warns(FutureWarning, match='Sample.old is deprecated'):
        assert instance.old == 1
    with pytest.warns(FutureWarning, match='Sample.old is deprecated'):
        instance.old = 2
    assert instance.new == 2


def test_deprecated_property_legacy_version_old_usage():
    class Sample:
        new = 1

    with pytest.warns(
        FutureWarning,
        match='Using positional arguments for add_deprecated_property',
    ):
        add_deprecated_property(
            Sample, 'old', 'new', version='0.1.0', since_version='0.0.0'
        )
    assert Sample.old is not None

    with pytest.warns(
        FutureWarning,
        match="Using 'obj' keyword argument for add_deprecated_property",
    ):
        add_deprecated_property(
            obj=Sample,
            previous_name='old2',
            new_name='new',
            version='0.1.0',
            since_version='0.0.0',
        )
    assert Sample.old2 is not None


@pytest.mark.parametrize(
    ('previous_name', 'new_name', 'message'),
    [
        ('value', 'target', 'value property already exists'),
        ('old', 'missing', 'missing property must exist'),
    ],
)
def test_deprecated_property_invalid_names(previous_name, new_name, message):
    class Sample:
        value = 1
        target = 2

    with pytest.raises(RuntimeError, match=message):
        add_deprecated_property(
            previous_name=previous_name, new_name=new_name
        )(Sample)
    assert 'old' not in vars(Sample)


def test_deprecated_class_name():
    """Test the deprecated class name function."""

    class macOS:
        pass

    MacOSX = deprecated_class_name(
        macOS, 'MacOSX', version='10.12', since_version='10.11'
    )

    with pytest.warns(FutureWarning, match='deprecated.*macOS'):
        _os = MacOSX()

    with pytest.warns(FutureWarning, match='deprecated.*macOS'):

        class MacOSXServer(MacOSX):
            pass


def test_deprecating_dict_with_renamed_in_deprecated_keys():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )
    assert 'c' in d.deprecated_keys


def test_deprecating_dict_with_renamed_getitem_deprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert d['c'] == 1


def test_deprecating_dict_with_renamed_get_deprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert d.get('c') == 1


def test_deprecating_dict_with_renamed_set_nondeprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )

    d['a'] = 3

    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert d['c'] == 3


def test_deprecating_dict_with_renamed_set_deprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )

    with pytest.warns(FutureWarning, match='is deprecated since'):
        d['c'] = 3

    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert d['c'] == 3
    assert d['a'] == 3


def test_deprecating_dict_with_renamed_update_nondeprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )

    d.update({'a': 3})

    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert d['c'] == 3


def test_deprecating_dict_with_renamed_update_deprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )

    with pytest.warns(FutureWarning, match='is deprecated since'):
        d.update({'c': 3})

    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert d['c'] == 3
    assert d['a'] == 3


def test_deprecating_dict_with_renamed_del_nondeprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )
    assert 'a' in d
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' in d

    with pytest.warns(FutureWarning, match='is deprecated since'):
        del d['c']

    assert 'a' not in d
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' not in d


def test_deprecating_dict_with_renamed_del_deprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' in d
    assert 'a' in d

    with pytest.warns(FutureWarning, match='is deprecated since'):
        del d['c']

    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' not in d
    assert 'a' not in d


def test_deprecating_dict_with_renamed_pop_nondeprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )
    assert 'a' in d
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' in d

    with pytest.warns(FutureWarning, match='is deprecated since'):
        d.pop('c')

    assert 'a' not in d
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' not in d


def test_deprecating_dict_with_renamed_pop_deprecated():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )
    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' in d
    assert 'a' in d

    with pytest.warns(FutureWarning, match='is deprecated since'):
        d.pop('c')

    with pytest.warns(FutureWarning, match='is deprecated since'):
        assert 'c' not in d
    assert 'a' not in d


def test_deprecating_dict_with_renamed_copy():
    d = _DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        from_name='c', to_name='a', version='v2.0', since_version='v1.6'
    )

    e = d.copy()

    assert d is not e
    assert e.data == d.data
    assert e.deprecated_keys == d.deprecated_keys


def test_deprecated_property_descriptor():
    class Sample:
        def __init__(self):
            self._value = 0

        @property
        def new_property(self):
            return self._value

        @new_property.setter
        def new_property(self, value):
            self._value = value

        old_property = RenamedProperty(
            new_name='new_property',
            since_version='0.1.0',
        )

    instance = Sample()

    assert instance.new_property == 0

    instance.new_property = 1

    msg = 'Sample.old_property is deprecated since 0.1.0. Please use new_property'

    with pytest.warns(FutureWarning, match=msg):
        assert instance.old_property == 1

    with pytest.warns(FutureWarning, match=msg):
        instance.old_property = 2

    assert Sample.old_property.category is FutureWarning
    assert Sample.old_property.new_name == 'new_property'
    assert re.match(
        '.*Sample.events.old_property is deprecated since 0.1.0. Please use .*Sample.events.new_property instead.',
        Sample.old_property.event_message,
    )
    assert re.match(
        '.*Sample.old_property is deprecated since 0.1.0. Please use new_property instead.',
        Sample.old_property.message,
    )
    assert Sample.old_property.name == 'old_property'

    assert instance.new_property == 2


@pytest.mark.parametrize('writable', [False, True])
def test_renamed_property_accessors(writable):
    class Sample:
        value = 1
        old = RenamedProperty(
            new_name='value', since_version='0.1.0', writable=writable
        )

    prop = Sample.old
    assert prop.fget is not None
    assert (prop.fset is not None) == writable
    instance = Sample()
    with pytest.warns(
        FutureWarning, match='Sample.old is deprecated'
    ) as recorded:
        assert prop.fget(instance) == 1
    assert recorded[0].filename == __file__

    if writable:
        with pytest.warns(FutureWarning, match='Sample.old is deprecated'):
            prop.fset(instance, 2)
        assert instance.value == 2


def test_deprecated_property_descriptor_nested():
    class SubSample:
        def __init__(self):
            self.value = 0

    class Sample:
        def __init__(self):
            self.subsample = SubSample()

        old_property = RenamedProperty(
            new_name='subsample.value',
            since_version='0.1.0',
            due_date='fall 2027',
        )

    instance = Sample()

    assert instance.subsample.value == 0

    with pytest.warns(
        FutureWarning,
        match='Sample.old_property is deprecated since 0.1.0. Removal is scheduled for fall 2027. Please use subsample.value instead.',
    ):
        assert instance.old_property == 0

    with pytest.warns(
        FutureWarning, match='Sample.old_property is deprecated since 0.1.0'
    ):
        instance.old_property = 1

    assert instance.subsample.value == 1


def test_deprecated_property_descriptor_no_writing():
    class Sample:
        def __init__(self):
            self._value = 0

        @property
        def new_property(self):  # pragma: no cover
            return self._value

        old_property = RenamedProperty(
            new_name='new_property', since_version='0.1.0', writable=False
        )

    instance = Sample()

    with pytest.raises(AttributeError, match='has no setter'):
        instance.old_property = 1

    with pytest.raises(AttributeError, match='has no setter'):
        instance.new_property = 2
