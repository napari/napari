import weakref
from contextlib import nullcontext
from functools import partial
from unittest.mock import Mock

import pytest

from napari.utils.events import (
    EmitterGroup,
    EventEmitter,
    RenamedWarningEmitter,
    WarningEmitter,
)
from napari.utils.events.event import DependantEmitter


def test_event_blocker_count_none():
    """Test event emitter block counter with no emission."""
    e = EventEmitter(type_name='test')
    with e.blocker() as block:
        pass
    assert block.count == 0


def test_event_blocker_count():
    """Test event emitter block counter with emission."""
    e = EventEmitter(type_name='test')
    with e.blocker() as block:
        e()
        e()
        e()
    assert block.count == 3


def test_weakref_event_emitter():
    """
    We are testing that an event blocker does not keep hard reference to
    the object we are blocking, especially if it's a bound method.

    The reason it used to keep references is to get the count of how many time
    a callback was blocked, but if the object does not exists, then the bound method
    does not and thus there is no way to ask for it's count.

    so we can keep only weak refs.

    """
    e = EventEmitter(type_name='test_weak')

    class Obj:
        def cb(self):
            pass

    o = Obj()
    ref_o = weakref.ref(o)

    e.connect(o.cb)

    #
    with e.blocker(o.cb):
        e()

    del o
    assert ref_o() is None


@pytest.mark.parametrize('disconnect_and_should_be_none', [True, False])
def test_weakref_event_emitter_cb(disconnect_and_should_be_none):
    """

    Note that as above but with pure callback, We keep a reference to it, the
    reason is that unlike with bound method, the callback may be a closure and
    may not stick around.

    We thus expect the wekref to be None only if explicitely disconnected

    """
    e = EventEmitter(type_name='test_weak')

    def cb(self):
        pass

    ref_cb = weakref.ref(cb)

    e.connect(cb)

    with e.blocker(cb):
        e()

    if disconnect_and_should_be_none:
        e.disconnect(cb)
        del cb
        assert ref_cb() is None
    else:
        del cb
        assert ref_cb() is not None


def test_error_on_connect():
    """Check that connections happen correctly even on decorated methods.

    Some decorators will alter method.__name__, so that obj.method
    will not be equal to getattr(obj, obj.method.__name__). We check here
    that event binding will be correct even in these situations.
    """

    def rename(newname):
        def decorator(f):
            f.__name__ = newname
            return f

        return decorator

    class Test:
        def __init__(self) -> None:
            self.m1, self.m2, self.m4 = 0, 0, 0

        @rename('nonexist')
        def meth1(self, _event):
            self.m1 += 1

        @rename('meth1')
        def meth2(self, _event):
            self.m2 += 1

        def meth3(self):
            pass

        def meth4(self, _event):
            self.m4 += 1

    t = Test()

    e = EventEmitter(type_name='test')

    e.connect(t.meth1)
    e()
    assert (t.m1, t.m2) == (1, 0)

    e.connect(t.meth2)
    e()
    assert (t.m1, t.m2) == (2, 1)

    meth = t.meth3

    t.meth3 = 'aaaa'

    with pytest.raises(RuntimeError):
        e.connect(meth)

    e.connect(t.meth4)
    assert t.m4 == 0
    e()
    assert t.m4 == 1
    t.meth4 = None
    with pytest.warns(RuntimeWarning, match='Problem with function'):
        e()
    assert t.m4 == 1


def test_event_order_func():
    res_li = []

    def fun1():
        res_li.append(1)

    def fun2(val):
        res_li.append(val)

    def fun3():
        res_li.append(3)

    def fun4():
        res_li.append(4)

    def fun5(val):
        res_li.append(val)

    def fun6(val):
        res_li.append(val)

    fun1.__module__ = 'napari.test.sample'
    fun3.__module__ = 'napari.test.sample'
    fun5.__module__ = 'napari.test.sample'

    e = EventEmitter(type_name='test')
    e.connect(fun1)
    e.connect(partial(fun2, val=2))
    e()
    assert res_li == [1, 2]
    res_li = []
    e.connect(fun3)
    e()
    assert res_li == [1, 3, 2]
    res_li = []
    e.connect(fun4)
    e()
    assert res_li == [1, 3, 2, 4]
    res_li = []
    e.connect(partial(fun5, val=5), position='first')
    e()
    assert res_li == [5, 1, 3, 2, 4]
    res_li = []
    e.connect(partial(fun6, val=6), position='first')
    e()
    assert res_li == [5, 1, 3, 6, 2, 4]


def test_event_order_methods():
    res_li = []

    class Test:
        def fun1(self):
            res_li.append(1)

        def fun2(self):
            res_li.append(2)

    class Test2:
        def fun3(self):
            res_li.append(3)

        def fun4(self):
            res_li.append(4)

    Test.__module__ = 'napari.test.sample'

    t1 = Test()
    t2 = Test2()

    e = EventEmitter(type_name='test')
    e.connect(t1.fun1)
    e.connect(t2.fun3)
    e()
    assert res_li == [1, 3]
    res_li = []
    e.connect(t1.fun2)
    e.connect(t2.fun4)
    e()
    assert res_li == [1, 2, 3, 4]


def test_no_event_arg():
    class TestOb:
        def __init__(self) -> None:
            self.count = 0

        def fun(self):
            self.count += 1

    count = [0]

    def simple_fun():
        count[0] += 1

    t = TestOb()

    e = EventEmitter(type_name='test')
    e.connect(t.fun)
    e.connect(simple_fun)
    e()
    assert t.count == 1
    assert count[0] == 1


def test_to_many_positional():
    class TestOb:  # pragma: no cover
        def fun(self, a, b, c=1):
            pass

    def simple_fun(a, b):
        pass

    t = TestOb()

    e = EventEmitter(type_name='test')
    with pytest.raises(RuntimeError):
        e.connect(t.fun)
    with pytest.raises(RuntimeError):
        e.connect(simple_fun)


def test_disconnect_object():
    count_list = []

    def fun1():
        count_list.append(1)

    class TestOb:
        call_list_1 = []
        call_list_2 = []

        def fun1(self):
            self.call_list_1.append(1)

        def fun2(self):
            self.call_list_2.append(1)

    t = TestOb()

    e = EventEmitter(type_name='test')
    e.connect(t.fun1)
    e.connect(t.fun2)
    e.connect(fun1)
    e()

    assert t.call_list_1 == [1]
    assert t.call_list_2 == [1]
    assert count_list == [1]

    e.disconnect(t)
    e()

    assert t.call_list_1 == [1]
    assert t.call_list_2 == [1]
    assert count_list == [1, 1]


def test_weakref_disconnect():
    class TestOb:
        call_list_1 = []

        def fun1(self):
            self.call_list_1.append(1)

        def fun2(self, event):
            self.call_list_1.append(2)

    t = TestOb()

    e = EventEmitter(type_name='test')
    e.connect(t.fun1)
    e()

    assert t.call_list_1 == [1]
    e.disconnect((weakref.ref(t), 'fun1'))
    e()
    assert t.call_list_1 == [1]
    e.connect(t.fun2)
    e()
    assert t.call_list_1 == [1, 2]


def test_none_disconnect():
    count_list = []

    def fun1():
        count_list.append(1)

    def fun2(event):
        count_list.append(2)

    e = EventEmitter(type_name='test')
    e.connect(fun1)
    e()
    assert count_list == [1]
    e.disconnect(None)
    e()
    assert count_list == [1]
    e.connect(fun2)
    e()
    assert count_list == [1, 2]


def test_warning_emitter():
    mock = Mock()
    e = WarningEmitter(
        type_name='test', message='This is a warning', category=FutureWarning
    )

    with pytest.warns(FutureWarning, match='This is a warning'):
        e.connect(mock)

    assert e.callbacks

    mock.assert_not_called()
    e()

    mock.assert_called_once()


def test_renamed_emitter_simple():
    class DummyEventEmitter:
        def __init__(self, parent):
            self.new_event = EventEmitter(type_name='new_event')
            self.old_event = RenamedWarningEmitter(
                type_name='old_event',
                source_path='new_event',
                source=parent,
                message='Warning message',
            )

    class NewNamespace:
        def __init__(self):
            self.events = DummyEventEmitter(self)

    mock = Mock()

    n = NewNamespace()

    assert not n.events.new_event.callbacks

    with pytest.warns(FutureWarning, match='Warning message'):
        n.events.old_event.connect(mock)
    assert n.events.new_event.callbacks

    mock.assert_not_called()

    n.events.new_event(value=7)

    mock.assert_called_once()

    n.events.old_event.disconnect(mock)

    assert not n.events.new_event.callbacks


def test_renamed_emitter_composite():
    class DummyEventEmitterComposite:
        def __init__(self, parent):
            self.new_event = EventEmitter(type_name='new_event')

    class DummyEventEmitterBase:
        def __init__(self, parent):
            self.old_event = RenamedWarningEmitter(
                type_name='old_event',
                source_path='composite.new_event',
                source=parent,
                message='Warning message',
            )

    class CompositeNamespace:
        def __init__(self):
            self.events = DummyEventEmitterComposite(self)

    class OldNamespace:
        def __init__(self):
            self.events = DummyEventEmitterBase(self)
            self.composite = CompositeNamespace()

    mock = Mock()

    o = OldNamespace()

    assert not o.composite.events.new_event.callbacks
    with pytest.warns(FutureWarning, match='Warning message'):
        o.events.old_event.connect(mock)
    assert o.composite.events.new_event.callbacks

    mock.assert_not_called()

    o.composite.events.new_event()

    mock.assert_called_once()


def test_renamed_emitter_reconnects_nested_objects():
    class Leaf:
        def __init__(self, value=1):
            self.events = EmitterGroup(source=self, value=None)
            self._value = value

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, value):
            self._value = value
            self.events.value(value=value)

    class Branch:
        def __init__(self, leaf: Leaf):
            self.events = EmitterGroup(source=self, leaf=None)
            self._leaf = leaf

        @property
        def leaf(self):
            return self._leaf

        @leaf.setter
        def leaf(self, value):
            self._leaf = value
            self.events.leaf(value=value)

    class Root:
        def __init__(self, branch: Branch):
            self.events = EmitterGroup(source=self, branch=None)
            self._branch = branch

        @property
        def branch(self):
            return self._branch

        @branch.setter
        def branch(self, value):
            self._branch = value
            self.events.branch(value=value)

    root = Root(branch=Branch(leaf=Leaf()))
    alias = RenamedWarningEmitter(
        source=root,
        type_name='old_value',
        source_path='branch.leaf.value',
        message='renamed',
    )
    callback = Mock()
    with pytest.warns(FutureWarning, match='renamed'):
        alias.connect(callback)

    old_branch = root.branch
    old_leaf = old_branch.leaf
    callback.assert_not_called()
    root.branch = Branch(leaf=Leaf(value=2))
    callback.assert_called_once()
    assert callback.call_args.args[0].value == 2
    callback.reset_mock()
    old_leaf.value = 3
    callback.assert_not_called()
    assert not old_branch.events.leaf.callbacks
    assert not old_leaf.events.value.callbacks

    root.branch.leaf.value = 4
    callback.assert_called_once()
    assert callback.call_args.args[0].value == 4
    callback.reset_mock()

    previous_leaf = root.branch.leaf
    root.branch.leaf = Leaf(value=5)
    callback.assert_called_once()
    assert callback.call_args.args[0].value == 5
    callback.reset_mock()
    previous_leaf.value = 6
    callback.assert_not_called()
    root.branch.leaf.value = 7
    callback.assert_called_once()
    callback.reset_mock()

    # Replacing a parent invalidates the alias even when the value is equal.
    root.branch = Branch(leaf=Leaf(value=7))
    callback.assert_called_once()
    assert callback.call_args.args[0].value == 7

    alias.disconnect(callback)
    assert not root.events.branch.callbacks
    assert not root.branch.events.leaf.callbacks
    assert not root.branch.leaf.events.value.callbacks


@pytest.mark.parametrize('writable', [False, True])
def test_renamed_emitter_missing_replacement_event(writable):
    class Child:
        def __init__(self):
            self.events = EmitterGroup(source=self, value=None)

    class Root:
        def __init__(self):
            self._child = Child()

        @property
        def child(self):
            return self._child

        if writable:

            @child.setter
            def child(self, value):  # pragma: no cover
                self._child = value

    root = Root()
    alias = RenamedWarningEmitter(
        source=root,
        type_name='old_value',
        source_path='child.value',
        message='renamed',
    )
    callback = Mock()

    expectation = (
        pytest.warns(UserWarning, match='has no replacement event')
        if writable
        else nullcontext()
    )
    with expectation, pytest.warns(FutureWarning, match='renamed'):
        alias.connect(callback)

    root.child.events.value(value=2)
    callback.assert_called_once()
    alias.disconnect(callback)


def test_renamed_emitter_reconnects_property():
    from napari.utils.events import EmitterGroup

    class Child:
        def __init__(self, value=1):
            self.events = EmitterGroup(source=self, value=None)
            self.value = value

    class Root:
        def __init__(self):
            self.events = EmitterGroup(source=self, child=None)
            self._child = Child()

        @property
        def child(self):
            return self._child

        @child.setter
        def child(self, value):
            self._child = value
            self.events.child(value=value)

    root = Root()
    alias = RenamedWarningEmitter(
        source=root,
        type_name='old_value',
        source_path='child.value',
        message='renamed',
    )
    callback = Mock()
    with pytest.warns(FutureWarning, match='renamed'):
        alias.connect(callback)
    old_child = root.child
    root.child = Child(value=3)
    assert not old_child.events.value.callbacks
    callback.assert_called_once()
    assert callback.call_args.args[0].value == 3
    callback.reset_mock()
    root.child.events.value(value=2)
    callback.assert_called_once()
    alias.disconnect(callback)
    assert not root.events.child.callbacks
    assert not root.child.events.value.callbacks


@pytest.mark.parametrize(
    ('old_has_event', 'new_has_event'),
    [(False, False), (False, True), (True, False)],
    ids=['eventless-step', 'event-appears', 'event-disappears'],
)
def test_renamed_emitter_reconnects_across_event_interfaces(
    old_has_event, new_has_event
):
    class Leaf:
        def __init__(self, value):
            self.value = value
            self.events = EmitterGroup(source=self, value=None)

    class Branch:
        def __init__(self, value, has_event):
            self._leaf = Leaf(value)
            self.events = EmitterGroup(source=self)
            if has_event:
                self.events.add(leaf=None)

        @property
        def leaf(self):
            return self._leaf

    class Root:
        def __init__(self, branch):
            self._branch = branch
            self.events = EmitterGroup(source=self, branch=None)

        @property
        def branch(self):
            return self._branch

        @branch.setter
        def branch(self, value):
            self._branch = value
            self.events.branch(value=value)

    old_branch = Branch(1, old_has_event)
    root = Root(old_branch)
    alias = RenamedWarningEmitter(
        source=root,
        type_name='old_value',
        source_path='branch.leaf.value',
        message='renamed',
    )
    callback = Mock()
    with pytest.warns(FutureWarning, match='renamed'):
        alias.connect(callback)

    root.branch = Branch(2, new_has_event)
    callback.assert_called_once()
    assert callback.call_args.args[0].value == 2
    callback.reset_mock()
    assert not old_branch.leaf.events.value.callbacks
    if old_has_event:
        assert not old_branch.events.leaf.callbacks

    old_branch.leaf.events.value(value=3)
    callback.assert_not_called()
    root.branch.leaf.events.value(value=4)
    callback.assert_called_once()
    assert callback.call_args.args[0].value == 4
    if new_has_event:
        assert root.branch.events.leaf.callbacks

    alias.disconnect(callback)
    assert not root.events.branch.callbacks
    assert not root.branch.leaf.events.value.callbacks
    if new_has_event:
        assert not root.branch.events.leaf.callbacks


def test_dependant_emitter():
    class A:
        def __init__(self):
            self.events = EmitterGroup(source=self, a=None, b=None)
            self._a = 1

        @property
        def a(self):
            return self._a

        @a.setter
        def a(self, value):
            self._a = value
            self.events.a(value=value)

    class B:
        def __init__(self):
            self.events = EmitterGroup(
                source=self,
                a=None,
                b=None,
                aa=DependantEmitter(
                    sources_list=['a.a', 'b.a'],
                    property_name='aa',
                    type_name='aa',
                ),
            )
            self._a = A()
            self._b = A()

        @property
        def a(self):
            return self._a

        @a.setter
        def a(self, value):
            self._a = value
            self.events.a(value=value)

        @property
        def b(self):
            return self._b

        @b.setter
        def b(self, value):  # pragma: no cover
            self._b = value
            self.events.b(value=value)

        @property
        def aa(self):
            return self.a.a + self.b.a

        @aa.setter
        def aa(self, value):
            self.a.a = value / 2
            self.b.a = value / 2

    b = B()

    assert b.aa == 2
    b.a.a = 2
    assert b.aa == 3

    mock = Mock()
    b.events.aa.connect(mock)
    b.a.a = 3
    assert mock.call_args.args[0].value == 4
    mock.assert_called_once()
    b.b.a = 2
    assert mock.call_count == 2
    assert mock.call_args.args[0].value == 5

    b.a = A()
    assert mock.call_count == 3
    assert mock.call_args.args[0].value == 3

    b.a.a = 3
    assert mock.call_count == 4
    assert mock.call_args.args[0].value == 5

    b.aa = 10
    assert b.a.a == 5
    assert b.b.a == 5
    assert (
        mock.call_count == 6
    )  # double emission because of two events being set
    assert mock.call_args.args[0].value == 10
