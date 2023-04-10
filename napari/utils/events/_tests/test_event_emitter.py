import weakref
from functools import partial

import pytest

from napari.utils.events import EventEmitter


def test_event_blocker_count_none():
    """Test event emitter block counter with no emission."""
    e = EventEmitter(type_name="test")
    with e.blocker() as block:
        pass
    assert block.count == 0


def test_event_blocker_count():
    """Test event emitter block counter with emission."""
    e = EventEmitter(type_name="test")
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

        @rename("nonexist")
        def meth1(self, _event):
            self.m1 += 1

        @rename("meth1")
        def meth2(self, _event):
            self.m2 += 1

        def meth3(self):
            pass

        def meth4(self, _event):
            self.m4 += 1

    t = Test()

    e = EventEmitter(type_name="test")

    e.connect(t.meth1)
    e()
    assert (t.m1, t.m2) == (1, 0)

    e.connect(t.meth2)
    e()
    assert (t.m1, t.m2) == (2, 1)

    meth = t.meth3

    t.meth3 = "aaaa"

    with pytest.raises(RuntimeError):
        e.connect(meth)

    e.connect(t.meth4)
    assert t.m4 == 0
    e()
    assert t.m4 == 1
    t.meth4 = None
    with pytest.warns(RuntimeWarning, match="Problem with function"):
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

    fun1.__module__ = "napari.test.sample"
    fun3.__module__ = "napari.test.sample"
    fun5.__module__ = "napari.test.sample"

    e = EventEmitter(type_name="test")
    e.connect(fun1)
    e.connect(partial(fun2, val=2))
    e()
    assert res_li == [1, 2]
    res_li = []
    e.connect(fun3)
    e()
    assert res_li == [3, 1, 2]
    res_li = []
    e.connect(fun4)
    e()
    assert res_li == [3, 1, 4, 2]
    res_li = []
    e.connect(partial(fun5, val=5), position="last")
    e()
    assert res_li == [3, 1, 5, 4, 2]
    res_li = []
    e.connect(partial(fun6, val=6), position="last")
    e()
    assert res_li == [3, 1, 5, 4, 2, 6]


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

    Test.__module__ = "napari.test.sample"

    t1 = Test()
    t2 = Test2()

    e = EventEmitter(type_name="test")
    e.connect(t1.fun1)
    e.connect(t2.fun3)
    e()
    assert res_li == [1, 3]
    res_li = []
    e.connect(t1.fun2)
    e.connect(t2.fun4)
    e()
    assert res_li == [2, 1, 4, 3]


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

    e = EventEmitter(type_name="test")
    e.connect(t.fun)
    e.connect(simple_fun)
    e()
    assert t.count == 1
    assert count[0] == 1


def test_to_many_positional():
    class TestOb:
        def fun(self, a, b, c=1):
            pass

    def simple_fun(a, b):
        pass

    t = TestOb()

    e = EventEmitter(type_name="test")
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

    e = EventEmitter(type_name="test")
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

    e = EventEmitter(type_name="test")
    e.connect(t.fun1)
    e()

    assert t.call_list_1 == [1]
    e.disconnect((weakref.ref(t), "fun1"))
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

    e = EventEmitter(type_name="test")
    e.connect(fun1)
    e()
    assert count_list == [1]
    e.disconnect(None)
    e()
    assert count_list == [1]
    e.connect(fun2)
    e()
    assert count_list == [1, 2]
