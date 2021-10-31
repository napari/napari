import weakref

import pytest

from napari.utils.events import EventEmitter


def test_event_blocker_count_none():
    """Test event emitter block counter with no emission."""
    e = EventEmitter(type="test")
    with e.blocker() as block:
        pass
    assert block.count == 0


def test_event_blocker_count():
    """Test event emitter block counter with emission."""
    e = EventEmitter(type="test")
    with e.blocker() as block:
        e()
        e()
        e()
    assert block.count == 3


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
        def __init__(self):
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

    e = EventEmitter(type="test")

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


def test_no_event_arg():
    class TestOb:
        def __init__(self):
            self.count = 0

        def fun(self):
            self.count += 1

    count = [0]

    def simple_fun():
        count[0] += 1

    t = TestOb()

    e = EventEmitter(type="test")
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

    e = EventEmitter(type="test")
    with pytest.raises(RuntimeError):
        e.connect(t.fun)
    with pytest.raises(RuntimeError):
        e.connect(simple_fun)


def test_disconnect_object():
    class TestOb:
        call_list_1 = []
        call_list_2 = []

        def fun1(self):
            self.call_list_1.append(1)

        def fun2(self):
            self.call_list_2.append(1)

    t = TestOb()

    e = EventEmitter(type="test")
    e.connect(t.fun1)
    e.connect(t.fun2)
    e()

    assert t.call_list_1 == [1]
    assert t.call_list_2 == [1]

    e.disconnect(t)
    e()

    assert t.call_list_1 == [1]
    assert t.call_list_2 == [1]


def test_weakref_disconnect():
    class TestOb:
        call_list_1 = []

        def fun1(self):
            self.call_list_1.append(1)

    t = TestOb()

    e = EventEmitter(type="test")
    e.connect(t.fun1)
    e()

    assert t.call_list_1 == [1]
    e.disconnect((weakref.ref(t), "fun1"))
    e()
    assert t.call_list_1 == [1]
