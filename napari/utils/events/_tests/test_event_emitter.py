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
