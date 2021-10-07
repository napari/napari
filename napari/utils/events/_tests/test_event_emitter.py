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


def test_no_event_arg():
    class TestOb:
        def fun(self):
            pass

    t = TestOb()

    e = EventEmitter(type="test")
    with pytest.raises(RuntimeError):
        e.connect(t.fun)
