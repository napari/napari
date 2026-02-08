import pytest

from napari.utils.events import EventEmitter
from napari.utils.events.migrations import deprecation_warning_event


def test_deprecation_warning_event() -> None:
    event = deprecation_warning_event(
        'obj.events', 'old', 'new', '0.1.0', '0.0.0'
    )

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def add(self, event) -> None:
            self.count += event.value

    counter = Counter()
    msg = (
        'obj.events.old is deprecated since 0.0.0 and will be removed in '
        '0.1.0. Please use obj.events.new'
    )

    with pytest.warns(FutureWarning, match=msg):
        event.connect(counter.add)

    event(value=1)

    assert counter.count == 1


def test_deprecation_warning_event_lazy_connect_from() -> None:
    source_event = EventEmitter(type_name='new')
    deprecated_event = deprecation_warning_event(
        'obj.events', 'old', 'new', '0.1.0', '0.0.0'
    )
    deprecated_event.connect_from(source_event)

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def add(self, _event) -> None:
            self.count += 1

    counter = Counter()
    msg = (
        'obj.events.old is deprecated since 0.0.0 and will be removed in '
        '0.1.0. Please use obj.events.new'
    )

    assert len(source_event.callbacks) == 0
    with pytest.warns(FutureWarning, match=msg):
        deprecated_event.connect(counter.add)
    assert len(source_event.callbacks) == 1

    source_event()
    assert counter.count == 1

    deprecated_event.disconnect(counter.add)
    assert len(source_event.callbacks) == 0
