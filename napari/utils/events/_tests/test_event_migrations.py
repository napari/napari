import pytest

from napari.utils.events.migrations import deprecation_warning_event


def test_deprecation_warning_event() -> None:
    event = deprecation_warning_event(
        "obj.events", "old", "new", "0.1.0", "0.0.0"
    )

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def add(self, event) -> None:
            self.count += event.value

    counter = Counter()
    msg = "obj.events.old is deprecated since 0.0.0 and will be removed in 0.1.0. Please use obj.events.new"

    with pytest.warns(FutureWarning, match=msg):
        event.connect(counter.add)
        event(value=1)

    assert counter.count == 1
