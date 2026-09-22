import pytest

from napari.utils.events.migrations import deprecation_warning_event


def test_deprecation_warning_event() -> None:
    event = deprecation_warning_event(
        prefix='obj.events',
        previous_name='old',
        new_name='new',
        since_version='0.0.0',
        window='2027-Q1',
    )

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def add(self, event) -> None:
            self.count += event.value

    counter = Counter()
    msg = (
        'obj.events.old is deprecated since 0.0.0. It may be removed as early '
        'as 2027-Q1. Please use obj.events.new instead.'
    )

    with pytest.warns(FutureWarning, match=msg):
        event.connect(counter.add)

    event(value=1)

    assert counter.count == 1
