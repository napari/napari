import pytest

from napari.utils.events.migrations import deprecation_warning_event


def test_deprecation_warning_event() -> None:
    event = deprecation_warning_event("obj.events", "old", "new", "0.0.1")
    event.connect(lambda x: print(x))

    with pytest.deprecated_call():
        event("test")
