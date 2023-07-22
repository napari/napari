import pytest

from napari.utils.events.migrations import deprecation_warning_event


def test_deprecation_warning_event() -> None:
    event = deprecation_warning_event(
        "obj.events", "old", "new", "0.1.0", "0.0.0"
    )

    def _print(msg: str) -> None:
        print(msg)

    with pytest.warns(FutureWarning):
        event.connect(_print)
        event(msg="test")
