import sys

import pytest

from napari.layers import Points
from napari.utils._test_utils import MouseEvent, read_only_mouse_event
from napari.utils.interactions import (
    Shortcut,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


@pytest.mark.parametrize(
    ('shortcut', 'reason'),
    [
        ('Atl-A', 'Alt misspelled'),
        ('Ctrl-AA', 'AA makes no sense'),
        ('BB', 'BB makes no sense'),
    ],
)
def test_shortcut_invalid(shortcut, reason):
    with pytest.warns(UserWarning, match='does not seem to be a valid'):
        Shortcut(shortcut)  # Should be Control-A


def test_minus_shortcut():
    """
    Misc tests minus is properly handled as it is the delimiter
    """
    assert str(Shortcut('-')) == '-'
    assert str(Shortcut('Control--')).endswith('-')
    assert str(Shortcut('Shift--')).endswith('-')


def test_shortcut_qt():
    assert Shortcut('Control-A').qt == 'Ctrl+A'


@pytest.mark.skipif(
    sys.platform != 'darwin', reason='Parsing macos specific keys'
)
@pytest.mark.parametrize(
    ('expected', 'shortcut'),
    [
        ('␣', 'Space'),
        ('⌥', 'Alt'),
        ('⌥-', 'Alt--'),
        ('⌘', 'Meta'),
        ('⌘-', 'Meta--'),
        ('⌘⌥', 'Meta-Alt'),
        ('⌥⌘P', 'Meta-Alt-P'),
    ],
)
def test_partial_shortcuts(shortcut, expected):
    assert str(Shortcut(shortcut)) == expected


def _drag_logger(log):
    def drag(layer, event):
        log.append('press')
        yield
        while event.type == 'mouse_move':
            log.append('move')
            yield
        log.append('release')

    return drag


def test_generator_exception_warns_and_is_cleaned_up():
    layer = Points()

    def drag(layer, event):
        yield
        raise ValueError('boom')

    layer.mouse_drag_callbacks.append(drag)
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    with pytest.warns(UserWarning, match=r'(?s)Traceback.*boom'):
        mouse_move_callbacks(
            layer, read_only_mouse_event(type='mouse_move', is_dragging=True)
        )
    assert not layer._mouse_drag_gen
    assert not layer._persisted_mouse_event


def test_release_cleans_up_when_generator_raises():
    layer = Points()

    def drag(layer, event):
        yield
        while event.type == 'mouse_move':
            yield
        raise ValueError('boom on release')

    layer.mouse_drag_callbacks.append(drag)
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    with pytest.warns(UserWarning, match='boom on release'):
        mouse_release_callbacks(
            layer, read_only_mouse_event(type='mouse_release')
        )
    assert not layer._mouse_drag_gen
    assert not layer._persisted_mouse_event

    # the next press starts a new drag
    log = []
    layer.mouse_drag_callbacks[:] = [_drag_logger(log)]
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    assert log == ['press']


def test_active_generator_accepts_unwrapped_events():
    """Callers (e.g. plugin tests) may pass plain, non-wrapped events."""
    layer = Points()
    log = []
    layer.mouse_drag_callbacks.append(_drag_logger(log))

    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    mouse_move_callbacks(
        layer, MouseEvent(type='mouse_move', is_dragging=True)
    )
    mouse_release_callbacks(layer, MouseEvent(type='mouse_release'))

    assert log == ['press', 'move', 'release']


def test_failing_close_does_not_leave_generators_registered():
    layer = Points()

    def drag(layer, event):
        try:
            while True:
                yield
        finally:
            raise ValueError('boom in finally')

    layer.mouse_drag_callbacks.append(drag)
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    with pytest.warns(UserWarning, match='boom in finally'):
        mouse_release_callbacks(
            layer, read_only_mouse_event(type='mouse_release')
        )
    assert not layer._mouse_drag_gen
    assert not layer._persisted_mouse_event
