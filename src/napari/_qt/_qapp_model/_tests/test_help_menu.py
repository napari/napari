"""For testing the Help menu"""

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import requests
from qtpy.QtCore import QRect, Qt
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from napari._app_model import get_app_model
from napari._qt._qapp_model.qactions._help import HELP_URLS, _start_viewer_tour
from napari._qt.widgets.qt_viewer_tour import (
    _BUILTIN_TOUR_TARGETS,
    _TOOLTIP_WIDTH,
    GuidedTour,
    TourAnchor,
    TourStep,
    _TourTooltip,
    build_viewer_tour,
    resolve_tour_target,
)


@pytest.mark.parametrize('url', HELP_URLS.keys())
def test_help_urls(url):
    if url == 'release_notes':
        pytest.skip('No release notes for dev version')

    r = requests.head(HELP_URLS[url])
    r.raise_for_status()


@pytest.mark.parametrize(
    'action_id',
    [
        'napari.window.help.info',
        'napari.window.help.about_macos',
    ]
    if sys.platform == 'darwin'
    else ['napari.window.help.info'],
)
def test_about_action(make_napari_viewer, action_id):
    app = get_app_model()
    viewer = make_napari_viewer()

    with mock.patch(
        'napari._qt.dialogs.qt_about.QtAbout.showAbout'
    ) as mock_about:
        app.commands.execute_command(action_id)
    mock_about.assert_called_once_with(viewer.window._qt_window)


def test_tour_tooltip_widens_for_long_nav_labels(qtbot):
    """A fixed width sized for English labels could crop longer translations."""
    parent = QWidget()
    qtbot.addWidget(parent)
    tooltip = _TourTooltip(parent)
    qtbot.addWidget(tooltip)

    tooltip.set_content(
        'Title', 'Body.', 2, 5
    )  # middle step: all 3 nav buttons visible
    assert tooltip.width() == _TOOLTIP_WIDTH

    tooltip._back.setText('Previous step in the guided tour')
    tooltip._next.setText('Next step in the guided tour')
    tooltip._skip.setText('Skip this entire guided tour now')
    tooltip._update_size()

    assert tooltip.width() > _TOOLTIP_WIDTH
    for button in (tooltip._back, tooltip._next, tooltip._skip):
        assert (
            button.geometry().width() >= button.minimumSizeHint().width() - 2
        )

    # width should shrink back down once labels return to normal
    tooltip.set_content('Title', 'Body.', 3, 5)
    assert tooltip.width() == _TOOLTIP_WIDTH


def test_tour_tooltip_center_anchor(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    tooltip = _TourTooltip(parent)
    qtbot.addWidget(tooltip)
    tooltip.set_content('Title', 'Body.', 1, 5)

    target_rect = QRect(100, 200, 1000, 800)
    bounds = QRect(0, 0, 2000, 1500)
    tooltip.place(target_rect, TourAnchor.CENTER, bounds)

    placed_center = tooltip.geometry().center()
    target_center = target_rect.center()
    assert abs(placed_center.x() - target_center.x()) <= 1
    assert abs(placed_center.y() - target_center.y()) <= 1


def test_build_viewer_tour_first_step_is_centered(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_window = viewer.window._qt_window
    tour = build_viewer_tour(qt_window, sample=None)

    first_step = tour._steps[0]
    assert first_step.target() is qt_window._qt_viewer.canvas.native
    assert first_step.anchor == TourAnchor.CENTER


def test_tour_tooltip_next_back_are_click_only(qtbot):
    """N/P keybinds were dropped as confusingly focus-dependent; clicking
    the buttons is the only way to navigate now (Escape stays global)."""
    parent = QWidget()
    qtbot.addWidget(parent)
    tooltip = _TourTooltip(parent)
    qtbot.addWidget(tooltip)
    tooltip.set_content('Title', 'Body.', 2, 5)

    with qtbot.assertNotEmitted(tooltip.next_clicked):
        qtbot.keyPress(tooltip, Qt.Key.Key_N)
    with qtbot.assertNotEmitted(tooltip.back_clicked):
        qtbot.keyPress(tooltip, Qt.Key.Key_P)

    with qtbot.waitSignal(tooltip.next_clicked, timeout=1000):
        qtbot.mouseClick(tooltip._next, Qt.MouseButton.LeftButton)
    with qtbot.waitSignal(tooltip.back_clicked, timeout=1000):
        qtbot.mouseClick(tooltip._back, Qt.MouseButton.LeftButton)


def test_tour_escape_closes_regardless_of_focus(qtbot):
    """Esc used to require the tooltip to have focus to close the tour."""
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 480)
    window.show()

    other_widget = QWidget(window)
    other_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    other_widget.setGeometry(0, 0, 50, 50)
    other_widget.show()

    tour = GuidedTour(
        [TourStep(target=lambda: other_widget, title='Only', body='')],
        window,
    )
    qtbot.addWidget(tour._tooltip)
    qtbot.addWidget(tour._overlay)
    tour.start()
    qtbot.waitUntil(lambda: tour._steps[tour._current].title == 'Only')

    # Escape should close the tour even when some other widget -- not the
    # tooltip -- has keyboard focus.
    other_widget.setFocus()
    qtbot.waitUntil(lambda: QApplication.focusWidget() is other_widget)
    qtbot.keyPress(other_widget, Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: not tour._active)


def _advance_tour_until(tour, title, qtbot):
    """Call _on_next() until the given step title is current, waiting for
    _current to actually update between calls -- revealing a hidden dock
    target defers that step's positioning by one event-loop tick."""
    while tour._steps[tour._current].title != title:
        before = tour._current
        tour._on_next()
        qtbot.waitUntil(lambda before=before: tour._current != before)


def _reveal(widget, shown):
    """Mirror build_viewer_tour's reveal() helper: show widget if hidden,
    tracking it so callers can verify/restore, without touching a real
    QtViewer/dock widget (which trips an unrelated leak-detection issue
    in napari's test fixtures when hidden then closed, even on CI)."""

    def _ensure_visible():
        if widget.isVisible():
            return False
        shown.append(widget)
        widget.show()
        return True

    return _ensure_visible


def test_tour_reveals_hidden_target_for_its_step(qtbot):
    """A hidden step target used to freeze Next/Back once the tour reached
    that step."""
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 480)
    window.show()

    layer_list = QWidget(window)
    layer_list.setGeometry(0, 0, 50, 50)
    layer_controls = QWidget(window)
    layer_controls.setGeometry(60, 0, 50, 50)
    shown = []

    tour = GuidedTour(
        [
            TourStep(target=lambda: window, title='Welcome', body=''),
            TourStep(
                target=lambda: layer_list,
                title='Layer list',
                body='',
                ensure_visible=_reveal(layer_list, shown),
            ),
            TourStep(
                target=lambda: layer_controls,
                title='Layer controls',
                body='',
                ensure_visible=_reveal(layer_controls, shown),
            ),
        ],
        window,
    )

    def _restore() -> None:
        for w in shown:
            w.hide()

    tour.finished.connect(_restore)
    qtbot.addWidget(tour._tooltip)
    qtbot.addWidget(tour._overlay)
    # not revealed yet -- only happens once the relevant step is shown
    assert not layer_list.isVisible()
    assert not layer_controls.isVisible()

    tour.start()
    qtbot.waitUntil(lambda: tour._steps[tour._current].title == 'Welcome')

    _advance_tour_until(tour, 'Layer list', qtbot)
    assert layer_list.isVisible()
    assert not layer_controls.isVisible()

    _advance_tour_until(tour, 'Layer controls', qtbot)
    assert layer_controls.isVisible()

    tour.close_tour()
    assert not layer_list.isVisible()
    assert not layer_controls.isVisible()


def test_tour_reveals_target_hidden_mid_tour(qtbot):
    """Hiding a step's target partway through the tour, after its step
    already passed, used to still freeze the tour once that step came up
    again -- the target was only ever checked once, up front."""
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 480)
    window.show()

    layer_list = QWidget(window)
    layer_list.setGeometry(0, 0, 50, 50)
    layer_list.show()
    shown = []

    tour = GuidedTour(
        [
            TourStep(target=lambda: window, title='Welcome', body=''),
            TourStep(
                target=lambda: layer_list,
                title='Layer list',
                body='',
                ensure_visible=_reveal(layer_list, shown),
            ),
        ],
        window,
    )

    def _restore() -> None:
        for w in shown:
            w.hide()

    tour.finished.connect(_restore)
    qtbot.addWidget(tour._tooltip)
    qtbot.addWidget(tour._overlay)
    tour.start()
    qtbot.waitUntil(lambda: tour._steps[tour._current].title == 'Welcome')

    layer_list.hide()
    assert not layer_list.isVisible()

    _advance_tour_until(tour, 'Layer list', qtbot)
    assert layer_list.isVisible()

    tour.close_tour()
    assert not layer_list.isVisible()


def test_build_viewer_tour_leaves_visible_docks_alone(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_viewer = viewer.window._qt_viewer
    assert qt_viewer.dockLayerList.isVisible()
    assert qt_viewer.dockLayerControls.isVisible()

    tour = build_viewer_tour(viewer.window._qt_window, sample=None)
    tour.finished.emit()
    assert qt_viewer.dockLayerList.isVisible()
    assert qt_viewer.dockLayerControls.isVisible()


@pytest.mark.parametrize(
    ('shape', 'expect_dims_step'),
    [((4, 4), False), ((4, 4, 4), True)],
)
def test_tour_skips_dims_step_without_extra_dims(
    make_napari_viewer, shape, expect_dims_step
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros(shape))
    tour = build_viewer_tour(viewer.window._qt_window)

    titles = []
    index = tour._seek(0, 1)
    while index is not None:
        titles.append(tour._steps[index].title)
        index = tour._seek(index + 1, 1)

    assert ('Dimension sliders' in titles) is expect_dims_step


def test_resolve_tour_target_builtins(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window

    for name in _BUILTIN_TOUR_TARGETS:
        assert resolve_tour_target(qt_window, name) is not None

    assert resolve_tour_target(qt_window, 'nonexistent') is None


def test_resolve_tour_target_plugin_dock_widget(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window

    label = QLabel('hello')
    viewer.window.add_dock_widget(label, name='My Plugin Widget')

    assert resolve_tour_target(qt_window, 'My Plugin Widget') is label


def test_tour_skips_missing_target(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 480)
    window.show()

    first = QWidget(window)
    first.setGeometry(0, 0, 50, 50)
    first.show()
    last = QWidget(window)
    last.setGeometry(60, 0, 50, 50)
    last.show()

    tour = GuidedTour(
        [
            TourStep(target=lambda: first, title='First', body=''),
            TourStep(target=lambda: None, title='Missing', body=''),
            TourStep(target=lambda: last, title='Last', body=''),
        ],
        window,
    )
    qtbot.addWidget(tour._tooltip)
    qtbot.addWidget(tour._overlay)
    tour.start()
    qtbot.waitUntil(lambda: tour._steps[tour._current].title == 'First')

    tour._on_next()
    assert tour._steps[tour._current].title == 'Last'
    tour.close_tour()


def test_tour_current_step_unchanged_when_target_becomes_hidden(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 480)
    window.show()

    first = QWidget(window)
    first.setGeometry(0, 0, 50, 50)
    first.show()
    second = QWidget(window)
    second.setGeometry(60, 0, 50, 50)
    second.show()

    tour = GuidedTour(
        [
            TourStep(target=lambda: first, title='First', body=''),
            TourStep(target=lambda: second, title='Second', body=''),
        ],
        window,
    )
    qtbot.addWidget(tour._tooltip)
    qtbot.addWidget(tour._overlay)
    tour.start()
    qtbot.waitUntil(lambda: tour._steps[tour._current].title == 'First')

    second.hide()
    tour._on_next()
    assert tour._steps[tour._current].title == 'First'
    tour.close_tour()


def test_start_viewer_tour():
    tour = mock.Mock()
    qt_window = SimpleNamespace(_viewer_tour=None)
    window = SimpleNamespace(_qt_window=qt_window)

    with mock.patch(
        'napari._qt._qapp_model.qactions._help.build_viewer_tour',
        return_value=tour,
    ) as mock_build_tour:
        _start_viewer_tour(window)
        _start_viewer_tour(window)

    mock_build_tour.assert_called_once_with(qt_window)
    tour.finished.connect.assert_called_once()
    tour.start.assert_called_once()
    assert qt_window._viewer_tour is tour

    tour.finished.connect.call_args.args[0]()
    assert qt_window._viewer_tour is None


def test_start_viewer_tour_accepts_custom_tour():
    custom_tour = mock.Mock()
    qt_window = SimpleNamespace(_viewer_tour=None)
    window = SimpleNamespace(_qt_window=qt_window)

    with mock.patch(
        'napari._qt._qapp_model.qactions._help.build_viewer_tour',
    ) as mock_build_tour:
        _start_viewer_tour(window, tour=custom_tour)

    mock_build_tour.assert_not_called()
    custom_tour.finished.connect.assert_called_once()
    custom_tour.start.assert_called_once()
    assert qt_window._viewer_tour is custom_tour


@pytest.mark.parametrize(
    ('has_layers', 'pass_sample', 'expect_loaded'),
    [
        (False, True, True),  # no layers, sample given -> loads it
        (True, True, False),  # layers already present -> skips loading
        (False, False, False),  # sample=None -> never loads
    ],
)
def test_build_viewer_tour_sample(
    make_napari_viewer, tmp_plugin, has_layers, pass_sample, expect_loaded
):
    @tmp_plugin.contribute.sample_data(key='fake')
    def _generate_fake_data():
        return [(np.zeros((4, 4, 4)), {'name': 'fake'})]

    viewer = make_napari_viewer()
    if has_layers:
        viewer.add_image(np.zeros((4, 4)))
    n_layers_before = len(viewer.layers)

    sample = (tmp_plugin.name, 'fake') if pass_sample else None
    build_viewer_tour(viewer.window._qt_window, sample=sample)

    n_new_layers = len(viewer.layers) - n_layers_before
    assert (n_new_layers > 0) is expect_loaded
