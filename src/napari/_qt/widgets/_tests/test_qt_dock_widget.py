import pytest
from qtpy.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari._qt.utils import combine_widgets


def test_add_dock_widget(make_napari_viewer):
    """Test basic add_dock_widget functionality"""
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    dwidg = viewer.window.add_dock_widget(widg, name='test', area='bottom')
    assert not dwidg.is_vertical
    assert viewer.window._qt_window.findChild(QDockWidget, 'test')
    assert dwidg.widget() == widg
    assert dwidg.titleBarWidget() is dwidg.title
    assert dwidg.title.vertical
    assert viewer.window.dock_widgets['test'] is widg
    with pytest.raises(KeyError):
        assert viewer.window.dock_widgets['test2']

    widg2 = QPushButton('button')
    dwidg2 = viewer.window.add_dock_widget(widg2, name='test2', area='right')
    assert dwidg2.is_vertical
    assert viewer.window._qt_window.findChild(QDockWidget, 'test2')
    assert dwidg2.widget() == widg2
    assert dwidg2.titleBarWidget() is dwidg2.title
    assert not dwidg2.title.vertical

    with pytest.raises(ValueError, match='area argument must be'):
        viewer.window.add_dock_widget(widg2, name='test2', area='under')

    with pytest.raises(ValueError, match='all allowed_areas argument must be'):
        viewer.window.add_dock_widget(
            widg2, name='test2', allowed_areas=['under']
        )

    with pytest.raises(TypeError, match='`allowed_areas` must be a list'):
        viewer.window.add_dock_widget(
            widg2, name='test2', allowed_areas='under'
        )


def test_add_dock_widget_from_list(make_napari_viewer):
    """Test that we can add a list of widgets and they will be combined"""
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    widg2 = QPushButton('button')

    dwidg = viewer.window.add_dock_widget(
        [widg, widg2], name='test', area='right'
    )
    assert viewer.window._qt_window.findChild(QDockWidget, 'test')
    assert isinstance(dwidg.widget().layout(), QVBoxLayout)

    dwidg = viewer.window.add_dock_widget(
        [widg, widg2], name='test2', area='bottom'
    )
    assert viewer.window._qt_window.findChild(QDockWidget, 'test2')
    assert isinstance(dwidg.widget().layout(), QHBoxLayout)


def test_add_dock_widget_raises(make_napari_viewer):
    """Test that the widget passed must be a DockWidget."""
    viewer = make_napari_viewer()
    widg = object()

    with pytest.raises(TypeError):
        viewer.window.add_dock_widget(widg, name='test')


def test_remove_dock_widget_orphans_widget(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    qtbot.addWidget(widg)

    assert not widg.parent()
    dw = viewer.window.add_dock_widget(
        widg, name='test', menu=viewer.window.window_menu
    )
    assert widg.parent() is dw
    assert dw.toggleViewAction() in viewer.window.window_menu.actions()
    viewer.window.remove_dock_widget(dw, menu=viewer.window.window_menu)
    assert dw.toggleViewAction() not in viewer.window.window_menu.actions()
    del dw
    # if dw didn't release widg, we'd get an exception when next accessing widg
    assert not widg.parent()


def test_remove_dock_widget_by_widget_reference(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    qtbot.addWidget(widg)

    dw = viewer.window.add_dock_widget(widg, name='test')
    assert widg.parent() is dw
    assert dw in viewer.window._qt_window.findChildren(QDockWidget)
    viewer.window.remove_dock_widget(widg)
    with pytest.raises(LookupError):
        # it's gone this time:
        viewer.window.remove_dock_widget(widg)
    assert not widg.parent()


def test_adding_modified_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widg = QWidget()
    # not uncommon to see people shadow the builtin layout()
    # which breaks our ability to add vertical stretch... but shouldn't crash
    widg.layout = None
    dw = viewer.window.add_dock_widget(widg, name='test', area='right')
    assert dw.widget() is widg


def test_adding_stretch(make_napari_viewer):
    """Make sure that vertical stretch only gets added when appropriate."""
    viewer = make_napari_viewer()

    # adding a widget to the left/right will usually addStretch to the layout
    widg = QWidget()
    widg.setLayout(QVBoxLayout())
    widg.layout().addWidget(QPushButton())
    assert widg.layout().count() == 1
    dw = viewer.window.add_dock_widget(widg, area='right')
    assert widg.layout().count() == 2
    dw.close()

    # ... unless the widget has a widget with a large vertical sizePolicy
    widg = QWidget()
    widg.setLayout(QVBoxLayout())
    widg.layout().addWidget(QTextEdit())
    assert widg.layout().count() == 1
    dw = viewer.window.add_dock_widget(widg, area='right')
    assert widg.layout().count() == 1
    dw.close()

    # ... widgets on the bottom do not get stretch
    widg = QWidget()
    widg.setLayout(QHBoxLayout())
    widg.layout().addWidget(QPushButton())
    assert widg.layout().count() == 1
    dw = viewer.window.add_dock_widget(widg, area='bottom')
    assert widg.layout().count() == 1
    dw.close()


def test_float_unfloat_title_bar(make_napari_viewer):
    """Float then unfloat must preserve the custom title bar with correct orientation.

    Regression test for https://github.com/napari/napari/issues/8887.

    The custom title bar must always be set (docked and floating) so that
    napari's dark theming and custom controls are always visible.

    Orientation rules:
      - docked left/right (is_vertical=True)  → vertical=False (horizontal bar)
      - docked top/bottom (is_vertical=False) → vertical=True  (side bar)
      - floating                               → always vertical=False
    """
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    dw = viewer.window.add_dock_widget(widg, name='test', area='right')

    assert dw.titleBarWidget() is dw.title
    assert not dw.title.vertical  # right-docked -> horizontal bar

    dw.setFloating(True)
    assert dw.isFloating()
    assert dw.titleBarWidget() is dw.title
    # Floating always uses a horizontal title bar regardless of window size.
    assert not dw.title.vertical

    # Re-dock the widget.
    dw.setFloating(False)
    assert not dw.isFloating()
    # Custom title bar restored with correct orientation.
    assert dw.titleBarWidget() is dw.title


def test_float_from_bottom_clears_vertical_titlebar_feature(
    make_napari_viewer,
):
    """Floating a widget from top/bottom must not leave the title bar as a left-side strip.

    Regression test for https://github.com/napari/napari/issues/8887.

    When docked at top/bottom Qt sets DockWidgetVerticalTitleBar, placing the
    custom title bar on the left side of the widget.  If that feature is not
    cleared when the widget becomes floating, the title bar ends up as a
    narrow unusable strip on the left edge of the floating window.
    """
    from qtpy.QtWidgets import QDockWidget

    viewer = make_napari_viewer()
    widg = QPushButton('button')
    dw = viewer.window.add_dock_widget(widg, name='test', area='bottom')

    # Docked at bottom: vertical feature should be set (sidebar-style bar).
    assert (
        dw.features()
        & QDockWidget.DockWidgetFeature.DockWidgetVerticalTitleBar
    )

    # Float the widget.
    dw.setFloating(True)
    assert dw.isFloating()
    # DockWidgetVerticalTitleBar must be cleared so the title bar spans the
    # top of the floating window, not the left edge.
    assert not (
        dw.features()
        & QDockWidget.DockWidgetFeature.DockWidgetVerticalTitleBar
    )
    assert dw.titleBarWidget() is dw.title
    assert not dw.title.vertical  # horizontal bar on the floating window


def test_combine_widgets_error():
    """Check error raised when combining widgets with invalid types."""
    with pytest.raises(
        TypeError, match='"widgets" must be a QWidget, a magicgui'
    ):
        combine_widgets(['string'])
