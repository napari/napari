import os
import platform

import numpy as np
import pytest

from napari import Viewer, layers
from napari._tests.utils import (
    add_layer_by_type,
    check_view_transform_consistency,
    check_viewer_functioning,
    layer_test_data,
)
from napari.utils._tests.test_naming import eval_with_filename


def test_viewer(make_test_viewer):
    """Test instantiating viewer."""
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.vbox_layout.count() == 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Switch to 3D rendering mode and back to 2D rendering mode
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2

    # Run all class key bindings
    for func in viewer.class_keymap.values():
        # skip fullscreen test locally
        if func.__name__ == 'toggle_fullscreen' and not os.getenv("CI"):
            continue
        if func.__name__ == 'play':
            continue
        func(viewer)


@pytest.mark.run(order=1)  # provided by pytest-ordering
def test_no_qt_loop():
    """Test informative error raised when no Qt event loop exists.

    Logically, this test should go at the top of the file. Howveer, that
    resulted in tests passing when only this file was run, but failing when
    other tests involving Qt-bot were run before this file. Putting this test
    second provides a sanity check that pytest-ordering is correctly doing its
    magic.
    """
    with pytest.raises(RuntimeError):
        _ = Viewer()


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
@pytest.mark.parametrize('visible', [True, False])
def test_add_layer(make_test_viewer, layer_class, data, ndim, visible):
    viewer = make_test_viewer()
    layer = add_layer_by_type(viewer, layer_class, data, visible=visible)
    check_viewer_functioning(viewer, viewer.window.qt_viewer, data, ndim)

    # Run all class key bindings
    for func in layer.class_keymap.values():
        func(layer)


@pytest.mark.parametrize('layer_class, a_unique_name, ndim', layer_test_data)
def test_add_layer_magic_name(
    make_test_viewer, layer_class, a_unique_name, ndim
):
    """Test magic_name works when using add_* for layers"""
    # Tests for issue #1709
    viewer = make_test_viewer()  # noqa: F841
    layer = eval_with_filename(
        "add_layer_by_type(viewer, layer_class, a_unique_name)", "somefile.py",
    )
    assert layer.name == "a_unique_name"


def test_screenshot(make_test_viewer):
    """Test taking a screenshot."""
    viewer = make_test_viewer()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot of the image canvas only
    screenshot = viewer.screenshot(canvas_only=True)
    assert screenshot.ndim == 3

    # Take screenshot with the viewer included
    screenshot = viewer.screenshot(canvas_only=False)
    assert screenshot.ndim == 3


def test_changing_theme(make_test_viewer):
    """Test changing the theme updates the full window."""
    viewer = make_test_viewer()
    viewer.add_points(data=None)
    assert viewer.palette['folder'] == 'dark'

    screenshot_dark = viewer.screenshot(canvas_only=False)

    viewer.theme = 'light'
    assert viewer.palette['folder'] == 'light'

    screenshot_light = viewer.screenshot(canvas_only=False)
    equal = (screenshot_dark == screenshot_light).min(-1)

    # more than 99.5% of the pixels have changed
    assert (np.count_nonzero(equal) / equal.size) < 0.05, "Themes too similar"

    with pytest.raises(ValueError):
        viewer.theme = 'nonexistent_theme'


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_roll_traspose_update(make_test_viewer, layer_class, data, ndim):
    """Check that transpose and roll preserve correct transform sequence."""

    viewer = make_test_viewer()

    np.random.seed(0)

    layer = add_layer_by_type(viewer, layer_class, data)

    # Set translations and scalings (match type of visual layer storing):
    transf_dict = {
        'translate': np.random.randint(0, 10, ndim).astype(np.float32),
        'scale': np.random.rand(ndim).astype(np.float32),
    }
    for k, val in transf_dict.items():
        setattr(layer, k, val)

    if layer_class in [layers.Image, layers.Labels]:
        transf_dict['translate'] -= transf_dict['scale'] / 2

    # Check consistency:
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Roll dims and check again:
    viewer.dims._roll()
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Transpose and check again:
    viewer.dims._transpose()
    check_view_transform_consistency(layer, viewer, transf_dict)


def test_toggling_axes(make_test_viewer):
    """Test toggling axes."""
    viewer = make_test_viewer()

    # Check axes are not visible
    assert not viewer.axes.visible

    # Make axes visible
    viewer.axes.visible = True
    assert viewer.axes.visible

    # Enter 3D rendering and check axes still visible
    viewer.dims.ndisplay = 3
    assert viewer.axes.visible

    # Make axes not visible
    viewer.axes.visible = False
    assert not viewer.axes.visible


def test_toggling_scale_bar(make_test_viewer):
    """Test toggling scale bar."""
    viewer = make_test_viewer()

    # Check scale bar is not visible
    assert not viewer.scale_bar.visible

    # Make scale bar visible
    viewer.scale_bar.visible = True
    assert viewer.scale_bar.visible

    # Enter 3D rendering and check scale bar is still visible
    viewer.dims.ndisplay = 3
    assert viewer.scale_bar.visible

    # Make scale bar not visible
    viewer.scale_bar.visible = False
    assert not viewer.scale_bar.visible


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows specific")
def test_windows_grouping_overwrite(make_test_viewer):
    import ctypes

    def get_app_id():
        mem = ctypes.POINTER(ctypes.c_wchar)()
        ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(
            ctypes.byref(mem)
        )
        res = ctypes.wstring_at(mem)
        ctypes.windll.Ole32.CoTaskMemFree(mem)
        return res

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("test_text")

    assert "test_text" == get_app_id()

    class OwnViewer(Viewer):
        _napari_app_id = "custom_string"

    make_test_viewer(viewer_class=OwnViewer)

    assert OwnViewer._napari_app_id == get_app_id()

    make_test_viewer()

    assert Viewer._napari_app_id == get_app_id()

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("test_text")

    class OwnViewer2(Viewer):
        _napari_app_id = ""

    make_test_viewer(viewer_class=OwnViewer2)

    assert "test_text" == get_app_id()


def test_icon_overwrite(make_test_viewer, monkeypatch):
    from napari._qt.qt_main_window import QApplication

    class OwnViewer(Viewer):
        _napari_global_logo = False

    called = [0]

    def set_icon_mock(self, _icon):
        called[0] = 1

    monkeypatch.setattr(QApplication, "setWindowIcon", set_icon_mock)

    make_test_viewer()

    assert called[0] == 1
    called = [0]

    make_test_viewer(viewer_class=OwnViewer)

    assert called[0] == 0
