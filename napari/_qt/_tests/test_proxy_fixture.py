import pytest

from napari.utils import misc


def test_proxy_fixture_warning(make_napari_viewer_proxy, monkeypatch):
    viewer = make_napari_viewer_proxy()

    monkeypatch.setattr(misc, 'ROOT_DIR', '/some/other/package')
    with pytest.warns(FutureWarning, match='Private attribute access'):
        viewer.window._qt_window


def test_proxy_fixture_thread_error(
    make_napari_viewer_proxy, single_threaded_executor
):
    viewer = make_napari_viewer_proxy()
    future = single_threaded_executor.submit(
        viewer.__setattr__, 'status', 'hi'
    )
    with pytest.raises(RuntimeError, match='Setting attributes'):
        future.result()
