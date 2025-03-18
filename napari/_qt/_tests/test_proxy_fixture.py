import pytest


def test_proxy_fixture_thread_error(
    make_napari_viewer_proxy, single_threaded_executor
):
    viewer = make_napari_viewer_proxy()
    future = single_threaded_executor.submit(
        viewer.__setattr__, 'status', 'hi'
    )
    with pytest.raises(RuntimeError, match='Setting attributes'):
        future.result()
