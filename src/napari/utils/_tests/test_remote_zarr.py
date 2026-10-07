import functools
import http.server
import threading

import numpy as np
import pytest
import zarr

from napari.utils import open_cached_zarr


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, failing_paths, **kwargs):
        self.failing_paths = failing_paths
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path in self.failing_paths:
            self.send_error(500)
            return
        super().do_GET()


@pytest.fixture
def failing_paths():
    return set()


@pytest.fixture
def server(tmp_path, failing_paths):
    served = tmp_path / 'served'
    array = zarr.create_array(
        served / 'array.zarr', shape=(4, 4), chunks=(2, 2), dtype='uint16'
    )
    array[:] = np.arange(16, dtype='uint16').reshape(4, 4)
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0),
        functools.partial(
            _Handler, failing_paths=failing_paths, directory=served
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def _url(server):
    return f'http://127.0.0.1:{server.server_port}/array.zarr'


def test_open_cached_zarr_reads_from_cache_without_server(server, tmp_path):
    """Read the same data from the cache after the server stops."""
    expected = np.arange(16, dtype='uint16').reshape(4, 4)
    np.testing.assert_array_equal(
        open_cached_zarr(_url(server), tmp_path / 'cache')[:], expected
    )
    url = _url(server)
    server.shutdown()
    server.server_close()

    np.testing.assert_array_equal(
        open_cached_zarr(url, tmp_path / 'cache')[:], expected
    )


def test_open_cached_zarr_raises_on_server_error(
    server, failing_paths, tmp_path
):
    """Raise on a chunk the server fails to return instead of filling it."""
    failing_paths.add('/array.zarr/c/0/0')
    array = open_cached_zarr(_url(server), tmp_path / 'cache')

    with pytest.raises(ConnectionError, match="'c/0/0'"):
        array[:]
