from unittest.mock import patch

import numpy as np
import pytest

from napari.components.viewer_model import ViewerModel
from napari.utils._proxies import PublicOnlyProxy, ReadOnlyWrapper


def test_ReadOnlyWrapper_setitem():
    """test that ReadOnlyWrapper prevents setting items"""
    d = {'hi': 3}
    d_read_only = ReadOnlyWrapper(d)

    with pytest.raises(TypeError):
        d_read_only['hi'] = 5


def test_ReadOnlyWrapper_setattr():
    """test that ReadOnlyWrapper prevents setting attributes"""

    class TestClass:
        x = 3

    tc = TestClass()
    tc_read_only = ReadOnlyWrapper(tc)

    with pytest.raises(TypeError):
        tc_read_only.x = 5


@pytest.fixture
def patched_root_dir():
    """Simulate a call from outside of napari"""
    with patch('napari.utils.misc.ROOT_DIR', new='/some/other/package'):
        yield


def test_PublicOnlyProxy(patched_root_dir):
    class X:
        a = 1
        _b = 'nope'

        def method(self):
            return 2

    class Tester:
        x = X()
        _private = 2

        def __getitem__(self, key):
            return X()

    t = Tester()
    proxy = PublicOnlyProxy(t)
    assert proxy.x.a == 1
    assert proxy[0].a == 1
    assert proxy.x.method() == 2

    assert isinstance(proxy, Tester)
    with pytest.warns(FutureWarning) as e:
        proxy._private
    assert 'Private attribute access' in str(e[0].message)

    with pytest.warns(FutureWarning) as e:
        # works on sub-objects too
        proxy.x._b
    assert 'Private attribute access' in str(e[0].message)

    with pytest.warns(FutureWarning) as e:
        # works on sub-items too
        proxy[0]._b
    assert 'Private attribute access' in str(e[0].message)

    assert '_private' not in dir(proxy)
    assert '_private' in dir(t)


def test_public_proxy_limited_to_napari(patched_root_dir):
    """Test that the recursive public proxy goes no farther than napari."""
    viewer = ViewerModel()
    viewer.add_points(None)
    pv = PublicOnlyProxy(viewer)
    assert not isinstance(pv.layers[0].data, PublicOnlyProxy)


def test_array_from_proxy_objects(patched_root_dir):
    """Test that the recursive public proxy goes no farther than napari."""
    viewer = ViewerModel()
    viewer.add_points(None)
    pv = PublicOnlyProxy(viewer)
    assert isinstance(np.array(pv.dims.displayed, dtype=int), np.ndarray)


def test_receive_return_proxy_object():
    """Test that an"""
    viewer = ViewerModel()
    viewer.add_image(np.random.random((20, 20)))
    pv = PublicOnlyProxy(viewer)

    # simulates behavior from outside of napari
    with patch('napari.utils.misc.ROOT_DIR', new='/some/other/package'):
        # the recursion means this layer will be a Proxy Object on __getitem__
        layer = pv.layers[-1]
        assert isinstance(layer, PublicOnlyProxy)
        # remove and add it back, should be fine
        add_layer = getattr(pv, 'add_layer')

    add_layer(layer)
    assert len(viewer.layers) == 2


def test_viewer_method():
    viewer = PublicOnlyProxy(ViewerModel())
    assert viewer.add_points() is not None
