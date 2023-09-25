from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from napari.components.viewer_model import ViewerModel
from napari.utils._proxies import PublicOnlyProxy, ReadOnlyWrapper
from napari.utils.events.containers._set import EventedSet


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
    with pytest.warns(FutureWarning, match='Private attribute access'):
        proxy._private

    with pytest.warns(FutureWarning, match='Private attribute access'):
        # warns on setattr
        proxy._private = 4

    with pytest.warns(FutureWarning, match='Private attribute access'):
        # works on sub-objects too
        proxy.x._b

    with pytest.warns(FutureWarning, match='Private attribute access'):
        # works on sub-items too
        proxy[0]._b

    assert '_private' not in dir(proxy)
    assert '_private' in dir(t)


@pytest.mark.filterwarnings("ignore:Qt libs are available but")
def test_thread_proxy_guard(monkeypatch, single_threaded_executor):
    class X:
        a = 1

    monkeypatch.setenv('NAPARI_ENSURE_PLUGIN_MAIN_THREAD', 'True')

    x = X()
    x_proxy = PublicOnlyProxy(x)

    f = single_threaded_executor.submit(x.__setattr__, 'a', 2)
    f.result()
    assert x.a == 2

    f = single_threaded_executor.submit(x_proxy.__setattr__, 'a', 3)
    with pytest.raises(RuntimeError):
        f.result()
    assert x.a == 2


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
        add_layer = pv.add_layer
        viewer.layers.pop()

    add_layer(layer)
    assert len(viewer.layers) == 1


def test_viewer_method():
    viewer = PublicOnlyProxy(ViewerModel())
    assert viewer.add_points() is not None


def test_unwrap_on_call():
    """Check that PublicOnlyProxy'd arguments to methods of a
    PublicOnlyProxy'd object are unwrapped before calling the method.
    """
    evset = EventedSet()
    public_only_evset = PublicOnlyProxy(evset)
    text = "aaa"
    wrapped_text = PublicOnlyProxy(text)
    public_only_evset.add(wrapped_text)
    retrieved_text = next(iter(evset))

    # check that the text in the set is not the version wrapped with
    # PublicOnlyProxy
    assert id(text) == id(retrieved_text)


def test_unwrap_setattr():
    """Check that objects added with __setattr__ of an object wrapped with
    PublicOnlyProxy are unwrapped before setting the attribute.
    """

    @dataclass
    class Sample:
        attribute = "aaa"

    sample = Sample()
    public_only_sample = PublicOnlyProxy(sample)

    text = "bbb"
    wrapped_text = PublicOnlyProxy(text)

    public_only_sample.attribute = wrapped_text
    attribute = sample.attribute  # use original, not wrapped object

    # check that the attribute in the unwrapped sample is itself not the
    # wrapped text, but the original text.
    assert id(text) == id(attribute)
