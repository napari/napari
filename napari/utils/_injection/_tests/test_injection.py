import sys
from concurrent.futures import Future
from contextlib import nullcontext
from typing import ContextManager
from unittest.mock import Mock

import numpy as np
import pytest

from napari.components import LayerList
from napari.layers import Image, Labels, Layer, Points
from napari.types import ImageData, LayerDataTuple
from napari.utils._injection import (
    inject_napari_dependencies,
    set_processors,
    set_providers,
)
from napari.viewer import Viewer, ViewerModel


def test_napari_injection():
    viewer = ViewerModel()
    points = viewer.add_points()

    @inject_napari_dependencies
    def f(v: Viewer, p: Points, ll: LayerList):
        return (v, p, ll)

    # NOTE: this is the current behavior. But we should probably go farther
    # and check whether the annotation allows Optional[T]... otherwise
    # we shouldn't pass None.
    assert f() == (None, None, None)
    lookup = {
        Viewer: lambda: viewer,
        Points: lambda: points,
        LayerList: lambda: viewer.layers,
    }

    with set_providers(lookup, clobber=True):
        assert f() == (viewer, points, viewer.layers)

    assert f() == (None, None, None)


def test_napari_injection_missing():
    @inject_napari_dependencies
    def f(x: int):
        return x

    with pytest.raises(TypeError) as e:
        f()
    assert 'missing 1 required positional argument' in str(e.value)
    assert f(4) == 4
    with set_providers({int: lambda: 1}):
        assert f() == 1


def test_processors():
    @inject_napari_dependencies
    def f1() -> LayerDataTuple:
        return (None, {'name': 'my points'}, 'points')

    @inject_napari_dependencies
    def f2() -> ImageData:
        return np.random.rand(4, 4)

    @inject_napari_dependencies
    def f3() -> Labels:
        return Labels(np.ones((4, 4), int))

    # we should still be able to call these... they don't need inputs.
    f1()
    f2()
    f3()

    v = ViewerModel()
    assert not v.layers

    with set_providers({Viewer: lambda: v}, clobber=True):
        f1()
        assert len(v.layers) == 1 and v.layers[0].name == 'my points'
        f2()
        assert len(v.layers) == 2 and isinstance(v.layers[-1], Image)
        f3()
        assert len(v.layers) == 3 and isinstance(v.layers[-1], Labels)

        # f1 gave a layer name, so it should update, not add a layer
        v.layers[0].data = np.arange(4).reshape(2, 2)
        f1()
        assert len(v.layers) == 3 and not any(v.layers[0].data)

    # trying to set an existing accessor without clobber is an error
    with pytest.raises(ValueError) as e:
        set_providers({Viewer: lambda: None})
    assert 'already has a provider and clobber is False' in str(e.value)


def test_set_processor():
    @inject_napari_dependencies
    def f2(x: int) -> int:
        return x

    with pytest.raises(ValueError) as e:
        set_processors({ImageData: lambda: None})
    assert 'already has a processor and clobber is False' in str(e.value)

    mock = Mock()
    with set_processors({int: mock}):
        assert f2(3) == 3
    mock.assert_called_once_with(3)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason='Futures not subscriptable before py3.9'
)
def test_future_processor():
    @inject_napari_dependencies
    def add_data() -> Future[ImageData]:
        future = Future()
        future.set_result(np.zeros((4, 4)))
        return future

    v = ViewerModel()

    future = add_data()
    assert future.result().shape == (4, 4)
    # with no active viewer accessor, calling the future just returns the data
    # but doesn't add them to anything
    assert not v.layers

    # setting the accessor to our local viewer
    with set_providers({Viewer: lambda: v}, clobber=True):
        future = add_data()
        assert future.result().shape == (4, 4)

    # now it's added
    assert len(v.layers) == 1 and isinstance(v.layers[0], Image)


def test_injection_with_generator():
    @inject_napari_dependencies
    def f(lay: Layer, v: Viewer):
        yield lay
        yield v

    v = ViewerModel()
    v.add_points()

    # setting the accessor to our local viewer
    with set_providers({Viewer: lambda: v}, clobber=True):
        assert tuple(f()) == (v.layers[0], v)


def test_injection_without_args():
    """it just returns the same function"""

    def f():
        ...

    assert inject_napari_dependencies(f) is f


modes = ['raise', 'warn', 'return', 'ignore']


def unknown(v: 'Unknown') -> int:  # type: ignore  # noqa
    ...


def unannotated(x) -> int:  # type: ignore  # noqa
    ...


def unknown_and_unannotated(v: 'Unknown', x) -> int:  # type: ignore  # noqa
    ...


@pytest.mark.parametrize('on_unresolved', modes)
@pytest.mark.parametrize('on_unannotated', modes)
@pytest.mark.parametrize(
    'in_func', [unknown, unannotated, unknown_and_unannotated]
)
def test_injection_errors(in_func, on_unresolved, on_unannotated):

    ctx: ContextManager = nullcontext()
    expect_same_func_back = False

    if (
        'unknown' in in_func.__name__
    ):  # required params with unknown annotations
        if on_unresolved == 'raise':
            ctx = pytest.raises(
                NameError,
                match="Could not resolve type hint for required parameter 'v'",
            )
        else:
            expect_same_func_back = True
            if on_unresolved == 'warn':
                ctx = pytest.warns(
                    UserWarning,
                    match="Could not resolve type hint for required parameter 'v'",
                )

    elif (
        'unannotated' in in_func.__name__
    ):  # required params without annotations
        if on_unannotated == 'raise':
            ctx = pytest.raises(
                TypeError,
                match="Injecting dependencies on function 'unannotated' with a required, unannotated parameter 'x'",
            )
        elif on_unannotated == 'warn':
            ctx = pytest.warns(
                UserWarning,
                match="Injecting dependencies on function 'unannotated' with a required, unannotated parameter 'x'",
            )
        elif on_unannotated == 'return':
            expect_same_func_back = True

    with ctx:
        out_func = inject_napari_dependencies(
            in_func,
            on_unannotated_required_args=on_unannotated,
            on_unresolved_required_args=on_unresolved,
        )

        if expect_same_func_back:
            assert out_func is in_func
        else:
            assert out_func is not in_func
