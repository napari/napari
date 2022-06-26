import types
from typing import Any, Callable, Optional, Sequence

import pytest

from napari.utils._injection import (
    resolve_single_type_hints,
    resolve_type_hints,
    type_resolved_signature,
)


def basic_sig(a: 'int', b: 'str', c: 'Optional[float]' = None) -> int:
    ...


def uses_napari(image: 'Image', layers: 'LayerList') -> 'Labels':  # type: ignore # noqa
    ...


def requires_unknown(param: 'Unknown', x) -> 'Unknown':  # type: ignore # noqa
    ...


def optional_unknown(param: 'Unknown' = 1) -> 'Unknown':  # type: ignore # noqa
    ...


def test_resolve_type_hints():
    with pytest.raises(NameError, match='not defined'):
        resolve_type_hints(uses_napari, inject_napari_namespace=False)
    with pytest.raises(NameError):
        resolve_type_hints(requires_unknown)

    hints = resolve_type_hints(uses_napari, inject_napari_namespace=True)
    from napari import layers

    assert hints['image'] == layers.Image

    hints = resolve_type_hints(basic_sig, inject_napari_namespace=False)
    assert hints['c'] == Optional[float]

    hints = resolve_type_hints(requires_unknown, localns={'Unknown': int})
    assert hints['param'] == int


def test_resolve_single_type_hints():
    hints = resolve_single_type_hints(
        int,
        'Optional[int]',
        'FunctionType',
        'Callable[..., Any]',
        'Optional[Sequence[Layer]]',
        'ViewerModel',
        'LayerList',
        'Viewer',
        inject_napari_namespace=True,
    )

    from napari import components, layers, viewer

    assert hints == (
        int,
        Optional[int],
        types.FunctionType,
        Callable[..., Any],
        Optional[Sequence[layers.Layer]],
        components.viewer_model.ViewerModel,
        components.layerlist.LayerList,
        viewer.Viewer,
    )


def test_type_resolved_signature():
    from napari import layers

    sig = type_resolved_signature(basic_sig)
    assert sig.parameters['c'].annotation == Optional[float]

    sig = type_resolved_signature(uses_napari, inject_napari_namespace=True)
    assert sig.parameters['image'].annotation == layers.Image
    assert sig.return_annotation == layers.Labels

    with pytest.raises(
        NameError, match='use `raise_unresolved_optional_args=False`'
    ):
        type_resolved_signature(optional_unknown)

    sig = type_resolved_signature(
        optional_unknown, raise_unresolved_optional_args=False
    )
    assert sig.parameters['param'].annotation == 'Unknown'

    with pytest.raises(NameError, match='Could not resolve all annotations'):
        type_resolved_signature(requires_unknown)

    with pytest.raises(
        NameError,
        match="Could not resolve type hint for required parameter 'param'",
    ):
        type_resolved_signature(
            requires_unknown, raise_unresolved_optional_args=False
        )

    sig = type_resolved_signature(requires_unknown, localns={'Unknown': int})
    assert sig.parameters['param'].annotation == int
