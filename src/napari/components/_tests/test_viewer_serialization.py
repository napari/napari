import json
from typing import Any

import numpy as np
import pytest

from napari.components import ViewerModel
from napari.utils.color import ColorArray, ColorValue
from napari.utils.events import EventedModel
from napari.utils.events.evented_model import (
    _serialize_arbitrary,
)


def test_viewer_model_dump_respects_custom_exclude():
    """User-provided exclude must be honored alongside defaults."""
    viewer = ViewerModel()
    dumped = viewer.model_dump(exclude={'title'})
    assert 'title' not in dumped
    assert 'theme' in dumped

    dumped_json = viewer.model_dump(mode='json', exclude={'title'})
    assert 'title' not in dumped_json

    raw = viewer.model_dump_json(exclude={'title'})
    assert 'title' not in json.loads(raw)


def test_viewer_model_dump_json_with_pint_units():
    """Dims.units (pint.Unit) must become JSON-compatible strings."""
    import pint

    viewer = ViewerModel()
    ureg = pint.UnitRegistry()
    viewer.dims.units = (ureg.meter, ureg.second)

    # python mode keeps Unit objects
    dumped = viewer.model_dump()
    assert dumped['dims']['units'][0] == ureg.meter

    # json mode must convert to strings, not raise
    raw = viewer.model_dump_json()
    data = json.loads(raw)
    assert data['dims']['units'] == ['meter', 'second']

    json_mode = viewer.model_dump(mode='json')
    assert json_mode['dims']['units'] == ['meter', 'second']


def test_viewer_model_dump_json_overlays_as_dicts():
    """EventedDictNamespace overlays must serialize as plain dicts."""
    viewer = ViewerModel()
    data = viewer.model_dump(mode='json')

    assert isinstance(data['canvas']['overlays'], dict)
    assert isinstance(data['scene']['overlays'], dict)
    # overlays contain nested EventedModels, which must also be dicts
    for overlay in data['canvas']['overlays'].values():
        assert isinstance(overlay, dict)

    raw = viewer.model_dump_json()
    data_json = json.loads(raw)
    assert isinstance(data_json['canvas']['overlays'], dict)


def test_viewer_model_dump_json_nested_cursor_and_dims():
    viewer = ViewerModel()
    viewer.cursor.position = (1.0, 2.0)
    viewer.dims.axis_labels = ('y', 'x')

    data = json.loads(viewer.model_dump_json())
    assert tuple(data['cursor']['position']) == (1.0, 2.0)
    assert tuple(data['dims']['axis_labels']) == ('y', 'x')


def test_viewer_model_dump_json_with_layers_present():
    """JSON must still succeed when layers exist."""
    viewer = ViewerModel()
    viewer.add_image(np.random.random((10, 15)))
    viewer.add_labels(np.zeros((10, 15), dtype=int))

    raw = viewer.model_dump_json()
    data = json.loads(raw)
    assert 'layers' in data


class _AnyModel(EventedModel):
    # A type annotated as any would normally fail with anything that
    # pydantic doesn't know how to serialize, even if it declares its own
    # serialization machinery, because pydantic only checks the schema
    # of the annotations, not runtime objects.
    x: Any


def test_any_with_color_value():
    m = _AnyModel(x=ColorValue('red'))
    assert m.model_dump_json() == '{"x":[1.0,0.0,0.0,1.0]}'
    assert m.model_dump(mode='json') == {
        'x': [1.0, 0.0, 0.0, 1.0],
    }


def test_any_with_color_array():
    m = _AnyModel(x=ColorArray(['red', 'blue']))
    data = json.loads(m.model_dump_json())
    assert data == {'x': [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]}


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (np.int64(3), 3),
        (np.int32(3), 3),
        (np.float32(1.5), 1.5),
        (np.bool_(True), True),
        (np.array([1, 2, 3]), [1, 2, 3]),
    ],
)
def test_any_with_numpy_scalars(value, expected):
    """Numpy scalars have no pydantic schema; they must become plain python."""
    m = _AnyModel(x=value)
    assert json.loads(m.model_dump_json()) == {'x': expected}


def test_any_with_dask_array():
    da = pytest.importorskip('dask.array')
    m = _AnyModel(x=da.ones((2, 2)))
    assert json.loads(m.model_dump_json()) == {'x': [[1.0, 1.0], [1.0, 1.0]]}


def test_any_with_dataframe():
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    m = _AnyModel(x=df)
    assert json.loads(m.model_dump_json()) == {
        'x': {'a': [1, 2], 'b': ['x', 'y']}
    }


def test_any_with_nested_arbitrary():
    """Arbitrary containers must be unraveled recursively."""
    import pint

    ureg = pint.UnitRegistry()
    m = _AnyModel(
        x={'arr': np.array([1, 2]), 'unit': ureg.meter, 'n': np.int64(5)}
    )
    assert json.loads(m.model_dump_json()) == {
        'x': {'arr': [1, 2], 'unit': 'meter', 'n': 5}
    }


def test_serialize_layer_to_json_compatible_dict():
    """Layers serialize via ``as_layer_data_tuple``, ignoring data/metadata.

    Data can be huge/arbitrary (numpy/dask) and metadata is an arbitrary
    user dict, so both are skipped for now. Only ``meta`` (minus
    ``metadata``) and ``type`` are kept.
    """
    viewer = ViewerModel()
    viewer.add_image(
        np.ones((4, 4)), name='im', metadata={'arbitrary': object()}
    )
    layer = viewer.layers[0]
    out = _serialize_arbitrary(layer)
    assert len(out) == 3
    assert out[0] == '...'
    assert 'affine' in out[1]
    assert out[2] == 'image'
    # meta stays JSON-compatible
    json.dumps(out)


def test_serialize_layerlist_and_points_features():
    """LayerLists serialize element-wise; Points.features (DataFrame) works."""
    viewer = ViewerModel()
    viewer.add_image(np.ones((4, 4)), name='im')
    viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        features={'a': [1, 2]},
        name='pts',
    )
    out = _serialize_arbitrary(viewer.layers)
    assert isinstance(out, list)
    assert [o[2] for o in out] == ['image', 'points']
    pts_meta = out[1][1]
    # features DataFrame became plain dict of lists
    assert pts_meta['features'] == {'a': [1, 2]}
    json.dumps(out)


def test_viewer_layers_explicit_serialization_roundtrip_state():
    """Explicit layers serialization is JSON-compatible (sans data)."""
    viewer = ViewerModel()
    viewer.add_image(
        np.ones((4, 4)), name='im', metadata={'arbitrary': object()}
    )
    layer = viewer.layers[0]
    out = _serialize_arbitrary(layer)
    raw = json.dumps(out)
    new = json.loads(raw)
    assert new[0] == '...'
    assert new[1]['metadata'] == '...'
    assert new[2] == 'image'


def test_serialize_arbitrary_on_error():
    class X:
        pass

    assert _serialize_arbitrary(
        {'a': 1, 'x': X(), 'nested': {'ok': True, 'y': X()}},
        on_error='ignore',
    ) == {'a': 1, 'nested': {'ok': True}}

    assert _serialize_arbitrary({'lst': [1, X()]}, on_error='ignore') == {}
    assert _serialize_arbitrary([1, 2], on_error='ignore') == [1, 2]

    with pytest.raises(
        Exception, match='Unable to generate pydantic-core schema'
    ):
        _serialize_arbitrary({'x': X()}, on_error='raise')

    with pytest.warns(
        RuntimeWarning, match='Unable to generate pydantic-core schema'
    ):
        assert _serialize_arbitrary({'f': X()}, on_error='warn') == {}
