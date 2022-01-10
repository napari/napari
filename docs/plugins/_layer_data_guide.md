(layer-data-tuples)=
## The LayerData tuple

When transfering data to and from plugins, napari does not pass `Layer` objects
directly. Instead, it passes (mostly) pure-python and array-like types,
deconstructed into a {class}`tuple` that we refer to as a `LayerData` tuple.  This type shows
up often in plugins and is explained here.

### Informal description

```py
(data, [attributes, [layer_type]])
```

A `LayerData` tuple is a tuple of length 1, 2, or 3 whose items, in order, are:

1. The `data` object that would be used for `layer.data` (such as a numpy array
for the `Image` layer)
2. *(Optional).* A {class}`dict` of layer attributes, suitable for passing as
keyword arguments to the corresponding layer constructor (e.g. `{'opacity': 0.7}`)
3. *(Optional).* A lower case {class}`str` indicating the layer type (e.g.`'image'`,
`'labels'`, etc...).  If not provided (i.e. if the tuple is only of length 2), the
layer type is assumed to be `'image`'.

### Formal type definition

Formally, the typing for `LayerData` looks like this:

```python
LayerData = Union[Tuple[DataType], Tuple[DataType, LayerProps], FullLayerData]
```

where ...

```python
from typing import Literal, Protocol, Sequence

LayerTypeName = Literal[
    "image", "labels", "points", "shapes", "surface", "tracks", "vectors"
]
LayerProps = Dict
DataType = Union[ArrayLike, Sequence[ArrayLike]]
FullLayerData = Tuple[DataType, LayerProps, LayerTypeName]
LayerData = Union[Tuple[DataType], Tuple[DataType, LayerProps], FullLayerData]


# where "ArrayLike" is very roughly ...
class ArrayLike(Protocol):
    shape: Tuple[int, ...]
    ndim: int
    dtype: np.dtype
    def __array__(self) -> np.ndarray: ...
    def __getitem__(self, key) -> ArrayLike: ...

# the main point is that we're more concerned with structural
# typing than literal array types (e.g. numpy, dask, xarray, etc...)
```

### Examples

Assume that `data` is a numpy array:

```python
import numpy as np
data = np.random.rand(64, 64)
```

All of the following are valid `LayerData` tuples:

```python
# the first three are equivalent, just an image array with default settings
(data,)
(data, {})
(data, {}, 'image')

# provide kwargs for image contructor
(data, {'name': 'My Image', 'colormap': 'red'})

# labels layer instead of image:
(data.astype(int), {'name': 'My Labels', 'blending': 'additive'}, 'labels')
```

### Creation from a `Layer` instance.

Note, the {meth}`~napari.layers.Layer.as_layer_data_tuple` method will create a layer data
tuple from a given layer

```python
>>> img = Image(np.random.rand(2, 2), colormap='green', scale=(4, 4))

>>> img.as_layer_data_tuple()
Out[7]:
(
    array([[0.94414642, 0.89192899],
       [0.21258344, 0.85242735]]),
    {
        'name': 'Image',
        'metadata': {},
        'scale': [4.0, 4.0],
        'translate': [0.0, 0.0],
        'rotate': [[1.0, 0.0], [0.0, 1.0]],
        'shear': [0.0],
        'opacity': 1,
        'blending': 'translucent',
        'visible': True,
        'experimental_clipping_planes': [],
        'rgb': False,
        'multiscale': False,
        'colormap': 'green',
        'contrast_limits': [0.2125834437981784, 0.9441464162780605],
        'interpolation': 'nearest',
        'rendering': 'mip',
        'experimental_slicing_plane': {'normal': (1.0, 0.0, 0.0), 'position': (0.0, 0.0, 0.0), 'enabled': False, 'thickness': 1.0},
        'iso_threshold': 0.5,
        'attenuation': 0.05,
        'gamma': 1
    },
    'image'
)
```