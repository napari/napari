(color)=

# Colors and colormaps

This document describes how to programmatically define colors and colormaps in napari.
It is intended for users of napari that are comfortable using the napari API and
developers who want to include color as part of something they are building.


## Colors

In many places, napari allows the specification of color.
This might be the color of an individual point, the color of text annotations, or a color value in a colormap.
For example, the face color of all points in a layer can be defined as red.

```python
In [1]: points.face_color = 'red'
```

### Standard form

Internally, napari typically coerces an input color value like `'red'` to an RGBA (red, green, blue, alpha)
array with `shape=(4,)` and `dtype=np.float32` where the values are in `[0, 1]`.
When storing N color values, a similar array of `shape=(N, 4)` is used where each row represents a single color value.
For example, reading the points layer's red face color gives us such an array.

```python
In [2]: points.face_color
Out[2]:
array([[1., 0., 0., 1.],
       [1., 0., 0., 1.],
       [1., 0., 0., 1.]])
```

The [`transform_color`](https://github.com/napari/napari/blob/5f96d5d814aad697c367bdadbb1a57750e2114ad/napari/utils/colormaps/standardize_color.py#L33)
function is responsible for coercing color input values to the standard form and supports a variety of inputs.

```{note}
The [`ColorType`](https://github.com/napari/napari/blob/43d9b59da89b9a2b97dba3967a7b28526f9aa0ce/napari/utils/colormaps/colormap_utils.py#L17)
type definition almost describes all the types of acceptable color inputs, except that it includes
vispy's `Color` and `ColorArray`, which are not accepted by `transform_color`.
```


### Supported single color input

The single face color could have been defined as red in many equivalent ways.

```python
# CSS3 name
points.face_color = 'red'

# matplotlib single character name
points.face_color = 'r'

# RGB or RGBA hex-code
points.face_color = '#ff0000'
points.face_color = '#ff0000ff'

# RGB or RGBA tuple, list, or array
points.face_color = (1, 0, 0)
points.face_color = (1, 0, 0, 1)
points.face_color = [1, 0, 0]
points.face_color = [1, 0, 0, 1]
points.face_color = np.array([1, 0, 0])
points.face_color = np.array([1, 0, 0, 1])
```

A named color string must either be from the [CSS3 specification](https://www.w3.org/TR/css-color-3/#svg-color)
or must be one of [matplotlib's single character shorthand names](https://matplotlib.org/stable/tutorials/colors/colors.html#specifying-colors).


### Supported multiple color input

Sometimes there is a need to specify many colors at once.
For example, the face colors of all the points in a layer can be specified individually

```python
In [3]: points.face_color = ['red', 'lime', 'blue']
```

which will then be coerced into the standard array form.

```python
In [4]: points.face_color
Out[4]:
array([[1., 0., 0., 1.],
       [0., 1., 0., 1.],
       [0., 0., 1., 1.]])
```

For the most part, any tuple, list, array, or generator of single colors
is acceptable input when specifying multiple colors.

```python
# Tuple, list, array, or generator of names
points.face_color = ('red', 'lime', 'blue')
points.face_color = ['red', 'lime', 'blue']
points.face_color = np.array(['red', 'lime', 'blue'])
points.face_color = c for c in ('red', 'lime', 'blue')

# List of RGB or RGBA tuples or lists
points.face_color = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]
points.face_color = [
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
]

# (N, 3) RGB or (N, 4) RGBA array
points.face_color = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
points.face_color = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
])
```

However, when specifying many colors in a sequence or generator *do not mix different single color types*.
For example, something like

```python
In [5]: points.face_color = np.array(['red', (0, 1, 0), '#0000ff'])
```

is not supported or guaranteed to work.


## Colormaps

In general, a colormap is a function that maps from an arbitrary domain to a some color space range.
As napari's standard form of color is an RGBA array, this range is the RGBA color space.

In napari, colormaps are used in two ways.
- To map an image pixel value to an RGBA color.
- To map a property value to an RGBA color.

In both cases the types of values in the domain may be continuous (e.g. `float`)
or categorical (e.g. `int`, `str`), so napari has two different types of colormaps to handle those cases.


### Continuous maps

The [`Colormap`](https://github.com/napari/napari/blob/c86cf87a788b1bbe62afc0b92b9ebdca1331a3e5/napari/utils/colormaps/colormap.py#L28)
class defines a map from a continuous or real value to a color.
For example, a 2D image of floats between 0 and 1 can be mapped to colors.

```python
In [6]: image.colormap = 'blue'
```

Internally, the [`ensure_colormap`](https://github.com/napari/napari/blob/c86cf87a788b1bbe62afc0b92b9ebdca1331a3e5/napari/utils/colormaps/colormap_utils.py#L492)
function is used to coerce colormap input values to an instance of napari's `Colormap`.

```python
In [7]: image.colormap
Out[7]:
Colormap(
    colors=np.array([0., 0., 0., 1], [0., 0., 1., 1]]),
    name='blue',
    interpolation=<ColormapInterpolationMode.LINEAR: 'linear'>,
    controls=array([0., 1.]),
)
```

In this case, values of 0 are mapped to black (`[0, 0, 0, 1]`),
values of 1 are mapped to blue (`[0, 0, 1, 1]`),
and values in `(0, 1)` are mapped to colors that are linearly interpolated
between black and blue.

A few types of input are supported.
The easiest to use is the colormap name as above.
```python
image.colormap = 'blue'
```

By default colormap names come from one of the following sources.

- Some of [matplotlib's colormap names](https://matplotlib.org/stable/tutorials/colors/colormaps.html).
- Standard optical primary and secondary colors: `'red', 'green', 'blue', 'yellow', 'magenta', 'cyan'`.
- BOP (blue, orange, purple) colormap names: `'bop blue', 'bop orange', 'bop purple'`.

More specifically, the available colormap names can be listed at any time.

```python
In [8]: list(napari.utils.colormaps.AVAILABLE_COLORMAPS)
Out[8]:
['PiYG',
 'blue',
 'bop blue',
 ...,
 'viridis',
 'yellow']
```

If there is a need for a custom colormap that is not available by default,
pass an instance of a napari `Colormap`


```python
image.colormap = Colormap(
    colors=[[0, 0, 0, 1], [0, 0, 1, 1]],
    name='blue',
    interpolation='linear',
    controls=[0, 1],
)
```

which will be added to `AVAILABLE_COLORMAPS` after it is successfully created.
Equivalently, pass a `dict` representation of the same colormap.

```python
image.colormap = {
    'colors': [[0, 0, 0, 1], [0, 0, 1, 1]],
    'name': 'blue',
    'interpolation': 'linear',
    'controls': [0, 1],
}
```

Any single color value that is not a string can also be used

```python
image.colormap = (0, 0, 1)
```

in which case a colormap from black to the given color will be generated.
In general, the color value cannot be given as a name string because strings
are reserved to refer to existing colormap names.


### Categorical maps

The [`CategoricalColormap`](https://github.com/napari/napari/blob/c86cf87a788b1bbe62afc0b92b9ebdca1331a3e5/napari/utils/colormaps/categorical_colormap.py#L13)
class defines a map from a categorical or discrete value to a color.
It is defined by two components.

1. A dictionary that maps from values to colors.
2. A cycle of fallback colors that used when a value is not found in the dictionary.

For example, a points layer's face color could be mapped from an optional cell type

```python
In  [9]: points.properties = {'cell_type': ['astrocyte', 'microglia', None]}
In [10]: points.face_color_cycle = {
             'colormap': {
                 'astrocyte': 'red',
                 'oligodendrocyte': 'yellow',
                 None: 'blue',
             },
             'fallback_color': ['lime', 'magenta', 'cyan'],
         }
In [11]: points.face_color = 'cell_type'
In [12]: points.face_color
Out[12]:
array([[1., 0., 0., 1.],
       [0., 1., 0., 1.],
       [0., 0., 1., 1.]], dtype=float32)
```

Alternatively, a dictionary alone can be used to define the mapping

```python
In [13]: points.face_color_cycle = {
            'astrocyte': 'red',
            'oligodendrocyte': 'yellow',
            None: 'blue',
         }
```

in which case the fallback color will default to `['white']`.

Or a sequence alone can be used to defined the fallback color cycle

```python
In [14]: points.face_color_cycle = ['lime', 'magenta', 'cyan']
```

in which case the mapping will initially be an empty dictionary.
As values from the fallback cycle are used, the value to color mappings
they define will be added to the dictionary.

