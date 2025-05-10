from functools import partial

import numpy as np

import napari
from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)
from napari.layers.utils.string_encoding import ConstantStringEncoding


def path_length(vertices, connect_last=False):
    segment_lengths = np.linalg.norm(
        vertices - np.roll(vertices, -1, axis=0), axis=1
    )
    if connect_last:
        return np.sum(segment_lengths)
    return np.sum(segment_lengths[:-1])


def polygon_area(vertices):
    coords = np.asarray(vertices)
    n, dims = coords.shape
    area = 0

    if n < 3:
        return 0

    area = 0.0

    # use a reference vertex to make the polygon out of a "triangle fan"
    ref = coords[0]
    for i in range(1, n - 1):
        u = coords[i] - ref
        v = coords[i + 1] - ref
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        cross_sq = uu * vv - uv**2
        triangle_area = 0.5 * np.sqrt(max(cross_sq, 0))
        area += triangle_area

    return area


def rectangle_area(vertices):
    return np.linalg.norm(vertices[1] - vertices[0]) * np.linalg.norm(
        vertices[2] - vertices[1]
    )


def ellipse_area(vertices):
    return rectangle_area(vertices) * np.pi / 4


def ellipse_perimeter(vertices):
    # no closed form exists, so we use Ramanujan's 2nd approximation
    # https://en.wikipedia.org/wiki/Perimeter_of_an_ellipse#Ramanujan's_approximations
    a = np.linalg.norm(vertices[1] - vertices[0]) / 2
    b = np.linalg.norm(vertices[2] - vertices[1]) / 2
    h = (a - b) ** 2 / (a + b) ** 2
    return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))


def update_features_with_measures(shapes_layer, event=None):
    for i, s in enumerate(shapes_layer._data_view.shapes):
        if isinstance(s, Polygon):
            a = polygon_area(s.data)
            p = path_length(s.data, connect_last=True)
        elif isinstance(s, Rectangle):
            a = rectangle_area(s.data)
            p = path_length(s.data, connect_last=True)
        elif isinstance(s, Ellipse):
            a = ellipse_area(s.data)
            p = ellipse_perimeter(s.data)
        elif isinstance(s, Path | Line):
            a = 0
            p = path_length(s.data, connect_last=False)
        shapes_layer.features.loc[i, ['_perimeter', '_area']] = p, a

    shapes_layer.refresh_text()


def get_connected_callback(shapes_layer):
    """Get the connected measure callback if present."""
    for callback in shapes_layer.events.set_data.callbacks:
        f = callback[0]() if isinstance(callback, tuple) else callback
        if isinstance(f, partial) and f.func is update_features_with_measures:
            return f
    return None


def toggle_shape_measures(shapes_layer: napari.layers.Shapes) -> None:
    """Toggle between updating and displaying measures and not."""
    if not get_connected_callback(shapes_layer):
        shapes_layer.features[['_perimeter', '_area']] = 0.0
        shapes_layer.feature_defaults = (
            shapes_layer.feature_defaults.to_dict().update(
                {'_perimeter': 0.0, '_area': 0.0}
            )
        )

        shapes_layer.events.set_data.connect(
            partial(update_features_with_measures, shapes_layer)
        )
        update_features_with_measures(shapes_layer)

        shapes_layer.text = 'P = {_perimeter:.3g}\nA = {_area:.3g}'
    else:
        shapes_layer.events.set_data.disconnect(
            get_connected_callback(shapes_layer)
        )

        # need to explicitly set constant encoding to avoid warning, why?
        shapes_layer.text = {'string': ConstantStringEncoding(constant='')}

        shapes_layer.features = shapes_layer.features.drop(
            columns=['_perimeter', '_area']
        )
        # feature_defaults are dropped automatically
