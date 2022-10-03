"""Methods to create a new viewer instance then add a particular layer type.

All functions follow this pattern, (where <layer_type> is replaced with one
of the layer types, like "image", "points", etc...):

.. code-block:: python

    def view_<layer_type>(*args, **kwargs):
        # ... pop all of the viewer kwargs out of kwargs into viewer_kwargs
        viewer = Viewer(**viewer_kwargs)
        add_method = getattr(viewer, f"add_{<layer_type>}")
        add_method(*args, **kwargs)
        return viewer
"""
import inspect
from typing import Any, List, Tuple

from numpydoc.docscrape import NumpyDocString as _NumpyDocString

from napari.components.dims import Dims
from napari.layers import Image

from .viewer import Viewer

__all__ = [
    'view_image',
    'view_labels',
    'view_path',
    'view_points',
    'view_shapes',
    'view_surface',
    'view_tracks',
    'view_vectors',
    'imshow',
]

_doc_template = """Create a viewer and add a{n} {layer_string} layer.

{params}

Returns
-------
viewer : :class:`napari.Viewer`
    The newly-created viewer.
"""

_VIEW_DOC = _NumpyDocString(Viewer.__doc__)
_VIEW_PARAMS = "    " + "\n".join(_VIEW_DOC._str_param_list('Parameters')[2:])


def _merge_docstrings(add_method, layer_string):
    # create combined docstring with parameters from add_* and Viewer methods
    import textwrap

    add_method_doc = _NumpyDocString(add_method.__doc__)

    # this ugliness is because the indentation of the parsed numpydocstring
    # is different for the first parameter :(
    lines = add_method_doc._str_param_list('Parameters')
    lines = lines[:3] + textwrap.dedent("\n".join(lines[3:])).splitlines()
    params = "\n".join(lines) + "\n" + textwrap.dedent(_VIEW_PARAMS)
    n = 'n' if layer_string.startswith(tuple('aeiou')) else ''
    return _doc_template.format(n=n, layer_string=layer_string, params=params)


def _merge_layer_viewer_sigs_docs(func):
    """Make combined signature, docstrings, and annotations for `func`.

    This is a decorator that combines information from `Viewer.__init__`,
    and one of the `viewer.add_*` methods.  It updates the docstring,
    signature, and type annotations of the decorated function with the merged
    versions.

    Parameters
    ----------
    func : callable
        `view_<layer_type>` function to modify

    Returns
    -------
    func : callable
        The same function, with merged metadata.
    """
    from .utils.misc import _combine_signatures

    # get the `Viewer.add_*` method
    layer_string = func.__name__.replace("view_", "")
    if layer_string == 'path':
        add_method = Viewer.open
    else:
        add_method = getattr(Viewer, f'add_{layer_string}')

    # merge the docstrings of Viewer and viewer.add_*
    func.__doc__ = _merge_docstrings(add_method, layer_string)

    # merge the signatures of Viewer and viewer.add_*
    func.__signature__ = _combine_signatures(
        add_method, Viewer, return_annotation=Viewer, exclude=('self',)
    )

    # merge the __annotations__
    func.__annotations__ = {
        **add_method.__annotations__,
        **Viewer.__init__.__annotations__,
        'return': Viewer,
    }

    # _forwardrefns_ is used by stubgen.py to populate the globalns
    # when evaluate forward references with get_type_hints
    func._forwardrefns_ = {**add_method.__globals__}
    return func


_viewer_params = inspect.signature(Viewer).parameters
_dims_params = Dims.__fields__


def _make_viewer_then(
    add_method: str,
    args,
    kwargs,
) -> Tuple[Viewer, Any]:
    """Create a viewer, call given add_* method, then return viewer and layer.

    This function will be deprecated soon (See #4693)

    Parameters
    ----------
    add_method : str
        Which ``add_`` method to call on the viewer, e.g. `add_image`,
        or `add_labels`.
    args : list
        Positional arguments for the ``add_`` method.
    kwargs : dict
        Keyword arguments for either the `Viewer` constructor or for the
        ``add_`` method.
    viewer : Viewer, optional
        A pre-existing viewer, which will be used provided, rather than
        creating a new one.

    Returns
    -------
    viewer : napari.Viewer
        The created viewer, or the same one that was passed in, if given.
    layer(s): napari.layers.Layer or List[napari.layers.Layer]
        The value returned by the add_method. Can be a list of layers if
        ``add_image`` is called with a ``channel_axis=`` keyword
        argument.
    """
    vkwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _viewer_params}
    # separate dims kwargs because we want to set those after adding data
    dims_kwargs = {
        k: vkwargs.pop(k) for k in list(vkwargs) if k in _dims_params
    }
    viewer = kwargs.pop("viewer", None)
    if viewer is None:
        viewer = Viewer(**vkwargs)
    kwargs.update(kwargs.pop("kwargs", {}))
    method = getattr(viewer, add_method)
    added = method(*args, **kwargs)
    if isinstance(added, list):
        added = tuple(added)
    for arg_name, arg_val in dims_kwargs.items():
        setattr(viewer.dims, arg_name, arg_val)
    return viewer, added


# Each of the following functions will have this pattern:
#
# def view_image(*args, **kwargs):
#     # ... pop all of the viewer kwargs out of kwargs into viewer_kwargs
#     viewer = Viewer(**viewer_kwargs)
#     viewer.add_image(*args, **kwargs)
#     return viewer


@_merge_layer_viewer_sigs_docs
def view_image(*args, **kwargs):
    return _make_viewer_then('add_image', args, kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_labels(*args, **kwargs):
    return _make_viewer_then('add_labels', args, kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_points(*args, **kwargs):
    return _make_viewer_then('add_points', args, kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_shapes(*args, **kwargs):
    return _make_viewer_then('add_shapes', args, kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_surface(*args, **kwargs):
    return _make_viewer_then('add_surface', args, kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_tracks(*args, **kwargs):
    return _make_viewer_then('add_tracks', args, kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_vectors(*args, **kwargs):
    return _make_viewer_then('add_vectors', args, kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_path(*args, **kwargs):
    return _make_viewer_then('open', args, kwargs)[0]


def imshow(
    data,
    *,
    channel_axis=None,
    rgb=None,
    colormap=None,
    contrast_limits=None,
    gamma=1,
    interpolation2d='nearest',
    interpolation3d='linear',
    rendering='mip',
    depiction='volume',
    iso_threshold=0.5,
    attenuation=0.05,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    rotate=None,
    shear=None,
    affine=None,
    opacity=1,
    blending=None,
    visible=True,
    multiscale=None,
    cache=True,
    plane=None,
    experimental_clipping_planes=None,
    viewer=None,
    title='napari',
    ndisplay=2,
    order=(),
    axis_labels=(),
    show=True,
) -> Tuple[Viewer, List[Image]]:
    """Load data into an Image layer and return the Viewer and Layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N >= 2 dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    channel_axis : int, optional
        Axis to expand image along.  If provided, each channel in the data
        will be added as an individual image layer.  In channel_axis mode,
        all other parameters MAY be provided as lists, and the Nth value
        will be applied to the Nth channel in the data.  If a single value
        is provided, it will be broadcast to all Layers.
    rgb : bool or list
        Whether the image is rgb RGB or RGBA. If not specified by user and
        the last dimension of the data has length 3 or 4 it will be set as
        `True`. If `False` the image is interpreted as a luminance image.
        If a list then must be same length as the axis that is being
        expanded as channels.
    colormap : str, napari.utils.Colormap, tuple, dict, list
        Colormaps to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap. If a list then must be same length as the axis that is
        being expanded as channels, and each colormap is applied to each
        new image layer.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image. If list of lists then must be same length as the axis
        that is being expanded and then each colormap is applied to each
        image.
    gamma : list, float
        Gamma correction for determining colormap linearity. Defaults to 1.
        If a list then must be same length as the axis that is being
        expanded as channels.
    interpolation : str or list
        Deprecated, to be removed in 0.6.0
    interpolation2d : str or list
        Interpolation mode used by vispy in 2D. Must be one of our supported
        modes. If a list then must be same length as the axis that is being
        expanded as channels.
    interpolation3d : str or list
        Interpolation mode used by vispy in 3D. Must be one of our supported
        modes. If a list then must be same length as the axis that is being
        expanded as channels.
    rendering : str or list
        Rendering mode used by vispy. Must be one of our supported
        modes. If a list then must be same length as the axis that is being
        expanded as channels.
    depiction : str
        Selects a preset volume depiction mode in vispy

        * volume: images are rendered as 3D volumes.
        * plane: images are rendered as 2D planes embedded in 3D.
    iso_threshold : float or list
        Threshold for isosurface. If a list then must be same length as the
        axis that is being expanded as channels.
    attenuation : float or list
        Attenuation rate for attenuated maximum intensity projection. If a
        list then must be same length as the axis that is being expanded as
        channels.
    name : str or list of str
        Name of the layer.  If a list then must be same length as the axis
        that is being expanded as channels.
    metadata : dict or list of dict
        Layer metadata. If a list then must be a list of dicts with the
        same length as the axis that is being expanded as channels.
    scale : tuple of float or list
        Scale factors for the layer. If a list then must be a list of
        tuples of float with the same length as the axis that is being
        expanded as channels.
    translate : tuple of float or list
        Translation values for the layer. If a list then must be a list of
        tuples of float with the same length as the axis that is being
        expanded as channels.
    rotate : float, 3-tuple of float, n-D array or list.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed. If a list then must have same length as
        the axis that is being expanded as channels.
    shear : 1-D array or list.
        A vector of shear values for an upper triangular n-D shear matrix.
        If a list then must have same length as the axis that is being
        expanded as channels.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a
        napari `Affine` transform object. Applied as an extra transform on
        top of the provided scale, rotate, and shear values.
    opacity : float or list
        Opacity of the layer visual, between 0.0 and 1.0.  If a list then
        must be same length as the axis that is being expanded as channels.
    blending : str or list
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}. If a list then
        must be same length as the axis that is being expanded as channels.
    visible : bool or list of bool
        Whether the layer visual is currently being displayed.
        If a list then must be same length as the axis that is
        being expanded as channels.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon
        retrieval. Currently, this only applies to dask arrays.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    viewer: Viewer object, optional, by default None.
    title : string, optional
        The title of the viewer window. By default 'napari'.
    ndisplay : {2, 3}, optional
        Number of displayed dimensions. By default 2.
    order : tuple of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3. By default None
    axis_labels : list of str, optional
        Dimension names. By default they are labeled with sequential numbers
    show : bool, optional
        Whether to show the viewer after instantiation. By default True.

    Returns
    -------
    viewer : napari.Viewer
        The created or passed viewer.
    layer(s) : napari.layers.Image or List[napari.layers.Image]
        The added layer(s). (May be more than one if the ``channel_axis`` keyword
        argument is given.
    """

    kwargs = dict(
        viewer=viewer,
        channel_axis=channel_axis,
        rgb=rgb,
        colormap=colormap,
        contrast_limits=contrast_limits,
        gamma=gamma,
        interpolation2d=interpolation2d,
        interpolation3d=interpolation3d,
        rendering=rendering,
        depiction=depiction,
        iso_threshold=iso_threshold,
        attenuation=attenuation,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        rotate=rotate,
        shear=shear,
        affine=affine,
        opacity=opacity,
        blending=blending,
        visible=visible,
        multiscale=multiscale,
        cache=cache,
        plane=plane,
        experimental_clipping_planes=experimental_clipping_planes,
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )

    args = tuple([data])

    viewer, layers = _make_viewer_then(
        'add_image',
        args,
        kwargs,
    )

    return viewer, layers
