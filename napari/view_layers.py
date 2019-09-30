import numpy as np
import itertools
from .viewer import Viewer
from .util.misc import ensure_iterable, is_iterable


def view_image(
    data,
    *,
    rgb=None,
    is_pyramid=None,
    colormap='gray',
    contrast_limits=None,
    interpolation='nearest',
    rendering='mip',
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=1,
    blending='translucent',
    visible=True,
):
    """Create a viewer and add an image layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        an image pyramid.
    rgb : bool
        Whether the image is rgb RGB or RGBA. If not specified by user and
        the last dimension of the data has length 3 or 4 it will be set as
        `True`. If `False` the image is interpreted as a luminance image.
    is_pyramid : bool
        Whether the data is an image pyramid or not. Pyramid data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be a pyramid. The first image in the list
        should be the largest.
    colormap : str, vispy.Color.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer()
    viewer.add_image(
        data,
        rgb=rgb,
        is_pyramid=is_pyramid,
        colormap=colormap,
        contrast_limits=contrast_limits,
        interpolation=interpolation,
        rendering=rendering,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
    )
    return viewer


def view_multichannel(
    data,
    *,
    channel=-1,
    colormap=None,
    contrast_limits=None,
    interpolation='nearest',
    rendering='mip',
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=1,
    blending='additive',
    visible=True,
):
    """Create a viewer and add images layers expanding along one axis.

    Parameters
    ----------
    data : array
        Image data. Can be N dimensional.
    channel : int
        Axis to expand colors along.
    colormap : list, str, vispy.Color.Colormap, tuple, dict
        Colormaps to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap. If a list then must be same length as the axis that is
        being expanded and then each colormap is applied to each image.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image. If list of lists then must be same length as the axis
        that is being expanded and then each colormap is applied to each
        image.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer()

    n_images = data.shape[channel]

    name = ensure_iterable(name)

    base_colormaps = ['cyan', 'yellow', 'magenta', 'red', 'green', 'blue']
    if colormap is None:
        colormap = itertools.cycle(base_colormaps)
    else:
        colormap = ensure_iterable(colormap)

    # If one pair of clim values is passed then need to iterate them to
    # all layers.
    if contrast_limits is not None and not is_iterable(contrast_limits[0]):
        contrast_limits = itertools.repeat(contrast_limits)
    else:
        contrast_limits = ensure_iterable(contrast_limits)

    zipped_args = zip(range(n_images), colormap, contrast_limits, name)
    for i, cmap, clims, name in zipped_args:
        image = data.take(i, axis=channel)
        layer = viewer.add_image(
            image,
            rgb=False,
            colormap=cmap,
            contrast_limits=clims,
            interpolation=interpolation,
            rendering=rendering,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )
    return viewer


def view_points(
    data,
    *,
    symbol='o',
    size=10,
    anisotropy=None,
    edge_width=1,
    edge_color='black',
    face_color='white',
    n_dimensional=False,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=1,
    blending='translucent',
    visible=True,
):
    """Create a viewer and add a points layer.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    symbol : str
        Symbol to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, tuple
        Size of the point marker. If given as a scalar, all points are made
        the same size. If given as a tuple, size must be the same length
        as the number of points.
    anisotropy : tuple, optional
        List of anisotropy factors, must be one per dimension. These act as
        pre-multipliers on the size of each point. If not provided,
        defaults as 1 for each dimension.
    edge_width : float
        Width of the symbol edge in pixels.
    edge_color : str
        Color of the point marker border.
    face_color : str
        Color of the point marker body.
    n_dimensional : bool
        If True, renders points not just in central plane but also in all
        n-dimensions according to specified point marker size.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.

    Notes
    -----
    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual
    """
    viewer = Viewer()
    viewer.add_points(
        data,
        symbol=symbol,
        size=size,
        anisotropy=anisotropy,
        edge_width=edge_width,
        edge_color=edge_color,
        face_color=face_color,
        n_dimensional=n_dimensional,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
    )
    return viewer


def view_labels(
    data,
    *,
    num_colors=50,
    seed=0.5,
    n_dimensional=False,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=0.7,
    blending='translucent',
    visible=True,
):
    """Create a viewer and add a labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    data : array
        Labels data.
    num_colors : int
        Number of unique colors to use in colormap.
    seed : float
        Seed for colormap random generator.
    n_dimensional : bool
        If `True`, paint and fill edit labels across all dimensions.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer()
    viewer.add_labels(
        data,
        num_colors=num_colors,
        seed=seed,
        n_dimensional=n_dimensional,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
    )
    return viewer


def view_shapes(
    data,
    *,
    shape_type='rectangle',
    edge_width=1,
    edge_color='black',
    face_color='white',
    z_index=0,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=0.7,
    blending='translucent',
    visible=True,
):
    """Create a viewer and add a shapes layer.

    Parameters
    ----------
    data : list or array
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions. Can be an 3-dimensional
        array if each shape has the same number of vertices.
    shape_type : string or list
        String of shape shape_type, must be one of "{'line', 'rectangle',
        'ellipse', 'path', 'polygon'}". If a list is supplied it must be
        the same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    edge_width : float or list
        Thickness of lines and edges. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    edge_color : str or list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    face_color : str or list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    z_index : int or list
        Specifier of z order priority. Shapes with higher z order are
        displayed ontop of others. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float or list
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer()
    viewer.add_shapes(
        data,
        shape_type=shape_type,
        edge_width=edge_width,
        edge_color=edge_color,
        face_color=face_color,
        z_index=z_index,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
    )
    return viewer


def view_surface(
    data,
    *,
    colormap='gray',
    contrast_limits=None,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=1,
    blending='translucent',
    visible=True,
):
    """Create a viewer and add a surface layer.

    Parameters
    ----------
    data : 3-tuple of array
        The first element of the tuple is an (N, D) array of vertices of
        mesh triangles. The second is an (M, 3) array of int of indices
        of the mesh triangles. The third element is the (N, ) array of
        values used to color vertices.
    colormap : str, vispy.Color.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer()
    viewer.add_surface(
        data,
        colormap=colormap,
        contrast_limits=contrast_limits,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
    )
    return viewer


def view_vectors(
    data,
    *,
    edge_width=1,
    edge_color='red',
    length=1,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=0.7,
    blending='translucent',
    visible=True,
):
    """Create a viewer and add a vectors layer.

    Parameters
    ----------
    data : (N, 2, D) or (N1, N2, ..., ND, D) array
        An (N, 2, D) array is interpreted as "coordinate-like" data and a
        list of N vectors with start point and projections of the vector in
        D dimensions. An (N1, N2, ..., ND, D) array is interpreted as
        "image-like" data where there is a length D vector of the
        projections at each pixel.
    edge_width : float
        Width for all vectors in pixels.
    length : float
         Multiplicative factor on projections for length of all vectors.
    edge_color : str
        Edge color of all the vectors.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer()
    viewer.add_vectors(
        data,
        edge_width=edge_width,
        edge_color=edge_color,
        length=length,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
    )
    return viewer
