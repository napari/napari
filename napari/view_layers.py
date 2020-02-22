from .viewer import Viewer


def view_image(
    data=None,
    *,
    channel_axis=None,
    rgb=None,
    is_pyramid=None,
    colormap=None,
    contrast_limits=None,
    gamma=1,
    interpolation='nearest',
    rendering='mip',
    iso_threshold=0.5,
    attenuation=0.5,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=1,
    blending=None,
    visible=True,
    path=None,
    title='napari',
    ndisplay=2,
    order=None,
    axis_labels=None,
    show=True,
):
    """Create a viewer and add an image layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        an image pyramid.
    channel_axis : int, optional
        Axis to expand image along.
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
    colormap : str, vispy.Color.Colormap, tuple, dict, list
        Colormaps to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap. If a list then must be same length as the axis that is
        being expanded as channels, and each colormap is applied to each new
        image layer.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image. If list of lists then must be same length as the axis
        that is being expanded and then each colormap is applied to each
        image.
    gamma : list, float
        Gamma correction for determining colormap linearity. Defaults to 1.
        If a list then must be same length as the axis that is being expanded
        and then each entry in the list is applied to each image.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    iso_threshold : float
        Threshold for isosurface.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.
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
    path : str or list of str
        Path or list of paths to image data. Paths can be passed as strings
        or `pathlib.Path` instances.
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer(
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )
    viewer.add_image(
        data=data,
        channel_axis=channel_axis,
        rgb=rgb,
        is_pyramid=is_pyramid,
        colormap=colormap,
        contrast_limits=contrast_limits,
        gamma=gamma,
        interpolation=interpolation,
        rendering=rendering,
        iso_threshold=iso_threshold,
        attenuation=attenuation,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
        path=path,
    )
    return viewer


def view_points(
    data=None,
    *,
    properties=None,
    symbol='o',
    size=10,
    edge_width=1,
    edge_color="black",
    edge_color_cycle=None,
    edge_colormap='viridis',
    edge_contrast_limits=None,
    face_color="white",
    face_color_cycle=None,
    face_colormap='viridis',
    face_contrast_limits=None,
    n_dimensional=False,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=1,
    blending='translucent',
    visible=True,
    title='napari',
    ndisplay=2,
    order=None,
    axis_labels=None,
    show=True,
):
    """Create a viewer and add a points layer.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    symbol : str
        Symbol to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, array
        Size of the point marker. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same
        broadcastable to the same shape as the data.
    edge_width : float
        Width of the symbol edge in pixels.
    edge_color : str, array-like
        Color of the point marker border. Numeric color values should be RGB(A).
    edge_color_cycle : np.ndarray, list, cycle
        Cycle of colors (provided as RGBA) to map to edge_color if a
        categorical attribute is used to set face_color.
    edge_colormap : str, vispy.color.colormap.Colormap
        Colormap to set edge_color if a continuous attribute is used to set face_color.
        See vispy docs for details: http://vispy.org/color.html#vispy.color.Colormap
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : str, array-like
        Color of the point marker body. Numeric color values should be RGB(A).
    face_color_cycle : np.ndarray, list, cycle
        Cycle of colors (provided as RGBA) to map to face_color if a
        categorical attribute is used to set face_color.
    face_colormap : str, vispy.color.colormap.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
        See vispy docs for details: http://vispy.org/color.html#vispy.color.Colormap
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
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
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.

    Notes
    -----
    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual
    """
    viewer = Viewer(
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )
    viewer.add_points(
        data=data,
        properties=properties,
        symbol=symbol,
        size=size,
        edge_width=edge_width,
        edge_color=edge_color,
        edge_color_cycle=edge_color_cycle,
        edge_colormap=edge_colormap,
        edge_contrast_limits=edge_contrast_limits,
        face_color=face_color,
        face_color_cycle=face_color_cycle,
        face_colormap=face_colormap,
        face_contrast_limits=face_contrast_limits,
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
    data=None,
    *,
    is_pyramid=None,
    num_colors=50,
    seed=0.5,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=0.7,
    blending='translucent',
    visible=True,
    path=None,
    title='napari',
    ndisplay=2,
    order=None,
    axis_labels=None,
    show=True,
):
    """Create a viewer and add a labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    data : array or list of array
        Labels data as an array or pyramid.
    is_pyramid : bool
        Whether the data is an image pyramid or not. Pyramid data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be a pyramid. The first image in the list
        should be the largest.
    num_colors : int
        Number of unique colors to use in colormap.
    seed : float
        Seed for colormap random generator.
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
    path : str or list of str
        Path or list of paths to image data. Paths can be passed as strings
        or `pathlib.Path` instances.
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer(
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )
    viewer.add_labels(
        data=data,
        is_pyramid=is_pyramid,
        num_colors=num_colors,
        seed=seed,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        opacity=opacity,
        blending=blending,
        visible=visible,
        path=path,
    )
    return viewer


def view_shapes(
    data=None,
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
    title='napari',
    ndisplay=2,
    order=None,
    axis_labels=None,
    show=True,
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
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer(
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )
    viewer.add_shapes(
        data=data,
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
    gamma=1,
    name=None,
    metadata=None,
    scale=None,
    translate=None,
    opacity=1,
    blending='translucent',
    visible=True,
    title='napari',
    ndisplay=2,
    order=None,
    axis_labels=None,
    show=True,
):
    """Create a viewer and add a surface layer.

    Parameters
    ----------
    data : 3-tuple of array
        The first element of the tuple is an (N, D) array of vertices of
        mesh triangles. The second is an (M, 3) array of int of indices
        of the mesh triangles. The third element is the (K0, ..., KL, N)
        array of values used to color vertices where the additional L
        dimensions are used to color the same mesh with different values.
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
    gamma : float
        Gamma correction for determining colormap linearity. Defaults to 1.
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
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer(
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )
    viewer.add_surface(
        data,
        colormap=colormap,
        contrast_limits=contrast_limits,
        gamma=gamma,
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
    title='napari',
    ndisplay=2,
    order=None,
    axis_labels=None,
    show=True,
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
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer(
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )
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
