import numpy as np
from math import inf
import itertools
from xml.etree.ElementTree import Element, tostring

from .dims import Dims
from .layerlist import LayerList
from .. import layers
from ..util import colormaps
from ..util.event import EmitterGroup, Event
from ..util.keybindings import KeymapMixin
from ..util.theme import palettes
from ..util.misc import ensure_iterable, is_iterable
from ..util import io


class ViewerModel(KeymapMixin):
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders, and control bars for color limits.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels = list of str
        Dimension names.

    Attributes
    ----------
    window : Window
        Parent window.
    layers : LayerList
        List of contained layers.
    dims : Dimensions
        Contains axes, indices, dimensions and sliders.
    themes : dict of str: dict of str: str
        Preset color palettes.
    """

    themes = palettes

    def __init__(
        self, title='napari', ndisplay=2, order=None, axis_labels=None
    ):
        super().__init__()

        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            status=Event,
            help=Event,
            title=Event,
            interactive=Event,
            cursor=Event,
            reset_view=Event,
            active_layer=Event,
            palette=Event,
            grid=Event,
            layers_change=Event,
        )

        self.dims = Dims(
            ndim=None, ndisplay=ndisplay, order=order, axis_labels=axis_labels
        )

        self.layers = LayerList()

        self._status = 'Ready'
        self._help = ''
        self._title = title
        self._cursor = 'standard'
        self._cursor_size = None
        self._interactive = True
        self._active_layer = None
        self._grid_size = (1, 1)
        self.grid_stride = 1

        self._palette = None
        self.theme = 'dark'

        self.dims.events.camera.connect(lambda e: self.reset_view())
        self.dims.events.ndisplay.connect(lambda e: self._update_layers())
        self.dims.events.order.connect(lambda e: self._update_layers())
        self.dims.events.axis.connect(lambda e: self._update_layers())
        self.layers.events.added.connect(self._on_layers_change)
        self.layers.events.removed.connect(self._on_layers_change)
        self.layers.events.added.connect(self._update_active_layer)
        self.layers.events.removed.connect(self._update_active_layer)
        self.layers.events.reordered.connect(self._update_active_layer)
        self.layers.events.added.connect(lambda e: self._update_grid())
        self.layers.events.removed.connect(lambda e: self._update_grid())
        self.layers.events.reordered.connect(lambda e: self._update_grid())

        # Hold callbacks for when mouse moves with nothing pressed
        self.mouse_move_callbacks = []
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks = []
        self._persisted_mouse_event = {}
        self._mouse_drag_gen = {}

    @property
    def palette(self):
        """dict of str: str : Color palette with which to style the viewer.
        """
        return self._palette

    @palette.setter
    def palette(self, palette):
        if palette == self.palette:
            return

        self._palette = palette
        self.events.palette(palette=palette)

    @property
    def theme(self):
        """string or None : Preset color palette.
        """
        for theme, palette in self.themes.items():
            if palette == self.palette:
                return theme

    @theme.setter
    def theme(self, theme):
        if theme == self.theme:
            return

        try:
            self.palette = self.themes[theme]
        except KeyError:
            raise ValueError(
                f"Theme '{theme}' not found; "
                f"options are {list(self.themes)}."
            )

    @property
    def grid_size(self):
        """tuple: Size of grid
        """
        return self._grid_size

    @grid_size.setter
    def grid_size(self, grid_size):
        if np.all(self.grid_size == grid_size):
            return
        self._grid_size = grid_size
        self.reset_view()
        self.events.grid()

    @property
    def status(self):
        """string: Status string
        """
        return self._status

    @status.setter
    def status(self, status):
        if status == self.status:
            return
        self._status = status
        self.events.status(text=self._status)

    @property
    def help(self):
        """string: String that can be displayed to the
        user in the status bar with helpful usage tips.
        """
        return self._help

    @help.setter
    def help(self, help):
        if help == self.help:
            return
        self._help = help
        self.events.help(text=self._help)

    @property
    def title(self):
        """string: String that is displayed in window title.
        """
        return self._title

    @title.setter
    def title(self, title):
        if title == self.title:
            return
        self._title = title
        self.events.title(text=self._title)

    @property
    def interactive(self):
        """bool: Determines if canvas pan/zoom interactivity is enabled or not.
        """
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        if interactive == self.interactive:
            return
        self._interactive = interactive
        self.events.interactive()

    @property
    def cursor(self):
        """string: String identifying cursor displayed over canvas.
        """
        return self._cursor

    @cursor.setter
    def cursor(self, cursor):
        if cursor == self.cursor:
            return
        self._cursor = cursor
        self.events.cursor()

    @property
    def cursor_size(self):
        """int | None: Size of cursor if custom. None is yields default size
        """
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, cursor_size):
        if cursor_size == self.cursor_size:
            return
        self._cursor_size = cursor_size
        self.events.cursor()

    @property
    def active_layer(self):
        """int: index of active_layer
        """
        return self._active_layer

    @active_layer.setter
    def active_layer(self, active_layer):
        if active_layer == self.active_layer:
            return
        self._active_layer = active_layer
        self.events.active_layer(item=self._active_layer)

    def _scene_shape(self):
        """Get shape of currently viewed dimensions.

        Returns
        ----------
        centroid : list
            List of center coordinates of scene, length 2 or 3 if displayed
            view is 2D or 3D.
        size : list
            List of size of scene, length 2 or 3 if displayed view is 2D or 3D.
        corner : list
            List of coordinates of top left corner of scene, length 2 or 3 if
            displayed view is 2D or 3D.
        """
        # Scale the camera to the contents in the scene
        min_shape, max_shape = self._calc_bbox()
        size = np.subtract(max_shape, min_shape)
        size = [size[i] for i in self.dims.displayed]
        corner = [min_shape[i] for i in self.dims.displayed]

        return size, corner

    def reset_view(self):
        """Resets the camera's view using `event.rect` a 4-tuple of the x, y
        corner position followed by width and height of the camera
        """

        scene_size, corner = self._scene_shape()
        grid_size = list(self.grid_size)
        if len(scene_size) > len(grid_size):
            grid_size = [1] * (len(scene_size) - len(grid_size)) + grid_size
        size = np.multiply(scene_size, grid_size)
        centroid = np.add(corner, np.divide(size, 2))

        if self.dims.ndisplay == 2:
            # For a PanZoomCamera emit a 4-tuple of the rect
            corner = np.subtract(corner, np.multiply(0.05, size))[::-1]
            size = np.multiply(1.1, size)[::-1]
            rect = tuple(corner) + tuple(size)
            self.events.reset_view(rect=rect)
        else:
            # For an ArcballCamera emit the center and scale_factor
            center = centroid[::-1]
            scale_factor = 1.1 * np.max(size[-2:])
            # set initial camera angle so that it matches top layer of 2D view
            # when transitioning to 3D view
            quaternion = [np.pi / 2, 1, 0, 0]
            self.events.reset_view(
                center=center, scale_factor=scale_factor, quaternion=quaternion
            )

    def to_svg(self, file=None, view_box=None):
        """Convert the viewer state to an SVG. Non visible layers will be
        ignored.

        Parameters
        ----------
        file : path-like object, optional
            An object representing a file system path. A path-like object is
            either a str or bytes object representing a path, or an object
            implementing the `os.PathLike` protocol. If passed the svg will be
            written to this file
        view_box : 4-tuple, optional
            View box of SVG canvas to be generated specified as `min-x`,
            `min-y`, `width` and `height`. If not specified, calculated
            from the last two dimensions of the view.

        Returns
        ----------
        svg : string
            SVG representation of the currently viewed layers.
        """

        if view_box is None:
            min_shape, max_shape = self._calc_bbox()
            min_shape = min_shape[-2:]
            max_shape = max_shape[-2:]
            shape = np.subtract(max_shape, min_shape)
        else:
            shape = view_box[2:]
            min_shape = view_box[:2]

        props = {
            'xmlns': 'http://www.w3.org/2000/svg',
            'xmlns:xlink': 'http://www.w3.org/1999/xlink',
        }

        xml = Element(
            'svg',
            height=f'{shape[0]}',
            width=f'{shape[1]}',
            version='1.1',
            **props,
        )

        transform = f'translate({-min_shape[1]} {-min_shape[0]})'
        xml_transform = Element('g', transform=transform)

        for layer in self.layers:
            if layer.visible:
                xml_list = layer.to_xml_list()
                for x in xml_list:
                    xml_transform.append(x)
        xml.append(xml_transform)

        svg = (
            '<?xml version=\"1.0\" standalone=\"no\"?>\n'
            + '<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n'
            + '\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n'
            + tostring(xml, encoding='unicode', method='xml')
        )

        if file:
            # Save svg to file
            with open(file, 'w') as f:
                f.write(svg)

        return svg

    def add_layer(self, layer):
        """Add a layer to the viewer.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        layer.events.select.connect(self._update_active_layer)
        layer.events.deselect.connect(self._update_active_layer)
        layer.events.status.connect(self._update_status)
        layer.events.help.connect(self._update_help)
        layer.events.interactive.connect(self._update_interactive)
        layer.events.cursor.connect(self._update_cursor)
        layer.events.cursor_size.connect(self._update_cursor_size)
        layer.events.data.connect(self._on_layers_change)
        layer.dims.events.ndisplay.connect(self._on_layers_change)
        layer.dims.events.order.connect(self._on_layers_change)
        layer.dims.events.range.connect(self._on_layers_change)
        self.layers.append(layer)
        self._update_layers(layers=[layer])

        if len(self.layers) == 1:
            self.reset_view()

    def add_image(
        self,
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
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending=None,
        visible=True,
        path=None,
    ):
        """Add an image layer to the layers list.

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
            expanded and then each entry in the list is applied to each image.
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
        path : str or list of str
            Path or list of paths to image data.

        Returns
        -------
        layer : :class:`napari.layers.Image` or list
            The newly-created image layer or list of image layers.
        """
        if data is None and path is None:
            raise ValueError("One of either data or path must be provided")
        elif data is not None and path is not None:
            raise ValueError("Only one of data or path can be provided")
        elif data is None:
            data = io.magic_imread(path)

        if channel_axis is None:
            if colormap is None:
                colormap = 'gray'
            if blending is None:
                blending = 'translucent'
            layer = layers.Image(
                data,
                rgb=rgb,
                is_pyramid=is_pyramid,
                colormap=colormap,
                contrast_limits=contrast_limits,
                gamma=gamma,
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
            self.add_layer(layer)
            return layer
        else:
            if is_pyramid:
                n_channels = data[0].shape[channel_axis]
            else:
                n_channels = data.shape[channel_axis]

            name = ensure_iterable(name)

            if blending is None:
                blending = 'additive'

            if colormap is None:
                if n_channels < 3:
                    colormap = colormaps.MAGENTA_GREEN
                else:
                    colormap = itertools.cycle(colormaps.CYMRGB)
            else:
                colormap = ensure_iterable(colormap)

            # If one pair of clim values is passed then need to iterate them to
            # all layers.
            if contrast_limits is not None and not is_iterable(
                contrast_limits[0]
            ):
                contrast_limits = itertools.repeat(contrast_limits)
            else:
                contrast_limits = ensure_iterable(contrast_limits)

            gamma = ensure_iterable(gamma)

            layer_list = []
            zipped_args = zip(
                range(n_channels), colormap, contrast_limits, gamma, name
            )
            for i, cmap, clims, _gamma, name in zipped_args:
                if is_pyramid:
                    image = [
                        data[j].take(i, axis=channel_axis)
                        for j in range(len(data))
                    ]
                else:
                    image = data.take(i, axis=channel_axis)
                layer = layers.Image(
                    image,
                    rgb=rgb,
                    colormap=cmap,
                    contrast_limits=clims,
                    gamma=_gamma,
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
                self.add_layer(layer)
                layer_list.append(layer)
            return layer_list

    def add_points(
        self,
        data=None,
        *,
        symbol='o',
        size=10,
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
        """Add a points layer to the layers list.

        Parameters
        ----------
        data : array (N, D)
            Coordinates for N points in D dimensions.
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
        layer : :class:`napari.layers.Points`
            The newly-created points layer.

        Notes
        -----
        See vispy's marker visual docs for more details:
        http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual
        """
        if data is None:
            ndim = max(self.dims.ndim, 2)
            data = np.empty([0, ndim])

        layer = layers.Points(
            data=data,
            symbol=symbol,
            size=size,
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
        self.add_layer(layer)
        return layer

    def add_labels(
        self,
        data,
        *,
        is_pyramid=None,
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
        """Add a labels (or segmentation) layer to the layers list.

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
        layer : :class:`napari.layers.Labels`
            The newly-created labels layer.
        """
        layer = layers.Labels(
            data,
            is_pyramid=is_pyramid,
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
        self.add_layer(layer)
        return layer

    def add_shapes(
        self,
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
    ):
        """Add a shapes layer to the layers list.

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
        layer : :class:`napari.layers.Shapes`
            The newly-created shapes layer.
        """
        if data is None:
            ndim = max(self.dims.ndim, 2)
            data = np.empty((0, 0, ndim))

        layer = layers.Shapes(
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
        self.add_layer(layer)
        return layer

    def add_surface(
        self,
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
    ):
        """Add a surface layer to the layers list.

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

        Returns
        -------
        layer : :class:`napari.layers.Surface`
            The newly-created surface layer.
        """
        layer = layers.Surface(
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
        self.add_layer(layer)
        return layer

    def add_vectors(
        self,
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
        """Add a vectors layer to the layers list.

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
        layer : :class:`napari.layers.Vectors`
            The newly-created vectors layer.
        """
        layer = layers.Vectors(
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
        self.add_layer(layer)
        return layer

    def _new_labels(self):
        if self.dims.ndim == 0:
            dims = (512, 512)
        else:
            dims = self._calc_bbox()[1]
            dims = [np.ceil(d).astype('int') if d > 0 else 1 for d in dims]
            if len(dims) < 1:
                dims = (512, 512)
        empty_labels = np.zeros(dims, dtype=int)
        self.add_labels(empty_labels)

    def _update_layers(self, layers=None):
        """Updates the contained layers.

        Parameters
        ----------
        layers : list of napari.layers.Layer, optional
            List of layers to update. If none provided updates all.
        """
        layers = layers or self.layers

        for layer in layers:
            # adjust the order of the global dims based on the number of
            # dimensions that a layer has - for example a global order of
            # [2, 1, 0, 3] -> [0, 1] for a layer that only has two dimesnions
            # or -> [1, 0, 2] for a layer with three as that corresponds to
            # the relative order of the last two and three dimensions
            # respectively
            offset = self.dims.ndim - layer.dims.ndim
            order = np.array(self.dims.order)
            if offset <= 0:
                order = list(range(-offset)) + list(order - offset)
            else:
                order = list(order[order >= offset] - offset)
            layer.dims.order = order
            layer.dims.ndisplay = self.dims.ndisplay

            # Update the point values of the layers for the dimensions that
            # the layer has
            for axis in range(layer.dims.ndim):
                point = self.dims.point[axis + offset]
                layer.dims.set_point(axis, point)

    def _update_active_layer(self, event):
        """Set the active layer by iterating over the layers list and
        finding the first selected layer. If multiple layers are selected the
        iteration stops and the active layer is set to be None

        Parameters
        ----------
        event : Event
            No Event parameters are used
        """
        # iteration goes backwards to find top most selected layer if any
        # if multiple layers are selected sets the active layer to None

        active_layer = None
        for layer in self.layers:
            if active_layer is None and layer.selected:
                active_layer = layer
            elif active_layer is not None and layer.selected:
                active_layer = None
                break

        if active_layer is None:
            self.status = 'Ready'
            self.help = ''
            self.cursor = 'standard'
            self.interactive = True
            self.active_layer = None
        else:
            self.status = active_layer.status
            self.help = active_layer.help
            self.cursor = active_layer.cursor
            self.interactive = active_layer.interactive
            self.active_layer = active_layer

    def _on_layers_change(self, event):
        if len(self.layers) == 0:
            self.dims.ndim = 2
            self.dims.reset()
        else:
            layer_range = self._calc_layers_ranges()
            self.dims.ndim = len(layer_range)
            for i, r in enumerate(layer_range):
                self.dims.set_range(i, r)
        self.events.layers_change()

    def _calc_layers_ranges(self):
        """Calculates the range along each axis from all present layers.
        """

        ndims = self._calc_layers_num_dims()
        ranges = [(inf, -inf, inf)] * ndims

        for layer in self.layers:
            layer_range = layer.dims.range[::-1]
            ranges = [
                (min(a, b), max(c, d), min(e, f))
                for (a, c, e), (b, d, f) in itertools.zip_longest(
                    ranges, layer_range, fillvalue=(inf, -inf, inf)
                )
            ]

        return ranges[::-1]

    def _calc_bbox(self):
        """Calculates the bounding box of all displayed layers.
        This assumes that all layers are stacked.
        """

        min_shape = []
        max_shape = []
        for min, max, step in self._calc_layers_ranges():
            min_shape.append(min)
            max_shape.append(max)
        if len(min_shape) == 0:
            min_shape = [0] * self.dims.ndim
            max_shape = [1] * self.dims.ndim

        return min_shape, max_shape

    def _calc_layers_num_dims(self):
        """Calculates the number of maximum dimensions in the contained images.
        """
        max_dims = 0
        for layer in self.layers:
            dims = layer.ndim
            if dims > max_dims:
                max_dims = dims

        return max_dims

    def _update_status(self, event):
        """Set the viewer status with the `event.status` string."""
        self.status = event.status

    def _update_help(self, event):
        """Set the viewer help with the `event.help` string."""
        self.help = event.help

    def _update_interactive(self, event):
        """Set the viewer interactivity with the `event.interactive` bool."""
        self.interactive = event.interactive

    def _update_cursor(self, event):
        """Set the viewer cursor with the `event.cursor` string."""
        self.cursor = event.cursor

    def _update_cursor_size(self, event):
        """Set the viewer cursor_size with the `event.cursor_size` int."""
        self.cursor_size = event.cursor_size

    def grid_view(self, n_row=None, n_column=None, stride=1):
        """Arrange the current layers is a 2D grid.

        Default behaviour is to make a square 2D grid.

        Parameters
        ----------
        n_row : int, optional
            Number of rows in the grid.
        n_column : int, optional
            Number of column in the grid.
        stride : int, optional
            Number of layers to place in each grid square before moving on to
            the next square. The default ordering is to place the most visible
            layer in the top left corner of the grid. A negative stride will
            cause the order in which the layers are placed in the grid to be
            reversed.
        """
        n_grid_squares = np.ceil(len(self.layers) / abs(stride)).astype(int)
        if n_row is None and n_column is None:
            n_row = np.ceil(np.sqrt(n_grid_squares)).astype(int)
            n_column = n_row
        elif n_row is None:
            n_row = np.ceil(n_grid_squares / n_column).astype(int)
        elif n_column is None:
            n_column = np.ceil(n_grid_squares / n_row).astype(int)

        n_row = max(1, n_row)
        n_column = max(1, n_column)
        self.grid_size = (n_row, n_column)
        self.grid_stride = stride
        for i, layer in enumerate(self.layers):
            if stride > 0:
                adj_i = len(self.layers) - i - 1
            else:
                adj_i = i
            adj_i = adj_i // abs(stride)
            adj_i = adj_i % (n_row * n_column)
            i_row = adj_i // n_column
            i_column = adj_i % n_column
            self._subplot(layer, (i_row, i_column))

    def stack_view(self):
        """Arrange the current layers is a stack.
        """
        self.grid_view(n_row=1, n_column=1, stride=1)

    def _update_grid(self):
        """Update grid with current grid values.
        """
        self.grid_view(
            n_row=self.grid_size[0],
            n_column=self.grid_size[1],
            stride=self.grid_stride,
        )

    def _subplot(self, layer, position):
        """Shift a layer to a specified position in a 2D grid.

        Parameters
        ----------
        layer : napar.layers.Layer
            Layer that is to be moved.
        position : 2-tuple of int
            New position of layer in grid.
        size : 2-tuple of int
            Size of the grid that is being used.
        """
        scene_size, corner = self._scene_shape()
        translate_2d = np.multiply(scene_size[-2:], position)
        translate = [0] * layer.ndim
        translate[-2:] = translate_2d
        layer.translate_grid = translate
