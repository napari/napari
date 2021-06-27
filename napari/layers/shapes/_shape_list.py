import numpy as np

from ...utils.translations import trans
from ._mesh import Mesh
from ._shapes_constants import ShapeType, shape_classes
from ._shapes_models import Line, Path, Shape
from ._shapes_utils import inside_triangles, triangles_intersect_box


class ShapeList:
    """List of shapes class.

    Parameters
    ----------
    data : list
        List of Shape objects
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    shapes : (N, ) list
        Shape objects.
    data : (N, ) list of (M, D) array
        Data arrays for each shape.
    ndisplay : int
        Number of displayed dimensions.
    slice_keys : (N, 2, P) array
        Array of slice keys for each shape. Each slice key has the min and max
        values of the P non-displayed dimensions, useful for slicing
        multidimensional shapes. If the both min and max values of shape are
        equal then the shape is entirely contained within the slice specified
        by those values.
    shape_types : (N, ) list of str
        Name of shape type for each shape.
    edge_color : (N x 4) np.ndarray
        Array of RGBA edge colors for each shape.
    face_color : (N x 4) np.ndarray
        Array of RGBA face colors for each shape.
    edge_widths : (N, ) list of float
        Edge width for each shape.
    z_indices : (N, ) list of int
        z-index for each shape.

    Notes
    -----
    _vertices : np.ndarray
        Mx2 array of all displayed vertices from all shapes
    _index : np.ndarray
        Length M array with the index (0, ..., N-1) of each shape that each
        vertex corresponds to
    _z_index : np.ndarray
        Length N array with z_index of each shape
    _z_order : np.ndarray
        Length N array with z_order of each shape. This must be a permutation
        of (0, ..., N-1).
    _mesh : Mesh
        Mesh object containing all the mesh information that will ultimately
        be rendered.
    """

    def __init__(self, data=[], ndisplay=2):

        self._ndisplay = ndisplay
        self.shapes = []
        self._displayed = []
        self._slice_key = []
        self.displayed_vertices = []
        self.displayed_index = []
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)

        self._mesh = Mesh(ndisplay=self.ndisplay)

        self._edge_color = np.empty((0, 4))
        self._face_color = np.empty((0, 4))

        for d in data:
            self.add(d)

    @property
    def data(self):
        """list of (M, D) array: data arrays for each shape."""
        return [s.data for s in self.shapes]

    @property
    def ndisplay(self):
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self.ndisplay == ndisplay:
            return

        self._ndisplay = ndisplay
        self._mesh.ndisplay = self.ndisplay
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        for index in range(len(self.shapes)):
            shape = self.shapes[index]
            shape.ndisplay = self.ndisplay
            self.remove(index, renumber=False)
            self.add(shape, shape_index=index)
        self._update_z_order()

    @property
    def slice_keys(self):
        """(N, 2, P) array: slice key for each shape."""
        return np.array([s.slice_key for s in self.shapes])

    @property
    def shape_types(self):
        """list of str: shape types for each shape."""
        return [s.name for s in self.shapes]

    @property
    def edge_color(self):
        """(N x 4) np.ndarray: Array of RGBA edge colors for each shape"""
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._set_color(edge_color, 'edge')

    @property
    def face_color(self):
        """(N x 4) np.ndarray: Array of RGBA face colors for each shape"""
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._set_color(face_color, 'face')

    def _set_color(self, colors, attribute):
        """Set the face_color or edge_color property

        Parameters
        ----------
        colors : (N, 4) np.ndarray
            The value for setting edge or face_color. There must
            be one color for each shape
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        n_shapes = len(self.data)
        if not np.all(colors.shape == (n_shapes, 4)):
            raise ValueError(
                trans._(
                    '{attribute}_color must have shape ({n_shapes}, 4)',
                    deferred=True,
                    attribute=attribute,
                    n_shapes=n_shapes,
                )
            )

        update_method = getattr(self, f'update_{attribute}_color')

        for i, col in enumerate(colors):
            update_method(i, col, update=False)
        self._update_displayed()

    @property
    def edge_widths(self):
        """list of float: edge width for each shape."""
        return [s.edge_width for s in self.shapes]

    @property
    def z_indices(self):
        """list of int: z-index for each shape."""
        return [s.z_index for s in self.shapes]

    @property
    def slice_key(self):
        """list: slice key for slicing n-dimensional shapes."""
        return self._slice_key

    @slice_key.setter
    def slice_key(self, slice_key):
        slice_key = list(slice_key)
        if not np.all(self._slice_key == slice_key):
            self._slice_key = slice_key
            self._update_displayed()

    def _update_displayed(self):
        """Update the displayed data based on the slice key."""
        # The list slice key is repeated to check against both the min and
        # max values stored in the shapes slice key.
        slice_key = np.array([self.slice_key, self.slice_key])

        # Slice key must exactly match mins and maxs of shape as then the
        # shape is entirely contained within the current slice.
        if len(self.shapes) > 0:
            self._displayed = np.all(self.slice_keys == slice_key, axis=(1, 2))
        else:
            self._displayed = []
        disp_indices = np.where(self._displayed)[0]

        z_order = self._mesh.triangles_z_order
        disp_tri = np.isin(
            self._mesh.triangles_index[z_order, 0], disp_indices
        )
        self._mesh.displayed_triangles = self._mesh.triangles[z_order][
            disp_tri
        ]
        self._mesh.displayed_triangles_index = self._mesh.triangles_index[
            z_order
        ][disp_tri]
        self._mesh.displayed_triangles_colors = self._mesh.triangles_colors[
            z_order
        ][disp_tri]

        disp_vert = np.isin(self._index, disp_indices)
        self.displayed_vertices = self._vertices[disp_vert]
        self.displayed_index = self._index[disp_vert]

    def add(
        self,
        shape,
        face_color=None,
        edge_color=None,
        shape_index=None,
        z_refresh=True,
    ):
        """Adds a single Shape object

        Parameters
        ----------
        shape : subclass Shape
            Must be a subclass of Shape, one of "{'Line', 'Rectangle',
            'Ellipse', 'Path', 'Polygon'}"
        shape_index : None | int
            If int then edits the shape date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new shape to end of shapes list
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When shape_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of shapes, set to false  and then call
            ShapesList._update_z_order() once at the end.
        """
        if not issubclass(type(shape), Shape):
            raise ValueError(
                trans._(
                    'shape must be subclass of Shape',
                    deferred=True,
                )
            )

        if shape_index is None:
            shape_index = len(self.shapes)
            self.shapes.append(shape)
            self._z_index = np.append(self._z_index, shape.z_index)

            if face_color is None:
                face_color = np.array([1, 1, 1, 1])
            self._face_color = np.vstack([self._face_color, face_color])
            if edge_color is None:
                edge_color = np.array([0, 0, 0, 1])
            self._edge_color = np.vstack([self._edge_color, edge_color])
        else:
            z_refresh = False
            self.shapes[shape_index] = shape
            self._z_index[shape_index] = shape.z_index

            if face_color is None:
                face_color = self._face_color[shape_index]
            else:
                self._face_color[shape_index, :] = face_color
            if edge_color is None:
                edge_color = self._edge_color[shape_index]
            else:
                self._edge_color[shape_index, :] = edge_color

        self._vertices = np.append(
            self._vertices, shape.data_displayed, axis=0
        )
        index = np.repeat(shape_index, len(shape.data))
        self._index = np.append(self._index, index, axis=0)

        # Add faces to mesh
        m = len(self._mesh.vertices)
        vertices = shape._face_vertices
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = shape._face_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = np.zeros(shape._face_vertices.shape)
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[shape_index, 0]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = shape._face_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[shape_index, 0]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([face_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        # Add edges to mesh
        m = len(self._mesh.vertices)
        vertices = (
            shape._edge_vertices + shape.edge_width * shape._edge_offsets
        )
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = shape._edge_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = shape._edge_offsets
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[shape_index, 1]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = shape._edge_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[shape_index, 1]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([edge_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        if z_refresh:
            # Set z_order
            self._update_z_order()

    def remove_all(self):
        """Removes all shapes"""
        self.shapes = []
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)
        self._mesh.clear()
        self._update_displayed()

    def remove(self, index, renumber=True):
        """Removes a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be removed.
        renumber : bool
            Bool to indicate whether to renumber all shapes or not. If not the
            expectation is that this shape is being immediately added back to the
            list using `add_shape`.
        """
        indices = self._index != index
        self._vertices = self._vertices[indices]
        self._index = self._index[indices]

        # Remove triangles
        indices = self._mesh.triangles_index[:, 0] != index
        self._mesh.triangles = self._mesh.triangles[indices]
        self._mesh.triangles_colors = self._mesh.triangles_colors[indices]
        self._mesh.triangles_index = self._mesh.triangles_index[indices]

        # Remove vertices
        indices = self._mesh.vertices_index[:, 0] != index
        self._mesh.vertices = self._mesh.vertices[indices]
        self._mesh.vertices_centers = self._mesh.vertices_centers[indices]
        self._mesh.vertices_offsets = self._mesh.vertices_offsets[indices]
        self._mesh.vertices_index = self._mesh.vertices_index[indices]
        indices = np.where(np.invert(indices))[0]
        num_indices = len(indices)
        if num_indices > 0:
            indices = self._mesh.triangles > indices[0]
            self._mesh.triangles[indices] = (
                self._mesh.triangles[indices] - num_indices
            )

        if renumber:
            del self.shapes[index]
            indices = self._index > index
            self._index[indices] = self._index[indices] - 1
            self._z_index = np.delete(self._z_index, index)
            indices = self._mesh.triangles_index[:, 0] > index
            self._mesh.triangles_index[indices, 0] = (
                self._mesh.triangles_index[indices, 0] - 1
            )
            indices = self._mesh.vertices_index[:, 0] > index
            self._mesh.vertices_index[indices, 0] = (
                self._mesh.vertices_index[indices, 0] - 1
            )
            self._update_z_order()

    def _update_mesh_vertices(self, index, edge=False, face=False):
        """Updates the mesh vertex data and vertex data for a single shape
        located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge : bool
            Bool to indicate whether to update mesh vertices corresponding to
            edges
        face : bool
            Bool to indicate whether to update mesh vertices corresponding to
            faces and to update the underlying shape vertices
        """
        shape = self.shapes[index]
        if edge:
            indices = np.all(self._mesh.vertices_index == [index, 1], axis=1)
            self._mesh.vertices[indices] = (
                shape._edge_vertices + shape.edge_width * shape._edge_offsets
            )
            self._mesh.vertices_centers[indices] = shape._edge_vertices
            self._mesh.vertices_offsets[indices] = shape._edge_offsets
            self._update_displayed()

        if face:
            indices = np.all(self._mesh.vertices_index == [index, 0], axis=1)
            self._mesh.vertices[indices] = shape._face_vertices
            self._mesh.vertices_centers[indices] = shape._face_vertices
            indices = self._index == index
            self._vertices[indices] = shape.data_displayed
            self._update_displayed()

    def _update_z_order(self):
        """Updates the z order of the triangles given the z_index list"""
        self._z_order = np.argsort(self._z_index)
        if len(self._z_order) == 0:
            self._mesh.triangles_z_order = np.empty((0), dtype=int)
        else:
            _, idx, counts = np.unique(
                self._mesh.triangles_index[:, 0],
                return_index=True,
                return_counts=True,
            )
            triangles_z_order = [
                np.arange(idx[z], idx[z] + counts[z]) for z in self._z_order
            ]
            self._mesh.triangles_z_order = np.concatenate(triangles_z_order)
        self._update_displayed()

    def edit(
        self, index, data, face_color=None, edge_color=None, new_type=None
    ):
        """Updates the data of a single shape located at index. If
        `new_type` is not None then converts the shape type to the new type

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        data : np.ndarray
            NxD array of vertices.
        new_type : None | str | Shape
            If string , must be one of "{'line', 'rectangle', 'ellipse',
            'path', 'polygon'}".
        """
        if new_type is not None:
            cur_shape = self.shapes[index]
            if type(new_type) == str:
                shape_type = ShapeType(new_type)
                if shape_type in shape_classes.keys():
                    shape_cls = shape_classes[shape_type]
                else:
                    raise ValueError(
                        trans._(
                            '{shape_type} must be one of {shape_classes}',
                            deferred=True,
                            shape_type=shape_type,
                            shape_classes=set(shape_classes),
                        )
                    )
            else:
                shape_cls = new_type
            shape = shape_cls(
                data,
                edge_width=cur_shape.edge_width,
                z_index=cur_shape.z_index,
                dims_order=cur_shape.dims_order,
            )
        else:
            shape = self.shapes[index]
            shape.data = data

        if face_color is not None:
            self._face_color[index] = face_color
        if edge_color is not None:
            self._edge_color[index] = edge_color

        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)
        self._update_z_order()

    def update_edge_width(self, index, edge_width):
        """Updates the edge width of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge_width : float
            thickness of lines and edges.
        """
        self.shapes[index].edge_width = edge_width
        self._update_mesh_vertices(index, edge=True)

    def update_edge_color(self, index, edge_color, update=True):
        """Updates the edge color of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        update : bool
            If True, update the mesh with the new color property. Set to False to avoid
            repeated updates when modifying multiple shapes. Default is True.
        """
        self._edge_color[index] = edge_color
        indices = np.all(self._mesh.triangles_index == [index, 1], axis=1)
        self._mesh.triangles_colors[indices] = self._edge_color[index]
        if update:
            self._update_displayed()

    def update_face_color(self, index, face_color, update=True):
        """Updates the face color of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        face_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        update : bool
            If True, update the mesh with the new color property. Set to False to avoid
            repeated updates when modifying multiple shapes. Default is True.
        """
        self._face_color[index] = face_color
        indices = np.all(self._mesh.triangles_index == [index, 0], axis=1)
        self._mesh.triangles_colors[indices] = self._face_color[index]
        if update:
            self._update_displayed()

    def update_dims_order(self, dims_order):
        """Updates dimensions order for all shapes.

        Parameters
        ----------
        dims_order : (D,) list
            Order that the dimensions are rendered in.
        """
        for index in range(len(self.shapes)):
            if not self.shapes[index].dims_order == dims_order:
                shape = self.shapes[index]
                shape.dims_order = dims_order
                self.remove(index, renumber=False)
                self.add(shape, shape_index=index)
        self._update_z_order()

    def update_z_index(self, index, z_index):
        """Updates the z order of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        z_index : int
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others.
        """
        self.shapes[index].z_index = z_index
        self._z_index[index] = z_index
        self._update_z_order()

    def shift(self, index, shift):
        """Performs a 2D shift on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        """
        self.shapes[index].shift(shift)
        self._update_mesh_vertices(index, edge=True, face=True)

    def scale(self, index, scale, center=None):
        """Performs a scaling on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        self.shapes[index].scale(scale, center=center)
        shape = self.shapes[index]
        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)
        self._update_z_order()

    def rotate(self, index, angle, center=None):
        """Performs a rotation on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        angle : float
            angle specifying rotation of shape in degrees.
        center : list
            length 2 list specifying coordinate of center of rotation.
        """
        self.shapes[index].rotate(angle, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)

    def flip(self, index, axis, center=None):
        """Performs an vertical flip on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        axis : int
            integer specifying axis of flip. `0` flips horizontal, `1` flips
            vertical.
        center : list
            length 2 list specifying coordinate of center of flip axes.
        """
        self.shapes[index].flip(axis, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)

    def transform(self, index, transform):
        """Performs a linear transform on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self.shapes[index].transform(transform)
        shape = self.shapes[index]
        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)
        self._update_z_order()

    def outline(self, indices):
        """Finds outlines of shapes listed in indices

        Parameters
        ----------
        indices : int | list
            Location in list of the shapes to be outline. If list must be a
            list of int

        Returns
        -------
        centers : np.ndarray
            Nx2 array of centers of outline
        offsets : np.ndarray
            Nx2 array of offsets of outline
        triangles : np.ndarray
            Mx3 array of any indices of vertices for triangles of outline
        """
        if type(indices) is list:
            meshes = self._mesh.triangles_index
            triangle_indices = [
                i
                for i, x in enumerate(meshes)
                if x[0] in indices and x[1] == 1
            ]
            meshes = self._mesh.vertices_index
            vertices_indices = [
                i
                for i, x in enumerate(meshes)
                if x[0] in indices and x[1] == 1
            ]
        else:
            triangle_indices = np.all(
                self._mesh.triangles_index == [indices, 1], axis=1
            )
            triangle_indices = np.where(triangle_indices)[0]
            vertices_indices = np.all(
                self._mesh.vertices_index == [indices, 1], axis=1
            )
            vertices_indices = np.where(vertices_indices)[0]

        offsets = self._mesh.vertices_offsets[vertices_indices]
        centers = self._mesh.vertices_centers[vertices_indices]
        triangles = self._mesh.triangles[triangle_indices]

        if type(indices) is list:
            t_ind = self._mesh.triangles_index[triangle_indices][:, 0]
            inds = self._mesh.vertices_index[vertices_indices][:, 0]
            starts = np.unique(inds, return_index=True)[1]
            for i, ind in enumerate(indices):
                inds = t_ind == ind
                adjust_index = starts[i] - vertices_indices[starts[i]]
                triangles[inds] = triangles[inds] + adjust_index
        else:
            triangles = triangles - vertices_indices[0]

        return centers, offsets, triangles

    def shapes_in_box(self, corners):
        """Determines which shapes, if any, are inside an axis aligned box.

        Looks only at displayed shapes

        Parameters
        ----------
        corners : np.ndarray
            2x2 array of two corners that will be used to create an axis
            aligned box.

        Returns
        -------
        shapes : list
            List of shapes that are inside the box.
        """

        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        intersects = triangles_intersect_box(triangles, corners)
        shapes = self._mesh.displayed_triangles_index[intersects, 0]
        shapes = np.unique(shapes).tolist()

        return shapes

    def inside(self, coord):
        """Determines if any shape at given coord by looking inside triangle
        meshes. Looks only at displayed shapes

        Parameters
        ----------
        coord : sequence of float
            Image coordinates to check if any shapes are at.

        Returns
        -------
        shape : int | None
            Index of shape if any that is at the coordinates. Returns `None`
            if no shape is found.
        """
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        indices = inside_triangles(triangles - coord)
        shapes = self._mesh.displayed_triangles_index[indices, 0]

        if len(shapes) > 0:
            z_list = self._z_order.tolist()
            order_indices = np.array([z_list.index(m) for m in shapes])
            ordered_shapes = shapes[np.argsort(order_indices)]
            return ordered_shapes[0]
        else:
            return None

    def to_masks(self, mask_shape=None, zoom_factor=1, offset=[0, 0]):
        """Returns N binary masks, one for each shape, embedded in an array of
        shape `mask_shape`.

        Parameters
        ----------
        mask_shape : np.ndarray | tuple | None
            2-tuple defining shape of mask to be generated. If non specified,
            takes the max of all the vertiecs
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        masks : (N, M, P) np.ndarray
            Array where there is one binary mask of shape MxP for each of
            N shapes
        """
        if mask_shape is None:
            mask_shape = self.displayed_vertices.max(axis=0).astype('int')

        masks = np.array(
            [
                s.to_mask(mask_shape, zoom_factor=zoom_factor, offset=offset)
                for s in self.shapes
            ]
        )

        return masks

    def to_labels(self, labels_shape=None, zoom_factor=1, offset=[0, 0]):
        """Returns a integer labels image, where each shape is embedded in an
        array of shape labels_shape with the value of the index + 1
        corresponding to it, and 0 for background. For overlapping shapes
        z-ordering will be respected.

        Parameters
        ----------
        labels_shape : np.ndarray | tuple | None
            2-tuple defining shape of labels image to be generated. If non
            specified, takes the max of all the vertiecs
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        labels : np.ndarray
            MxP integer array where each value is either 0 for background or an
            integer up to N for points inside the corresponding shape.
        """
        if labels_shape is None:
            labels_shape = self.displayed_vertices.max(axis=0).astype(np.int)

        labels = np.zeros(labels_shape, dtype=int)

        for ind in self._z_order[::-1]:
            mask = self.shapes[ind].to_mask(
                labels_shape, zoom_factor=zoom_factor, offset=offset
            )
            labels[mask] = ind + 1

        return labels

    def to_colors(
        self, colors_shape=None, zoom_factor=1, offset=[0, 0], max_shapes=None
    ):
        """Rasterize shapes to an RGBA image array.

        Each shape is embedded in an array of shape `colors_shape` with the
        RGBA value of the shape, and 0 for background. For overlapping shapes
        z-ordering will be respected.

        Parameters
        ----------
        colors_shape : np.ndarray | tuple | None
            2-tuple defining shape of colors image to be generated. If non
            specified, takes the max of all the vertiecs
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.
        max_shapes : None | int
            If provided, this is the maximum number of shapes that will be rasterized.
            If the number of shapes in view exceeds max_shapes, max_shapes shapes
            will be randomly selected from the in view shapes. If set to None, no
            maximum is applied. The default value is None.

        Returns
        -------
        colors : (N, M, 4) array
            rgba array where each value is either 0 for background or the rgba
            value of the shape for points inside the corresponding shape.
        """
        if colors_shape is None:
            colors_shape = self.displayed_vertices.max(axis=0).astype(np.int)

        colors = np.zeros(tuple(colors_shape) + (4,), dtype=float)
        colors[..., 3] = 1

        z_order = self._z_order[::-1]
        shapes_in_view = np.argwhere(self._displayed)
        z_order_in_view_mask = np.isin(z_order, shapes_in_view)
        z_order_in_view = z_order[z_order_in_view_mask]

        # If there are too many shapes to render responsively, just render
        # the top max_shapes shapes
        if max_shapes is not None and len(z_order_in_view) > max_shapes:
            z_order_in_view = z_order_in_view[0:max_shapes]

        for ind in z_order_in_view:
            mask = self.shapes[ind].to_mask(
                colors_shape, zoom_factor=zoom_factor, offset=offset
            )
            if type(self.shapes[ind]) in [Path, Line]:
                col = self._edge_color[ind]
            else:
                col = self._face_color[ind]
            colors[mask, :] = col

        return colors
