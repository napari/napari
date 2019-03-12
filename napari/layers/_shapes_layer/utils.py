import numpy as np
from vispy.geometry import PolygonData
from copy import copy

class ShapesData():
    """Shapes class.
    Parameters
    ----------
    lines : np.ndarray
        Nx2x2 array of endpoints of lines.
    rectangles : np.ndarray
        Nx2x2 array of corners of rectangles.
    ellipses : np.ndarray
        Nx2x2 array of corners of ellipses.
    paths : list
        list of Nx2 arrays of points on each path.
    polygons : list
        list of Nx2 arrays of vertices of each polygon.
    thickness : float
        thickness of lines and edges.
    rotation : float
        any rotation to be applied to objects and bounding boxes in degrees.
    """
    objects = ['line', 'rectangle', 'ellipse', 'path', 'polygon']
    types = ['face', 'edge']
    _ellipse_segments = 100
    _rotion_handle_length = 20

    def __init__(self, lines=None, rectangles=None, ellipses=None, paths=None,
                 polygons=None, thickness=1, rotation=0):

        self.id = np.empty((0), dtype=int) # For N objects, array of shape ids
        self.vertices = np.empty((0, 2)) # Array of M vertices from all N objects
        self.index = np.empty((0), dtype=int) # Object index (0, ..., N-1) for each of M vertices
        self.boxes = np.empty((0, 10, 2)) # Bounding box + center point + rotation handle for each of N objects
        self.count = 0
        self._thickness = np.empty((0)) # For N objects, array of thicknesses
        self._rotation = np.empty((0)) # For N objects, array of rotation

        self._mesh_vertices = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_index = np.empty((0, 3), dtype=int) #Mx3 array of object indices, shape id, and types of vertices
        self._mesh_faces = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form a triangle
        self._mesh_faces_index = np.empty((0, 3), dtype=int) #Px3 array of object indices of faces, shape id, and types of vertices

        self._mesh_vertices_centers = np.empty((0, 2)) # Mx2 array of vertices of centers of lines, or vertices of faces
        self._mesh_vertices_offsets = np.empty((0, 2)) # Mx2 array of vertices of offsets of lines, or 0 for faces

        self.selected_box = None

        self.add_shapes(lines=lines, rectangles=rectangles, ellipses=ellipses,
                        paths=paths, polygons=polygons, thickness=thickness,
                        rotation=rotation)

    def add_shapes(self, lines=None, rectangles=None, ellipses=None, paths=None,
                   polygons=None, thickness=1, rotation=0):

        cur_shapes = len(self.id)
        new_shapes = cur_shapes
        if lines is not None:
            new_shapes = new_shapes + len(lines)
        if rectangles is not None:
            new_shapes = new_shapes + len(rectangles)
        if ellipses is not None:
            new_shapes = new_shapes + len(ellipses)
        if paths is not None:
            new_shapes = new_shapes + len(paths)
        if polygons is not None:
            new_shapes = new_shapes + len(polygons)

        # update thickness for all new shapes
        if np.isscalar(thickness):
            self._thickness = np.concatenate((self._thickness, np.repeat(thickness, new_shapes)), axis=0)
        elif isinstance(index, (list, np.ndarray)):
            if len(thickness) != new_shapes:
             raise TypeError('If thickness is a list/array, must be the same length as '\
                             'number of shapes')
            if isinstance(size, list):
                thickness = np.array(thickness)
            self._thickness = np.concatenate((self._thickness, thickness), axis=0)
        else:
            raise TypeError('thickness should be float or ndarray')

        if lines is not None:
            for i, shape in enumerate(lines):
                self._add_path(shape, thickness=self._thickness[cur_shapes+i])
        if rectangles is not None:
            for i, shape in enumerate(rectangles):
                self._add_rectangle(shape, thickness=self._thickness[cur_shapes+i])
        if ellipses is not None:
            for i, shape in enumerate(ellipses):
                self._add_ellipse(shape, thickness=self._thickness[cur_shapes+i])
        if paths is not None:
            for i, shape in enumerate(paths):
                self._add_path(shape, thickness=self._thickness[cur_shapes+i])
        if polygons is not None:
            for i, shape in enumerate(polygons):
                self._add_polygon(shape, thickness=self._thickness[cur_shapes+i])
        self.count = len(self.id)

        # set rotation for all new shapes
        # if np.isscalar(rotation):
        #     self._rotation = np.concatenate((self._rotation, np.repeat(rotation, new_shapes)), axis=0)
        # elif isinstance(size, Iterable):
        #     if len(rotation) != new_shapes
        #      raise TypeError('If rotation is a list/array, must be the same length as '\
        #                      'number of shapes')
        #     if isinstance(size, list):
        #         rotation = np.asarray(rotation)
        #     self._rotation = np.concatenate((self._rotation, rotation), axis=0)
        # else:
        #     raise TypeError('rotation should be float or ndarray')
        #
        # self.update_rotation(list(range(cur_shapes, new_shapes)))

    def set_shapes(self, lines=None, rectangles=None, ellipses=None, paths=None,
                   polygons=None, thickness=1, rotation=0):

        self.remove_all_shapes()
        self.add_shapes(lines=lines, rectangles=rectangles, ellipses=ellipses,
                        paths=paths, polygons=polygons, thickness=thickness,
                        rotation=rotation)

    def remove_all_shapes(self):
        self.id = np.empty((0), dtype=int) # For N objects, array of shape ids
        self.vertices = np.empty((0, 2)) # Array of M vertices from all N objects
        self.index = np.empty((0), dtype=int) # Object index (0, ..., N-1) for each of M vertices
        self.boxes = np.empty((0, 10, 2)) # Bounding box + center point + rotation handle for each of N objects
        self.count = 0
        self._thickness = np.empty((0)) # For N objects, array of thicknesses
        self._rotation = np.empty((0)) # For N objects, array of rotation

        self._mesh_vertices = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_index = np.empty((0, 3), dtype=int) #Mx3 array of object indices, shape id, and types of vertices
        self._mesh_faces = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form a triangle
        self._mesh_faces_index = np.empty((0, 3), dtype=int) #Px3 array of object indices of faces, shape id, and types of vertices

        self._mesh_vertices_centers = np.empty((0, 2)) # Mx2 array of vertices of centers of lines, or vertices of faces
        self._mesh_vertices_offsets = np.empty((0, 2)) #

        self.selected_box = None

    def remove_one_shape(self, index, renumber=True):
        self.vertices = self.vertices[self.index!=index]
        self.index = self.index[self.index!=index]
        if renumber:
            self.index[self.index>index] = self.index[self.index>index]-1
            self.id = np.delete(self.id, index, axis=0)
            self.boxes = np.delete(self.boxes, index, axis=0)
            self._thickness = np.delete(self._thickness, index, axis=0)
            self.count = self.count - 1

        indices = self._select_meshes(index, self._mesh_faces_index)
        self._mesh_faces_index = np.delete(self._mesh_faces_index, indices, axis=0)
        self._mesh_faces = np.delete(self._mesh_faces, indices, axis=0)

        indices = self._select_meshes(index, self._mesh_vertices_index)
        self._mesh_vertices_index = np.delete(self._mesh_vertices_index, indices, axis=0)
        self._mesh_vertices = np.delete(self._mesh_vertices, indices, axis=0)
        self._mesh_vertices_centers = np.delete(self._mesh_vertices_centers, indices, axis=0)
        self._mesh_vertices_offsets = np.delete(self._mesh_vertices_offsets, indices, axis=0)
        if renumber:
            self._mesh_faces_index[self._mesh_faces_index[:,0]>index, 0] = self._mesh_faces_index[self._mesh_faces_index[:,0]>index, 0]-1
            self._mesh_vertices_index[self._mesh_vertices_index[:,0]>index, 0] = self._mesh_vertices_index[self._mesh_vertices_index[:,0]>index, 0]-1
        self._mesh_faces[self._mesh_faces>indices[0]] = self._mesh_faces[self._mesh_faces>indices[0]] - len(indices)

    def update_thickness(self, index):
        if index is True:
            for i in range(len(self._thickness)):
                indices = self._select_meshes(i, self._mesh_vertices_index, 1)
                self._mesh_vertices[indices] = (self._mesh_vertices_centers[indices]
                                                + self._thickness[i]*self._mesh_vertices_offsets[indices])
        elif isinstance(index, (list, np.ndarray)):
            for i in index:
                indices = self._select_meshes(i, self._mesh_vertices_index, 1)
                self._mesh_vertices[indices] = (self._mesh_vertices_centers[indices]
                                                + self._thickness[i]*self._mesh_vertices_offsets[indices])
        else:
            indices = self._select_meshes(index, self._mesh_vertices_index, 1)
            self._mesh_vertices[indices] = (self._mesh_vertices_centers[indices]
                                            + self._thickness[index]*self._mesh_vertices_offsets[indices])

    def scale_shapes(self, scale, center=None, index=True):
        """Perfroms a scaling on selected shapes
        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shapes in 2D.
        center : list
            length 2 list specifying coordinate of center of rotation.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        if isinstance(scale, (list, np.ndarray)):
            transform = np.array([[scale[0], 0], [0, scale[1]]])
        else:
            transform = np.array([[scale, 0], [0, scale]])
        if center is None:
            self.transform_shapes(transform, index=index)
        else:
            self.shift_shapes(-center, index=index)
            self.transform_shapes(transform, index=index)
            self.shift_shapes(center, index=index)

    def flip_vertical_shapes(self, center=None, index=True):
        """Perfroms an vertical flip on selected shapes
        Parameters
        ----------
        center : list
            length 2 list specifying coordinate of center of flip axes.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        if center is None:
            transform = np.array([[-1, 0], [0, 1]])
            self.transform_shapes(transform, index=index)
        else:
            self.shift_shapes(-center, index=index)
            transform = np.array([[-1, 0], [0, 1]])
            self.transform_shapes(transform, index=index)
            self.shift_shapes(-center, index=index)

    def flip_horizontal_shapes(self, center=None, index=True):
        """Perfroms an horizontal flip on selected shapes
        Parameters
        ----------
        center : list
            length 2 list specifying coordinate of center of flip axes.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        if center is None:
            transform = np.array([[1, 0], [0, -1]])
            self.transform_shapes(transform, index=index)
        else:
            self.shift_shapes(-center, index=index)
            transform = np.array([[1, 0], [0, -1]])
            self.transform_shapes(transform, index=index)
            self.shift_shapes(-center, index=index)

    def rotate_shapes(self, angle, center=None, index=True):
        """Perfroms a rotation on selected shapes
        Parameters
        ----------
        angle : float
            angle specifying rotation of shapes in degrees.
        center : list
            length 2 list specifying coordinate of center of rotation.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        theta = np.radians(angle)
        if center is None:
            transform = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            self.transform_shapes(transform, index=index)
        else:
            self.shift_shapes(-center, index=index)
            transform = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            self.transform_shapes(transform, index=index)
            self.shift_shapes(center, index=index)

    def shift_shapes(self, shift, index=True):
        """Perfroms an 2D shift on selected shapes
        Parameters
        ----------
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        shift = np.array(shift)

        indices = self._select_meshes(index, self._mesh_vertices_index)
        self._mesh_vertices[indices] = self._mesh_vertices[indices] + shift
        self._mesh_vertices_centers[indices] = self._mesh_vertices_centers[indices] + shift

        self.boxes[index] = self.boxes[index] + shift
        if self.selected_box is not None:
            self.selected_box = self.selected_box + shift

        indices = np.where(np.isin(self.index, index))[0]
        self.vertices[indices] = self.vertices[indices] + shift

    def transform_shapes(self, transform, index=True):
        """Perfroms an affine transform on selected shapes
        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        A = transform.T
        indices = self._select_meshes(index, self._mesh_vertices_index)

        self._mesh_vertices[indices] = np.matmul(self._mesh_vertices[indices], A)
        self._mesh_vertices_centers[indices] = np.matmul(self._mesh_vertices_centers[indices], A)
        x = self._mesh_vertices_offsets[indices]
        original_norms = np.expand_dims(np.linalg.norm(x, axis=1), axis=1)
        offsets = np.matmul(x, A)
        transform_norms = np.expand_dims(np.linalg.norm(offsets, axis=1), axis=1)
        transform_norms[transform_norms==0]=1
        self._mesh_vertices_offsets[indices] = offsets/transform_norms*original_norms

        boxes = np.matmul(self.boxes[index], A)
        if type(index) is list:
            boxes[:, 9] = boxes[:, 1] + (boxes[:, 9]-boxes[:, 1])/np.expand_dims(np.linalg.norm(boxes[:,9]-boxes[:,1],axis=1), axis=1)*self._rotion_handle_length
        else:
            boxes[9] = boxes[1] + (boxes[9]-boxes[1])/np.linalg.norm(boxes[9]-boxes[1])*self._rotion_handle_length
        self.boxes[index] = boxes

        if self.selected_box is not None:
            boxes = np.matmul(self.selected_box, A)
            boxes[9] = boxes[1] + (boxes[9]-boxes[1])/np.linalg.norm(boxes[9]-boxes[1])*self._rotion_handle_length
            self.selected_box = boxes

        indices = np.where(np.isin(self.index, index))[0]
        self.vertices[indices] = np.matmul(self.vertices[indices], A)

        self.update_thickness(index)

    def _select_meshes(self, index, meshes, object_type=None):
        if object_type is None:
            if index is True:
                indices = [i for i in range(len(meshes))]
            elif isinstance(index, (list, np.ndarray)):
                indices = [i for i, x in enumerate(meshes) if x[0] in index]
            elif np.isscalar(index):
                indices = meshes[:,0] == index
                indices = np.where(indices)[0]
            else:
                indices = []
        else:
            if index is True:
                indices = meshes[:,2]==object_type
            elif isinstance(index, (list, np.ndarray)):
                indices = [i for i, x in enumerate(meshes) if x[0] in index and x[2]==object_type]
            elif np.isscalar(index):
                index = np.broadcast_to([index, object_type], (len(meshes), 2))
                indices = np.all(np.equal(meshes[:,[0, 2]], index), axis=1)
                indices = np.where(indices)[0]
            else:
                indices = []
        return indices

    def _add_rectangle(self, shape, index=None, thickness=1):
        object_id = self.objects.index('rectangle')
        box = self._expand_box(shape)
        if index is None:
            m = len(self.id)
            self.id = np.append(self.id, [object_id], axis=0)
            self.boxes = np.append(self.boxes, [box], axis=0)
        else:
            m = index
            self.id[m] = object_id
            self.boxes[m] = box
        rectangle = self._expand_rectangle(shape)
        self.vertices = np.append(self.vertices, rectangle, axis=0)
        indices = np.repeat(m, len(rectangle))
        self.index = np.append(self.index, indices, axis=0)
        # Build objects to be rendered
        # For rectanges add four boundary lines and then two triangles for each
        fill_faces = np.array([[0, 1, 2], [0, 2, 3]])
        self._compute_meshes(rectangle, edge=True, fill=True, closed=True, thickness=thickness, index=[m, object_id],
                             fill_vertices=rectangle, fill_faces=fill_faces)

    def _add_ellipse(self, shape, index=None, thickness=1):
        object_id = self.objects.index('ellipse')
        box = self._expand_box(shape)
        if index is None:
            m = len(self.id)
            self.id = np.append(self.id, [object_id], axis=0)
            self.boxes = np.append(self.boxes, [box], axis=0)
        else:
            m = index
            self.id[m] = object_id
            self.boxes[m] = box
        if len(shape) == 2:
            ellipse = self._expand_ellipse(shape)
        else:
            ellipse = shape
        self.vertices = np.append(self.vertices, ellipse, axis=0)
        indices = np.repeat(m, len(ellipse))
        self.index = np.append(self.index, indices, axis=0)
        # Build objects to be rendered
        # For ellipses build boundary vertices with num_segments
        points = self._generate_ellipse(shape, self._ellipse_segments)
        fill_faces = np.array([[0, i+1, i+2] for i in range(self._ellipse_segments)])
        fill_faces[-1, 2] = 1
        self._compute_meshes(points[1:-1], edge=True, fill=True, closed=True, thickness=thickness, index=[m, object_id],
                             fill_vertices=points, fill_faces=fill_faces)

    def _add_path(self, shape, index=None, thickness=1):
        object_id = self.objects.index('path')
        box = self._expand_box(shape)
        if index is None:
            m = len(self.id)
            self.id = np.append(self.id, [object_id], axis=0)
            self.boxes = np.append(self.boxes, [box], axis=0)
        else:
            m = index
            self.id[m] = object_id
            self.boxes[m] = box
        self.vertices = np.append(self.vertices, shape, axis=0)
        indices = np.repeat(m, len(shape))
        self.index = np.append(self.index, indices, axis=0)
        # Build objects to be rendered
        # For paths connect every vertex in each path
        self._compute_meshes(shape, edge=True, thickness=thickness, index=[m, object_id])

    def _add_polygon(self, shape, index=None, thickness=1):
        object_id = self.objects.index('polygon')
        box = self._expand_box(shape)
        if index is None:
            m = len(self.id)
            self.id = np.append(self.id, [object_id], axis=0)
            self.boxes = np.append(self.boxes, [box], axis=0)
        else:
            m = index
            self.id[m] = object_id
            self.boxes[m] = box
        self.vertices = np.append(self.vertices, shape, axis=0)
        indices = np.repeat(m, len(shape))
        self.index = np.append(self.index, indices, axis=0)
        # Build objects to be rendered
        # For polygons connect every vertex in each polygon, including loop back to close
        self._compute_meshes(shape, edge=True, fill=True, closed=True, thickness=thickness, index=[m, object_id])

    def _expand_box(self, corners):
        min_val = [corners[:,0].min(axis=0), corners[:,1].min(axis=0)]
        max_val = [corners[:,0].max(axis=0), corners[:,1].max(axis=0)]
        tl = np.array([min_val[0], min_val[1]])
        tr = np.array([max_val[0], min_val[1]])
        br = np.array([max_val[0], max_val[1]])
        bl = np.array([min_val[0], max_val[1]])
        rot = (tl+tr)/2
        rot[1] = rot[1]-self._rotion_handle_length
        return np.array([tl, (tl+tr)/2, tr, (tr+br)/2, br, (br+bl)/2, bl, (bl+tl)/2, (tl+tr+br+bl)/4, rot])

    def _expand_rectangle(self, corners):
        tl = np.array([min(corners[0][0],corners[1][0]), min(corners[0][1],corners[1][1])])
        tr = np.array([max(corners[0][0],corners[1][0]), min(corners[0][1],corners[1][1])])
        br = np.array([max(corners[0][0],corners[1][0]), max(corners[0][1],corners[1][1])])
        bl = np.array([min(corners[0][0],corners[1][0]), max(corners[0][1],corners[1][1])])
        return np.array([tl, tr, br, bl])

    def _expand_ellipse(self, corners):
        tl = np.array([min(corners[0][0],corners[1][0]), min(corners[0][1],corners[1][1])])
        tr = np.array([max(corners[0][0],corners[1][0]), min(corners[0][1],corners[1][1])])
        br = np.array([max(corners[0][0],corners[1][0]), max(corners[0][1],corners[1][1])])
        bl = np.array([min(corners[0][0],corners[1][0]), max(corners[0][1],corners[1][1])])
        return np.array([(tl+tr)/2, (tr+br)/2, (br+bl)/2, (bl+tl)/2])

    def _generate_ellipse(self, corners, num_segments):
        center = corners.mean(axis=0)
        xr = abs(corners[0][0]-center[0])
        yr = abs(corners[0][1]-center[1])

        vertices = np.empty((num_segments + 1, 2), dtype=np.float32)
        theta = np.linspace(0, np.deg2rad(360), num_segments)

        vertices[1:, 0] = center[0] + xr * np.cos(theta)
        vertices[1:, 1] = center[1] + yr * np.sin(theta)

        # set center point to first vertex
        vertices[0] = np.float32([center[0], center[1]])
        return vertices

    def select_box(self, index=True):
        if index is True:
            box = self._expand_box(self.vertices)
        elif isinstance(index, (list, np.ndarray)):
            if len(index) == 0:
                box = None
            elif len(index) == 1:
                box = copy(self.boxes[index[0]])
            else:
                box = self._expand_box(self.vertices[np.isin(self.index, index)])
        else:
            box = copy(self.boxes[index])
        self.selected_box = box

    def _append_meshes(self, vertices, faces, index=[0, 0, 0],
                       centers=None, offsets=None):
        m = len(self._mesh_vertices)
        vertices_indices = np.repeat([index], len(vertices), axis=0)
        faces_indices = np.repeat([index], len(faces), axis=0)
        if centers is None and offsets is None:
            centers = vertices
            offsets = np.zeros((len(vertices),2))
        self._mesh_vertices = np.append(self._mesh_vertices, vertices, axis=0)
        self._mesh_vertices_index = np.append(self._mesh_vertices_index, vertices_indices, axis=0)
        self._mesh_vertices_centers = np.append(self._mesh_vertices_centers, centers, axis=0)
        self._mesh_vertices_offsets = np.append(self._mesh_vertices_offsets, offsets, axis=0)
        self._mesh_faces = np.append(self._mesh_faces, m+faces, axis=0)
        self._mesh_faces_index = np.append(self._mesh_faces_index, faces_indices, axis=0)

    def _compute_meshes(self, points, closed=False, fill=False, edge=False, thickness=1, index=[0, 0],
                        fill_vertices=None, fill_faces=None):
        if edge:
            centers, offsets, faces = path_triangulate(points, closed=closed)
            vertices = centers+thickness*offsets
            self._append_meshes(vertices, faces, index=index + [1],
                                centers=centers, offsets=offsets)
        if fill:
            if fill_vertices is not None and fill_faces is not None:
                self._append_meshes(fill_vertices, fill_faces, index=index + [0])
            else:
                if len(points) > 2:
                    vertices, faces = PolygonData(vertices=points).triangulate()
                    if len(faces) > 0:
                        self._append_meshes(vertices, faces.astype(np.uint32), index=index + [0])

def path_triangulate(path, closed=False, limit=5, bevel=False):
    if closed:
        full_path = np.concatenate(([path[-1]], path, [path[0]]),axis=0)
        normals = [segment_normal(full_path[i], full_path[i+1]) for i in range(len(path))]
        path_length = [np.linalg.norm(full_path[i]-full_path[i+1]) for i in range(len(path))]
        normals=np.array(normals)
        full_path = np.concatenate((path, [path[0]]),axis=0)
        full_normals = np.concatenate((normals, [normals[0]]),axis=0)
    else:
        full_path = np.concatenate((path, [path[-2]]),axis=0)
        normals = [segment_normal(full_path[i], full_path[i+1]) for i in range(len(path))]
        path_length = [np.linalg.norm(full_path[i]-full_path[i+1]) for i in range(len(path))]
        normals[-1] = -normals[-1]
        normals=np.array(normals)
        full_path = path
        full_normals = np.concatenate(([normals[0]], normals),axis=0)

    miters = np.array([full_normals[i:i+2].mean(axis=0) for i in range(len(full_path))])
    miters = np.array([miters[i]/np.dot(miters[i], full_normals[i])
                      if np.dot(miters[i], full_normals[i]) != 0 else full_normals[i]
                      for i in range(len(full_path))])
    miter_lengths = np.linalg.norm(miters,axis=1)
    miters = 0.5*miters
    vertex_offsets = []
    central_path = []
    faces = []
    m = 0

    for i in range(len(full_path)):
        if i==0:
            if (bevel or miter_lengths[i]>limit) and closed:
                offset = np.array([miters[i,1], -miters[i,0]])
                offset = 0.5*offset/np.linalg.norm(offset)
                flip = np.sign(np.dot(offset, full_normals[i]))
                vertex_offsets.append(offset)
                vertex_offsets.append(-flip*miters[i]/miter_lengths[i]*limit)
                vertex_offsets.append(-offset)
                central_path.append(full_path[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
                faces.append([0, 1, 2])
                m=m+1
            else:
                vertex_offsets.append(-miters[i])
                vertex_offsets.append(miters[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
        elif i==len(full_path)-1:
            if closed:
                a = vertex_offsets[m+1] - full_path[i-1]
                b = vertex_offsets[1] - full_path[i-1]
                ray = full_path[i] - full_path[i-1]
                if np.cross(a,ray)*np.cross(b,ray)>0:
                    faces.append([m, m+1, 1])
                    faces.append([m, 0, 1])
                else:
                    faces.append([m, m+1, 1])
                    faces.append([m+1, 0, 1])
            else:
                vertex_offsets.append(-miters[i])
                vertex_offsets.append(miters[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
                a = vertex_offsets[m+1] - full_path[i-1]
                b = vertex_offsets[m+3] - full_path[i-1]
                ray = full_path[i] - full_path[i-1]
                if np.cross(a,ray)*np.cross(b,ray)>0:
                    faces.append([m, m+1, m+3])
                    faces.append([m, m+2, m+3])
                else:
                    faces.append([m, m+1, m+3])
                    faces.append([m+1, m+2, m+3])
        elif (bevel or miter_lengths[i]>limit):
            offset = np.array([miters[i,1], -miters[i,0]])
            offset = 0.5*offset/np.linalg.norm(offset)
            flip = np.sign(np.dot(offset, full_normals[i]))
            vertex_offsets.append(offset)
            vertex_offsets.append(-flip*miters[i]/miter_lengths[i]*limit)
            vertex_offsets.append(-offset)
            central_path.append(full_path[i])
            central_path.append(full_path[i])
            central_path.append(full_path[i])
            a = vertex_offsets[m+1] - full_path[i-1]
            b = vertex_offsets[m+3] - full_path[i-1]
            ray = full_path[i] - full_path[i-1]
            if np.cross(a,ray)*np.cross(b,ray)>0:
                faces.append([m, m+1, m+3])
                faces.append([m, m+2, m+3])
            else:
                faces.append([m, m+1, m+3])
                faces.append([m+1, m+2, m+3])
            faces.append([m+2, m+3, m+4])
            m = m + 3
        else:
            vertex_offsets.append(-miters[i])
            vertex_offsets.append(miters[i])
            central_path.append(full_path[i])
            central_path.append(full_path[i])
            a = vertex_offsets[m+1] - full_path[i-1]
            b = vertex_offsets[m+3] - full_path[i-1]
            ray = full_path[i] - full_path[i-1]
            if np.cross(a,ray)*np.cross(b,ray)>0:
                faces.append([m, m+1, m+3])
                faces.append([m, m+2, m+3])
            else:
                faces.append([m, m+1, m+3])
                faces.append([m+1, m+2, m+3])
            m = m + 2
    return np.array(central_path), np.array(vertex_offsets), np.array(faces)

def segment_normal(a, b):
    d = b-a
    normal = np.array([d[1], -d[0]])
    norm = np.linalg.norm(normal)
    if norm==0:
        return np.array([1, 0])
    else:
        return normal/norm
