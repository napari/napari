import numpy as np
from vispy.geometry import PolygonData

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
    thicknes : float
        thickness of lines and edges.
    """
    objects = ['lines', 'rectangles', 'ellipses', 'paths', 'polygons']
    types = ['face', 'edge']
    _ellipse_segments = 100

    def __init__(self, lines=None, rectangles=None, ellipses=None, paths=None,
                 polygons=None, thickness=1):

        self.thickness = thickness

        self.id = np.empty((0), dtype=int) # For N objects, array of shape ids
        self.vertices = np.empty((0, 2)) # Array of M vertices from all N objects
        self.index = np.empty((0), dtype=int) # Object index (0, ..., N-1) for each of M vertices
        self.boxes = np.empty((0, 9, 2)) # Bounding box + center point for each of N objects

        self._mesh_vertices = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_index = np.empty((0, 3), dtype=int) #Mx3 array of object indices, shape id, and types of vertices
        self._mesh_faces = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form a triangle
        self._mesh_faces_index = np.empty((0, 3), dtype=int) #Px3 array of object indices of faces, shape id, and types of vertices

        self._mesh_vertices_centers = np.empty((0, 2)) # Mx2 array of vertices of centers of lines, or vertices of faces
        self._mesh_vertices_offsets = np.empty((0, 2)) # Mx2 array of vertices of offsets of lines, or 0 for faces


        self._add_lines(lines)
        self._add_rectangles(rectangles)
        self._add_ellipses(ellipses)
        self._add_paths(paths)
        self._add_polygons(polygons)

    def add_shapes(self, lines=None, rectangles=None, ellipses=None, paths=None,
                   polygons=None, thickness=1):

        self._add_lines(lines)
        self._add_rectangles(rectangles)
        self._add_ellipses(ellipses)
        self._add_paths(paths)
        self._add_polygons(polygons)

    def set_shapes(self, lines=None, rectangles=None, ellipses=None, paths=None,
                   polygons=None, thickness=1):

        self.id = np.empty((0), dtype=int) # For N objects, array of shape ids
        self.vertices = np.empty((0, 2)) # Array of M vertices from all N objects
        self.index = np.empty((0), dtype=int) # Object index (0, ..., N-1) for each of M vertices
        self.boxes = np.empty((0, 9, 2)) # Bounding box + center point for each of N objects

        self._mesh_vertices = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_index = np.empty((0, 3), dtype=int) #Mx3 array of object indices, shape id, and types of vertices
        self._mesh_faces = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form a triangle
        self._mesh_faces_index = np.empty((0, 3), dtype=int) #Px3 array of object indices of faces, shape id, and types of vertices

        self._mesh_vertices_centers = np.empty((0, 2)) # Mx2 array of vertices of centers of lines, or vertices of faces
        self._mesh_vertices_offsets = np.empty((0, 2)) # Mx2 array of vertices of offsets of lines, or 0 for faces

        self._add_lines(lines)
        self._add_rectangles(rectangles)
        self._add_ellipses(ellipses)
        self._add_paths(paths)
        self._add_polygons(polygons)

    def _add_lines(self, lines):
        if lines is None:
            return
        self.id = np.append(self.id, np.repeat(0, len(lines)), axis=0)
        self.vertices = np.append(self.vertices, lines.reshape((-1, lines.shape[-1])), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.arange(0, 2*len(lines))//2
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in lines])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For lines just add edge
        for i in range(len(lines)):
            self._compute_meshes(lines[i], edge=True, thickness=self.thickness, index=[m+i, 0])

    def _add_rectangles(self, rectangles):
        if rectangles is None:
            return
        self.id = np.append(self.id, np.repeat(1, len(rectangles)), axis=0)
        r = np.array([self._expand_rectangle(x) for x in rectangles])
        self.vertices = np.append(self.vertices, r.reshape((-1, r.shape[-1])), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.arange(0, 4*len(rectangles))//4
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in rectangles])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For rectanges add four boundary lines and then two triangles for each
        for i in range(len(rectangles)):
            fill_faces = np.array([[0, 1, 2], [0, 2, 3]])
            self._compute_meshes(r[i], edge=True, fill=True, closed=True, thickness=self.thickness, index=[m+i, 1],
                                 fill_vertices=r[i], fill_faces=fill_faces)

    def _add_ellipses(self, ellipses):
        if ellipses is None:
            return
        self.id = np.append(self.id, np.repeat(3, len(ellipses)), axis=0)
        e = np.array([self._expand_ellipse(x) for x in ellipses])
        self.vertices = np.append(self.vertices, e.reshape((-1, e.shape[-1])), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.arange(0, 4*len(ellipses))//4
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in ellipses])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For ellipses build boundary vertices with num_segments
        for i in range(len(ellipses)):
            points = self._generate_ellipse(ellipses[i], self._ellipse_segments)
            fill_faces = np.array([[0, i+1, i+2] for i in range(self._ellipse_segments)])
            fill_faces[-1, 2] = 1
            self._compute_meshes(points[1:-1], edge=True, fill=True, closed=True, thickness=self.thickness, index=[m+i, 2],
                                 fill_vertices=points, fill_faces=fill_faces)

    def _add_paths(self, paths):
        if paths is None:
            return
        self.id = np.append(self.id, np.repeat(4, len(paths)), axis=0)
        self.vertices = np.append(self.vertices, np.concatenate(paths, axis=0), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.concatenate([np.repeat(i, len(paths[i])) for i in range(len(paths))])
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in paths])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For paths connect every vertex in each path
        for i in range(len(paths)):
            self._compute_meshes(paths[i], edge=True, thickness=self.thickness, index=[m+i, 3])

    def _add_polygons(self, polygons):
        if polygons is None:
            return
        self.id = np.append(self.id, np.repeat(5, len(polygons)), axis=0)
        self.vertices = np.append(self.vertices, np.concatenate(polygons, axis=0), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.concatenate([np.repeat(i, len(polygons[i])) for i in range(len(polygons))])
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in polygons])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For polygons connect every vertex in each polygon, including loop back to close
        for i in range(len(polygons)):
            self._compute_meshes(polygons[i], edge=True, fill=True, closed=True, thickness=self.thickness, index=[m+i, 4])

    def _expand_box(self, corners):
        min_val = [corners[:,0].min(axis=0), corners[:,1].min(axis=0)]
        max_val = [corners[:,0].max(axis=0), corners[:,1].max(axis=0)]
        tl = np.array([min_val[0], min_val[1]])
        tr = np.array([max_val[0], min_val[1]])
        br = np.array([max_val[0], max_val[1]])
        bl = np.array([min_val[0], max_val[1]])
        return np.array([tl, (tl+tr)/2, tr, (tr+br)/2, br, (br+bl)/2, bl, (bl+tl)/2, (tl+tr+br+bl)/4])

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
                vertices, faces = PolygonData(vertices=points).triangulate()
                self._append_meshes(vertices, faces.astype(np.uint32), index=index + [0])

def path_triangulate(path, closed=False, limit=5, bevel=False):
    if closed:
        full_path = np.concatenate(([path[-1]], path, [path[0]]),axis=0)
        normals = [segment_normal(full_path[i], full_path[i+1]) for i in range(len(path))]
        normals=np.array(normals)
        full_path = np.concatenate((path, [path[0]]),axis=0)
        full_normals = np.concatenate((normals, [normals[0]]),axis=0)
        miters = np.array([full_normals[i:i+2].mean(axis=0) for i in range(len(full_path))])
        miters = np.array([miters[i]/np.dot(miters[i], full_normals[i]) for i in range(len(full_path))])
    else:
        full_path = np.concatenate((path, [path[-2]]),axis=0)
        normals = [segment_normal(full_path[i], full_path[i+1]) for i in range(len(path))]
        normals[-1] = -normals[-1]
        normals=np.array(normals)
        full_path = path
        full_normals = np.concatenate(([normals[0]], normals),axis=0)
        miters = np.array([full_normals[i:i+2].mean(axis=0) for i in range(len(full_path))])
        miters = np.array([miters[i]/np.dot(miters[i], full_normals[i]) for i in range(len(full_path))])
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
                vertex_offsets.append(-flip*miters[i])
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
            vertex_offsets.append(-flip*miters[i])
            vertex_offsets.append(-offset)
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
    unit = normal/np.linalg.norm(normal)
    return unit
