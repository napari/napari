import numpy as np
from vispy.geometry import PolygonData

class ShapesData():
    """Shapes class.
    Parameters
    ----------
    points : np.ndarray
        Nx2 array of vertices.
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
    """
    objects = ['points', 'lines', 'rectangles', 'ellipses', 'paths', 'polygons']
    _ellipse_segments = 100

    def __init__(self, points=None, lines=None, rectangles=None, ellipses=None,
                 paths=None, polygons=None):

        self.id = np.empty((0), dtype=int) # For N objects, array of object ids
        self.vertices = np.empty((0, 2)) # Array of M vertices from all N objects
        self.index = np.empty((0), dtype=int) # Object index (0, ..., N-1) for each of M vertices
        self.boxes = np.empty((0, 9, 2)) # Bounding box + center point for each of N objects

        self._points_vertices = np.empty((0, 2)) # Mx2 array of vertices to be rendered as markers
        self._points_vertices_index = np.empty((0), dtype=int) # Mx1 array of object indices of vertices

        self._lines_vertices = np.empty((0, 2)) # Mx2 array of vertices to be rendered as lines
        self._lines_vertices_index = np.empty((0), dtype=int) #Mx1 array of object indices of vertices
        self._lines_connect = np.empty((0, 2), dtype=np.uint32) # Px2 array of vertex indices to be connected
        self._lines_connect_index = np.empty((0), dtype=int) #Px1 array of object indices of connections

        self._triangles_vertices = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._triangles_vertices_index = np.empty((0), dtype=int) #Mx1 array of object indices of vertices
        self._triangles_faces = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form a triangle
        self._triangles_faces_index = np.empty((0), dtype=int) #Px1 array of object indices of faces

        if points is not None:
            self._add_points(points)

        if lines is not None:
            self._add_lines(lines)

        if rectangles is not None:
            self._add_rectangles(rectangles)

        if ellipses is not None:
            self._add_ellipses(ellipses)

        if paths is not None:
            self._add_paths(paths)

        if polygons is not None:
            self._add_polygons(polygons)

    def _add_points(self, points):
        self.id = np.append(self.id, np.repeat(0, len(points)), axis=0)
        self.vertices = np.append(self.vertices, points, axis=0)
        m = max(self.index, default=-1) + 1
        indices = np.arange(m, m+len(points))
        self.index = np.append(self.index, indices, axis=0)
        self.boxes = np.append(self.boxes, np.stack([points]*9, axis=1), axis=0)
        # Build objects to be rendered
        # For points just render vertices as is
        self._points_vertices = np.append(self._points_vertices, points, axis=0)
        self._points_vertices_index = np.append(self._points_vertices_index, indices, axis=0)

    def _add_lines(self, lines):
        self.id = np.append(self.id, np.repeat(1, len(lines)), axis=0)
        self.vertices = np.append(self.vertices, lines.reshape((-1, lines.shape[-1])), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.arange(0, 2*len(lines))//2
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in lines])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For lines just add lines
        m = len(self._lines_vertices)
        connect = [[m+2*i, m+2*i+1] for i in range(len(lines))]
        self._lines_vertices = np.append(self._lines_vertices, lines.reshape((-1, lines.shape[-1])), axis=0)
        self._lines_vertices_index = np.append(self._lines_vertices_index, indices, axis=0)
        self._lines_connect = np.append(self._lines_connect, connect, axis=0)
        self._lines_connect_index = np.append(self._lines_connect_index, indices[::2], axis=0)

    def _add_rectangles(self, rectangles):
        self.id = np.append(self.id, np.repeat(2, len(rectangles)), axis=0)
        r = np.array([self._expand_rectangle(x) for x in rectangles])
        self.vertices = np.append(self.vertices, r.reshape((-1, r.shape[-1])), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.arange(0, 4*len(rectangles))//4
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in rectangles])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For rectanges just add four boundary lines for each
        m = len(self._lines_vertices)
        connect = m + np.array([self._box_connect(i) for i in range(4*len(rectangles))])
        self._lines_vertices = np.append(self._lines_vertices, r.reshape((-1, r.shape[-1])), axis=0)
        self._lines_vertices_index = np.append(self._lines_vertices_index, indices, axis=0)
        self._lines_connect = np.append(self._lines_connect, connect, axis=0)
        self._lines_connect_index = np.append(self._lines_connect_index, indices[::4], axis=0)
        # Add two triangle faces for each rectangle
        m = len(self._triangles_vertices)
        faces = m + np.array([self._box_face(i) for i in range(2*len(rectangles))]).astype(np.uint32)
        self._triangles_vertices = np.append(self._triangles_vertices, r.reshape((-1, r.shape[-1])), axis=0)
        self._triangles_vertices_index = np.append(self._triangles_vertices_index, indices, axis=0)
        self._triangles_faces = np.append(self._triangles_faces, faces, axis=0)
        self._triangles_faces_index = np.append(self._triangles_faces_index, indices[::2], axis=0)

    def _add_ellipses(self, ellipses):
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
        n = self._ellipse_segments
        vertices = [self._generate_ellipse(x, n) for x in ellipses]
        indices = m + np.arange(0, n*len(ellipses))//n
        m = len(self._lines_vertices)
        connect = [[[n*i+j, n*i+np.mod(j+1,n)] for j in range(n)] for i in range(len(ellipses))]
        connect = m + np.concatenate(connect, axis=0)
        self._lines_vertices = np.append(self._lines_vertices, np.concatenate([v[1:] for v in vertices], axis=0), axis=0)
        self._lines_vertices_index = np.append(self._lines_vertices_index, indices, axis=0)
        self._lines_connect = np.append(self._lines_connect, connect, axis=0)
        self._lines_connect_index = np.append(self._lines_connect_index, indices, axis=0)
        # Add triangulation data for each ellipse
        vertices = np.concatenate(vertices, axis=0)
        face_indices = indices
        indices = m + np.arange(0, (n+1)*len(ellipses))//(n+1)
        m = len(self._triangles_vertices)
        faces = np.array([[0, i+1, i+2] for i in range(n)])
        faces[-1, 2] = 1
        faces = m + np.concatenate([(n+1)*i + faces for i in range(len(ellipses))]).astype(np.uint32)
        self._triangles_vertices = np.append(self._triangles_vertices, vertices, axis=0)
        self._triangles_vertices_index = np.append(self._triangles_vertices_index, indices, axis=0)
        self._triangles_faces = np.append(self._triangles_faces, faces, axis=0)
        self._triangles_faces_index = np.append(self._triangles_faces_index, face_indices, axis=0)


    def _add_paths(self, paths):
        self.id = np.append(self.id, np.repeat(4, len(paths)), axis=0)
        self.vertices = np.append(self.vertices, np.concatenate(paths, axis=0), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.concatenate([np.repeat(i, len(paths[i])) for i in range(len(paths))])
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in paths])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For paths connect every vertex in each path
        connect_indices = m + np.concatenate([np.repeat(i, len(paths[i]-1)) for i in range(len(paths))])
        m = len(self._lines_vertices)
        offsets = np.concatenate(([0], np.array([len(p) for p in paths]).cumsum()))
        connect = [[[offsets[i]+j, offsets[i]+j+1] for j in range(len(paths[i])-1)] for i in range(len(paths))]
        connect = m + np.concatenate(connect, axis=0)
        self._lines_vertices = np.append(self._lines_vertices, np.concatenate(paths, axis=0), axis=0)
        self._lines_vertices_index = np.append(self._lines_vertices_index, indices, axis=0)
        self._lines_connect = np.append(self._lines_connect, connect, axis=0)
        self._lines_connect_index = np.append(self._lines_connect_index, connect_indices, axis=0)

    def _add_polygons(self, polygons):
        self.id = np.append(self.id, np.repeat(5, len(polygons)), axis=0)
        self.vertices = np.append(self.vertices, np.concatenate(polygons, axis=0), axis=0)
        m = max(self.index, default=-1) + 1
        indices = m + np.concatenate([np.repeat(i, len(polygons[i])) for i in range(len(polygons))])
        self.index = np.append(self.index, indices, axis=0)
        boxes = np.array([self._expand_box(x) for x in polygons])
        self.boxes = np.append(self.boxes, boxes, axis=0)
        # Build objects to be rendered
        # For polygons connect every vertex in each polygon, including loop back to close
        m = len(self._lines_vertices)
        offsets = np.concatenate(([0], np.array([len(p) for p in polygons]).cumsum()))
        connect = [[[offsets[i]+j, offsets[i]+np.mod(j+1,len(polygons[i]))] for j in range(len(polygons[i]))] for i in range(len(polygons))]
        connect = m + np.concatenate(connect, axis=0)
        self._lines_vertices = np.append(self._lines_vertices, np.concatenate(polygons, axis=0), axis=0)
        self._lines_vertices_index = np.append(self._lines_vertices_index, indices, axis=0)
        self._lines_connect = np.append(self._lines_connect, connect, axis=0)
        self._lines_connect_index = np.append(self._lines_connect_index, indices, axis=0)
        # Add triangulation data for each polygon
        data = [PolygonData(vertices=np.array(p, dtype=np.float32)).triangulate() for p in polygons]
        vertices = np.concatenate([d[0] for d in data])
        indices = m + np.concatenate([np.repeat(i, len(data[i][0])) for i in range(len(polygons))])
        face_indices = m + np.concatenate([np.repeat(i, len(data[i][1])) for i in range(len(polygons))])
        m = len(self._triangles_vertices)
        offsets = np.concatenate(([0], np.array([len(d[0]) for d in data]).cumsum()))
        faces = m + np.concatenate([data[i][1]+offsets[i] for i in range(len(polygons))]).astype(np.uint32)
        self._triangles_vertices = np.append(self._triangles_vertices, vertices, axis=0)
        self._triangles_vertices_index = np.append(self._triangles_vertices_index, indices, axis=0)
        self._triangles_faces = np.append(self._triangles_faces, faces, axis=0)
        self._triangles_faces_index = np.append(self._triangles_faces_index, face_indices, axis=0)

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

    def _box_connect(self, i):
        if np.mod(i, 4) == 3:
            return [i, i-3]
        else:
            return [i, i+1]

    def _box_face(self, i):
        if np.mod(i, 2) == 1:
            return [2*(i-1),2*(i-1)+3, 2*(i-1)+2]
        else:
            return [2*i, 2*i+1, 2*i+2]

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
