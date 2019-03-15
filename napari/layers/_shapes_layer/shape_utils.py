import numpy as np

def inside_triangles(triangles):
    """Checks which triangles contain the origin
    Parameters
    ----------
    boxes : np.ndarray
        Nx3x2 array of N triangles that should be checked
    """

    AB = triangles[:,1,:] - triangles[:,0,:]
    AC = triangles[:,2,:] - triangles[:,0,:]
    BC = triangles[:,2,:] - triangles[:,1,:]

    s_AB = -AB[:,0]*triangles[:,0,1] + AB[:,1]*triangles[:,0,0] >= 0
    s_AC = -AC[:,0]*triangles[:,0,1] + AC[:,1]*triangles[:,0,0] >= 0
    s_BC = -BC[:,0]*triangles[:,1,1] + BC[:,1]*triangles[:,1,0] >= 0

    return np.all(np.array([s_AB != s_AC, s_AB == s_BC]), axis=0)

def inside_boxes(boxes):
    """Checks which boxes contain the origin
    Parameters
    ----------
    boxes : np.ndarray
        Nx8x2 array of N boxes that should be checked
    """

    AB = boxes[:,0] - boxes[:,6]
    AM = boxes[:,0]
    BC = boxes[:,6] - boxes[:,4]
    BM = boxes[:,6]

    ABAM = np.multiply(AB, AM).sum(1)
    ABAB = np.multiply(AB, AB).sum(1)
    BCBM = np.multiply(BC, BM).sum(1)
    BCBC = np.multiply(BC, BC).sum(1)

    c1 = 0 <= ABAM
    c2 = ABAM <= ABAB
    c3 = 0 <= BCBM
    c4 = BCBM <= BCBC

    return np.all(np.array([c1, c2, c3, c4]), axis=0)


def point_to_lines(point, lines):
    """Calculate the distance between a point and line segments. First calculates
    the distance to the infinite line, then checks if the projected point lies
    between the line segment endpoints. If not, calculates distance to the endpoints
    Parameters
    ----------
    point : np.ndarray
        1x2 array of point should be checked
    lines : np.ndarray
        Nx2x2 array of line segments
    """

    # shift and normalize vectors
    lines_vectors = lines[:,1] - lines[:,0]
    point_vectors = point - lines[:,0]
    end_point_vectors = point - lines[:,1]
    norm_lines = np.linalg.norm(lines_vectors, axis=1, keepdims=True)
    reject = (norm_lines==0).squeeze()
    norm_lines[reject] = 1
    unit_lines = lines_vectors / norm_lines

    # calculate distance to line
    line_dist = abs(np.cross(unit_lines, point_vectors))

    # calculate scale
    line_loc = (unit_lines*point_vectors).sum(axis=1)/norm_lines.squeeze()

    # for points not falling inside segment calculate distance to appropriate endpoint
    line_dist[line_loc<0] = np.linalg.norm(point_vectors[line_loc<0], axis=1)
    line_dist[line_loc>1] = np.linalg.norm(end_point_vectors[line_loc>1], axis=1)
    line_dist[reject] = np.linalg.norm(point_vectors[reject], axis=1)
    line_loc[reject] = 0.5

    # calculate closet line
    ind = np.argmin(line_dist)

    return ind, line_loc[ind]

def create_box(data):
    """Finds the bounding box of the shape
    """
    min_val = [data[:,0].min(axis=0), data[:,1].min(axis=0)]
    max_val = [data[:,0].max(axis=0), data[:,1].max(axis=0)]
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    return np.array([tl, (tl+tr)/2, tr, (tr+br)/2, br, (br+bl)/2, bl, (bl+tl)/2, (tl+tr+br+bl)/4])

def expand_box(data):
    """Expands four corner rectangle into bounding box
    """
    return np.array([data[0], (data[0]+data[1])/2, data[1], (data[1]+data[2])/2,
                    data[2], (data[2]+data[3])/2, data[3], (data[3]+data[0])/2,
                    data.mean(axis=0)])

def expand_rectangle(data):
    """Expands two corners into four corner rectangle
    """
    min_val = [data[:,0].min(axis=0), data[:,1].min(axis=0)]
    max_val = [data[:,0].max(axis=0), data[:,1].max(axis=0)]
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    return np.array([tl, tr, br, bl])

def expand_ellipse(data):
    """Expands a center and radius into four corner rectangle
    """
    data = np.array([data[0]+data[1], data[0]-data[1]])
    return expand_rectangle(data)

def generate_ellipse(corners, num_segments):
    center = corners.mean(axis=0)
    xr = abs((corners[0][0] + corners[3][0])/2 - center[0])
    yr = abs((corners[0][1] + corners[1][1])/2 - center[1])

    vertices = np.empty((num_segments + 1, 2), dtype=np.float32)
    theta = np.linspace(0, np.deg2rad(360), num_segments)

    vertices[1:, 0] = center[0] + xr * np.cos(theta)
    vertices[1:, 1] = center[1] + yr * np.sin(theta)

    # set center point to first vertex
    vertices[0] = np.float32([center[0], center[1]])
    return vertices

def triangulate_path(path, closed=False, limit=3, bevel=False):
    # Remove any equal adjacent points
    if len(path) > 2:
        clean_path = np.array([p for i, p in enumerate(path) if i==0 or not np.all(p == path[i-1])])
    else:
        clean_path = path

    if closed:
        if np.all(clean_path[0] == clean_path[-1]) and len(clean_path)>2:
            clean_path = clean_path[:-1]
        full_path = np.concatenate(([clean_path[-1]], clean_path, [clean_path[0]]),axis=0)
        normals = [segment_normal(full_path[i], full_path[i+1]) for i in range(len(clean_path))]
        path_length = [np.linalg.norm(full_path[i]-full_path[i+1]) for i in range(len(clean_path))]
        normals=np.array(normals)
        full_path = np.concatenate((clean_path, [clean_path[0]]),axis=0)
        full_normals = np.concatenate((normals, [normals[0]]),axis=0)
    else:
        full_path = np.concatenate((clean_path, [clean_path[-2]]),axis=0)
        normals = [segment_normal(full_path[i], full_path[i+1]) for i in range(len(clean_path))]
        path_length = [np.linalg.norm(full_path[i]-full_path[i+1]) for i in range(len(clean_path))]
        normals[-1] = -normals[-1]
        normals=np.array(normals)
        full_path = clean_path
        full_normals = np.concatenate(([normals[0]], normals),axis=0)

    miters = np.array([full_normals[i:i+2].mean(axis=0) for i in range(len(full_path))])
    miters = np.array([miters[i]/np.dot(miters[i], full_normals[i])
                      if np.dot(miters[i], full_normals[i]) != 0 else full_normals[i]
                      for i in range(len(full_path))])
    miter_lengths = np.linalg.norm(miters,axis=1)
    miters = 0.5*miters
    vertex_offsets = []
    central_path = []
    triangles = []
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
                triangles.append([0, 1, 2])
                m=m+1
            else:
                vertex_offsets.append(-miters[i])
                vertex_offsets.append(miters[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
        elif i==len(full_path)-1:
            if closed:
                a = vertex_offsets[m+1]
                b = vertex_offsets[1]
                ray = full_path[i] - full_path[i-1]
                if np.cross(a,ray)*np.cross(b,ray)>0:
                    triangles.append([m, m+1, 1])
                    triangles.append([m, 0, 1])
                else:
                    triangles.append([m, m+1, 1])
                    triangles.append([m+1, 0, 1])
            else:
                vertex_offsets.append(-miters[i])
                vertex_offsets.append(miters[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
                a = vertex_offsets[m+1]
                b = vertex_offsets[m+3]
                ray = full_path[i] - full_path[i-1]
                if np.cross(a,ray)*np.cross(b,ray)>0:
                    triangles.append([m, m+1, m+3])
                    triangles.append([m, m+2, m+3])
                else:
                    triangles.append([m, m+1, m+3])
                    triangles.append([m+1, m+2, m+3])
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
            a = vertex_offsets[m+1]
            b = vertex_offsets[m+3]
            ray = full_path[i] - full_path[i-1]
            if np.cross(a,ray)*np.cross(b,ray)>0:
                triangles.append([m, m+1, m+3])
                triangles.append([m, m+2, m+3])
            else:
                triangles.append([m, m+1, m+3])
                triangles.append([m+1, m+2, m+3])
            triangles.append([m+2, m+3, m+4])
            m = m + 3
        else:
            vertex_offsets.append(-miters[i])
            vertex_offsets.append(miters[i])
            central_path.append(full_path[i])
            central_path.append(full_path[i])
            a = vertex_offsets[m+1]
            b = vertex_offsets[m+3]
            ray = full_path[i] - full_path[i-1]
            if np.cross(a,ray)*np.cross(b,ray)>0:
                triangles.append([m, m+1, m+3])
                triangles.append([m, m+2, m+3])
            else:
                triangles.append([m, m+1, m+3])
                triangles.append([m+1, m+2, m+3])
            m = m + 2

    return np.array(central_path), np.array(vertex_offsets), np.array(triangles)

def segment_normal(a, b):
    d = b-a
    normal = np.array([d[1], -d[0]])
    norm = np.linalg.norm(normal)
    if norm==0:
        return np.array([1, 0])
    else:
        return normal/norm
