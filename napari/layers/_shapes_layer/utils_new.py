import numpy as np

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
