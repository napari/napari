import numpy as np
import pandas as pd
import networkx as nx


def read_swc(swc_path):
    """Read a swc file into a pandas dataframe.

    Parameters
    ----------
    swc_path : str
        String representing the path to the swc file
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc.
        Coordinates are in spatial units.
    """

    # check input
    file = open(swc_path, 'r')
    in_header = True
    offset_found = False
    header_length = -1
    offset = np.array([0, 0, 0])
    while in_header:
        line = file.readline().split()
        if 'OFFSET' in line:
            offset_found = True
            idx = line.index('OFFSET') + 1
            offset = [float(line[i]) for i in np.arange(idx, idx + 3)]
        elif line[0] != '#':
            in_header = False
        header_length += 1

    if not offset_found:
        raise IOError('No offset information found in: ' + swc_path)
    # read coordinates
    df = pd.read_table(
        swc_path,
        names=['sample', 'structure', 'x', 'y', 'z', 'r', 'parent'],
        skiprows=header_length,
        delim_whitespace=True,
    )

    # adjust coordinates by offset
    df['x'] = df['x'] + offset[0]
    df['y'] = df['y'] + offset[1]
    df['z'] = df['z'] + offset[2]

    return df


def space_to_voxel(spatial_coord, spacing, origin=np.array([0, 0, 0])):
    """Converts coordinate from spatial units to voxel units.

    Parameters
    ----------
    spatial_coord : :class:`numpy.array`
        3D coordinate in spatial units. Assumed to be np.array[(x,y,z)]
    spacing : :class:`numpy.array`
        Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z])
    origin : :class:`numpy.array`
        Origin of the spatial coordinate. Default is (0,0,0). Assumed to be
        np.array([x,y,z])
    Returns
    -------
    voxel_coord : :class:`numpy.array`
        Coordinate in voxel units. Assumed to be np.array([x,y,z])
    """

    voxel_coord = np.round(np.divide(spatial_coord - origin, spacing))
    voxel_coord = voxel_coord.astype(np.int64)
    return voxel_coord


def swc_to_voxel(df, spacing, origin=np.array([0, 0, 0])):
    """Converts coordinates in pd.DataFrame representing swc from spatial units
    to voxel units

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc. Coordinates
        are in spatial units.
    spacing : :class:`numpy.array`
        Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z])
    origin : :class:`numpy.array`
        Origin of the spatial coordinate. Default is (0,0,0). Assumed to be
        np.array([x,y,z])
    Returns
    -------
    df_voxel : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc. Coordinates
        are in voxel units.
    """
    x = []
    y = []
    z = []
    df_voxel = df.copy()
    for index, row in df_voxel.iterrows():
        vox = space_to_voxel(row[['x', 'y', 'z']].to_numpy(), spacing, origin)
        x.append(vox[0])
        y.append(vox[1])
        z.append(vox[2])

    df_voxel['x'] = x
    df_voxel['y'] = y
    df_voxel['z'] = z

    return df_voxel


def df_to_graph(df_voxel):
    """Converts dataframe of swc in voxel coordinates into a directed graph

    Parameters
    ----------
    df_voxel : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc. Coordinates
        are in voxel units.
    Returns
    -------
    G : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    """
    G = nx.DiGraph()

    # add nodes
    for index, row in df_voxel.iterrows():
        id = int(row['sample'])

        G.add_node(id)
        G.nodes[id]['x'] = int(row['x'])
        G.nodes[id]['y'] = int(row['y'])
        G.nodes[id]['z'] = int(row['z'])

    # add edges
    for index, row in df_voxel.iterrows():
        child = int(row['sample'])
        parent = int(row['parent'])

        if parent > min(df_voxel['parent']):
            G.add_edge(parent, child)

    return G


def get_sub_neuron(G, bounding_box):
    """Returns sub-neuron with node coordinates bounded by start and end

    Parameters
    ----------
    G : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    bounding_box : tuple or list or None
        Defines a bounding box around a sub-region around the neuron. Length 2
        tuple/list. First element is the coordinate of one corner (inclusive) and second element is the coordinate of the opposite corner (exclusive). Both coordinates are numpy.array([x,y,z])in voxel units.
    Returns
    -------
    G_sub : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    """
    G_sub = G.copy()  # make copy of input G
    start = bounding_box[0]
    end = bounding_box[1]

    # remove nodes that are not neighbors of nodes bounded by start and end
    for node in list(G_sub.nodes):
        neighbors = list(G_sub.successors(node)) + list(
            G_sub.predecessors(node)
        )

        remove = True

        for id in neighbors + [node]:
            x = G_sub.nodes[id]['x']
            y = G_sub.nodes[id]['y']
            z = G_sub.nodes[id]['z']

            if x >= start[0] and y >= start[1] and z >= start[2]:
                if x < end[0] and y < end[1] and z < end[2]:
                    remove = False

        if remove:
            G_sub.remove_node(node)

    # set origin to start of bounding box
    for id in list(G_sub.nodes):
        G_sub.nodes[id]['x'] = G_sub.nodes[id]['x'] - start[0]
        G_sub.nodes[id]['y'] = G_sub.nodes[id]['y'] - start[1]
        G_sub.nodes[id]['z'] = G_sub.nodes[id]['z'] - start[2]

    return G_sub


def graph_to_paths(G):
    """Converts neuron represented as a directed graph with no cycles into a
    list of paths.

    Parameters
    ----------
    G : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    Returns
    -------
    paths : list
        List of Nx3 numpy.array. Rows of the array are 3D coordinates in voxel
        units. Each array is one path.
    """
    G_cp = G.copy()  # make copy of input G
    branches = []
    while len(G_cp.edges) != 0:  # iterate over branches
        # get longest branch
        longest = nx.algorithms.dag.dag_longest_path(
            G_cp
        )  # list of nodes on the path
        branches.append(longest)

        # remove longest branch
        for idx, e in enumerate(longest):
            if idx < len(longest) - 1:
                G_cp.remove_edge(longest[idx], longest[idx + 1])

    # convert branches into list of paths
    paths = []
    for branch in branches:
        # get vertices in branch as n by 3 numpy.array; n = length of branches
        path = np.zeros((len(branch), 3), dtype=np.int64)
        for idx, node in enumerate(branch):
            path[idx, 0] = np.int64(G_cp.nodes[node]['x'])
            path[idx, 1] = np.int64(G_cp.nodes[node]['y'])
            path[idx, 2] = np.int64(G_cp.nodes[node]['z'])

        paths.append(path)

    return paths
