import numpy as np
import pandas as pd
import networkx as nx

from napari.layers import swc

# read in swc file to dataframe
swc_path = 'napari/layers/swc/tests/2018-08-01_G-002_consensus.swc'
df = swc.read_swc(swc_path)

# convert swc dataframe from spatial units to voxel units
spacing = np.array([0.29875923, 0.3044159, 0.98840415])
origin = np.array([70093.276, 15071.596, 29306.737])
df_voxel = swc.swc_to_voxel(df, spacing=spacing, origin=origin)

# convert from dataframe to directed graph
G = swc.df_to_graph(df_voxel=df_voxel)

# convert directed graph into list of paths
paths = swc.graph_to_paths(G)


def test_read_swc_dataframe():
    """test if output is correct type (pd.DataFrame)"""
    assert isinstance(df, pd.DataFrame)


def test_read_swc_shape():
    """test if output is correct shape"""
    correct_shape = (1650, 7)
    assert df.shape == correct_shape


def test_read_swc_columns():
    """test if columns are correct"""
    col = ['sample', 'structure', 'x', 'y', 'z', 'r', 'parent']
    assert list(df.columns) == col


def test_space_to_voxel_int64():
    """test if output is numpy.array of int"""
    spatial_coord = np.array([73940.221323, 18869.828297, 33732.256716])
    voxel_coord = swc.space_to_voxel(
        spatial_coord=spatial_coord, spacing=spacing, origin=origin
    )
    assert all(isinstance(n, np.int64) for n in voxel_coord)


def test_swc_to_voxel_dataframe():
    """test if output is correct type (pd.DataFrame)"""
    assert isinstance(df_voxel, pd.DataFrame)


def test_swc_to_voxel_shape():
    """test if output is correct shape"""
    correct_shape = (1650, 7)
    assert df_voxel.shape == correct_shape


def test_swc_to_voxel_columns():
    """test if columns are correct"""
    col = ['sample', 'structure', 'x', 'y', 'z', 'r', 'parent']
    assert list(df_voxel.columns) == col


def test_swc_to_voxel_nonnegative():
    """test if coordinates are all nonnegative"""
    coord = df_voxel[['x', 'y', 'z']].values
    assert np.greater_equal(coord, np.zeros(coord.shape)).all()


def test_df_to_graph_digraph():
    """test if output is directed graph"""
    assert isinstance(G, nx.DiGraph)


def test_df_to_graph_nodes():
    """test if graph has correct number of nodes"""
    assert len(G.nodes) == len(df_voxel)


def test_df_to_graph_coordinates():
    """test if graph coordinates are same as that of df_voxel"""
    coord_df = df_voxel[['x', 'y', 'z']].values

    x_dict = nx.get_node_attributes(G, 'x')
    y_dict = nx.get_node_attributes(G, 'y')
    z_dict = nx.get_node_attributes(G, 'z')

    x = [x_dict[i] for i in G.nodes]
    y = [y_dict[i] for i in G.nodes]
    z = [z_dict[i] for i in G.nodes]

    coord_graph = np.array([x, y, z]).T

    assert np.array_equal(coord_graph, coord_df)


def test_get_sub_neuron_digraph():
    """test if output is directed graph"""
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6656])
    G_sub = swc.get_sub_neuron(G, bounding_box=(start, end))
    assert isinstance(G_sub, nx.DiGraph)


def test_get_sub_neuron_bounding_box():
    """test if bounding box produces correct number of nodes and edges"""

    # case 1: bounding box has nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6656])
    G_sub = swc.get_sub_neuron(G, bounding_box=(start, end))
    num_nodes = 308
    num_edges = 287
    assert len(G_sub.nodes) == num_nodes
    assert len(G_sub.edges) == num_edges

    # case 2: bounding box has no nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6448])
    G_sub = swc.get_sub_neuron(G, bounding_box=(start, end))
    assert len(G_sub.nodes) == 0
    assert len(G_sub.edges) == 0


def test_graph_to_paths_length():
    """test if output has correct length"""
    num_branches = 179
    assert len(paths) == num_branches


def test_graph_to_paths_path_dim():
    """test if numpy.arrays have 3 columns"""
    assert all(a.shape[1] == 3 for a in paths)
