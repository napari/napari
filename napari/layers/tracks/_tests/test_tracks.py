import numpy as np

from napari.layers import Tracks


def _circle(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _sphere(r, theta, psi):
    x = r * np.sin(theta) + r * np.cos(psi)
    y = r * np.sin(theta) + r * np.sin(psi)
    z = r * np.cos(theta)
    return x, y, z


def tracks_2d(num_tracks=10):
    """ create 2d+t track data """
    tracks, properties = [], []

    for i in range(num_tracks):

        track = np.zeros((100, 3), dtype=np.float32)
        track[:, 0] = np.arange(track.shape[0])

        r = 50 * np.random.random()
        phase = np.mod(
            track[:, 0] * 0.1 + np.random.random() * np.pi, 2 * np.pi
        )
        x, y = _circle(r, phase)

        track[:, 1] = 50.0 + x
        track[:, 2] = 50.0 + y

        tracks.append(track)
        properties.append(
            {'time': track[:, 0], 'theta': phase.tolist(), 'radius': r}
        )

    return tracks, properties


def tracks_3d(num_tracks=10):
    """ create 3d+t track data """
    tracks, properties = [], []

    for i in range(num_tracks):

        track = np.zeros((100, 4), dtype=np.float32)
        track[:, 0] = np.arange(track.shape[0])

        r = 50 * np.random.random()
        theta = np.mod(
            track[:, 0] * 0.1 + np.random.random() * np.pi, 2 * np.pi
        )
        psi = np.mod(track[:, 0] * 0.1 + np.random.random() * np.pi, 2 * np.pi)
        x, y, z = _sphere(r, theta, psi)

        track[:, 1] = 50.0 + x
        track[:, 2] = 50.0 + y
        track[:, 3] = 50.0 + z

        tracks.append(track)

        properties.append(
            {'time': track[:, 0], 'theta': theta, 'psi': psi, 'radius': r}
        )

    return tracks, properties


def test_tracks_layer_2dt_ndim():
    """Test instantiating Tracks layer, check 2D+t dimensionality."""
    data = [np.zeros((1, 3))]
    layer = Tracks(data)
    assert layer.ndim == 3


def test_tracks_layer_3dt_ndim():
    """Test instantiating Tracks layer, check 3D+t dimensionality."""
    data = [np.zeros((1, 4))]
    layer = Tracks(data)
    assert layer.ndim == 4


def test_2dt_tracks():
    """Test instantiating Tracks layer with 2D+t data."""

    data, properties = tracks_2d()
    layer = Tracks(data, properties=properties)

    assert layer.ndim == 3
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert np.all(
        [np.all(lp == p) for lp, p in zip(layer.properties, properties)]
    )


def test_3dt_tracks():
    """Test instantiating Tracks layer with 3D+t data."""

    data, properties = tracks_3d()
    layer = Tracks(data, properties=properties)

    assert layer.ndim == 4
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert np.all(
        [np.all(lp == p) for lp, p in zip(layer.properties, properties)]
    )
